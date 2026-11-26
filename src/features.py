"""Cleaning and features.

Everything here is available to a homeowner asking "what's my home worth?":
list price, days on market and the list-to-sale ratio are never used.

Market features for a row use only sales that went firm strictly before that
row's sale_date, so a model never sees the future, in training or at
prediction time (``build(query, history=all_sales)`` with today's date).
"""

from __future__ import annotations

import logging
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from . import geo

log = logging.getLogger(__name__)

# Residential types we value. Land, farms, mobile homes and "Other" are out:
# too few sales, and their price is mostly the land.
KEEP_TYPES = {
    "Detached", "Semi-Detached", "Att/Row/Townhouse", "Link", "Duplex", "Triplex", "Fourplex",
    "Condo Apartment", "Condo Townhouse", "Detached Condo", "Semi-Detached Condo", "Common Element Condo",
}
CONDO_TYPES = {"Condo Apartment", "Condo Townhouse", "Detached Condo", "Semi-Detached Condo", "Common Element Condo"}

MIN_PRICE, MAX_PRICE = 75_000, 8_000_000
DUPLICATE_DAYS = 60        # same address sold twice within this many days: one sale
OUTLIER_MAD = 4.0          # robust z-score on log($/sqft) within property type
AREA_DAYS, AREA_MIN = 90, 5
COMP_DAYS, COMP_K, COMP_MIN = 180, 10, 3
EPOCH = pd.Timestamp("2024-01-01")

NUMERIC = [
    "sqft_mid", "beds_above", "beds_below", "beds_total", "baths", "lot_area", "lot_width", "lot_depth",
    "age_years", "basement_finished", "has_basement", "parking_total", "has_ac", "tax_annual",
    "tax_years_old", "condo_fee", "lat", "lng", "dist_downtown_km", "dist_station_km",
    "sale_month", "t_months",
    "garage_spaces", "kitchens", "rooms_above", "fireplace_yn", "family_room_yn", "central_vac_yn",
    "pool_inground", "septic", "well_water",
    "area_ppsf_90d", "area_n_90d", "city_ppsf_90d", "comp_price_med", "comp_ppsf_med", "comp_dist_km", "comp_n",
]
CATEGORICAL = [
    "property_type", "type_group", "city_region", "basement_kind", "garage_type", "style", "heat_type",
    "age_band", "sqft_range", "foundation_kind", "heat_source", "exterior_kind",
]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "log_price"


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

_num = re.compile(r"\d+(?:\.\d+)?")


def range_mid(s, *, top_bump: float = 1.1, bottom: float = 0.85) -> float:
    """'1500-2000' -> 1750, '< 700' -> 595, '5000 +' -> 5500, 'New' -> 0."""
    if s is None or (isinstance(s, float) and np.isnan(s)) or s is pd.NA:
        return np.nan
    s = str(s).strip()
    if s.lower() == "new":
        return 0.0
    nums = [float(x) for x in _num.findall(s)]
    if not nums:
        return np.nan
    if len(nums) >= 2:
        return (nums[0] + nums[1]) / 2
    if "<" in s:
        return nums[0] * bottom
    if "+" in s or ">" in s:
        return nums[0] * top_bump
    return nums[0]


def _basement_kind(s) -> str | None:
    if s is None or s is pd.NA or (isinstance(s, float) and np.isnan(s)):
        return None
    tokens = sorted(t.strip() for t in str(s).split(",") if t.strip())
    return ", ".join(tokens) or None


def _finished(s) -> float:
    if s is None or s is pd.NA or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    s = str(s)
    if "None" in s:
        return 0.0
    if "Unfinished" in s:
        return 0.0
    if "Partially Finished" in s:
        return 0.5
    if "Finished" in s:
        return 1.0
    return np.nan


def _foundation(s) -> str | None:
    """Poured / concrete -> concrete; block; stone; other. Block and stone mostly mean older homes."""
    if s is None or s is pd.NA or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).lower()
    for key, name in (("block", "block"), ("stone", "stone"), ("poured", "concrete"), ("concrete", "concrete"),
                      ("wood", "wood"), ("slab", "slab")):
        if key in s:
            return name
    return "other"


def _has(col: pd.Series, word: str) -> pd.Series:
    return col.map(lambda v: np.nan if v is None or v is pd.NA else float(word.lower() in str(v).lower()))


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Drop rows we can't or shouldn't learn from. Returns (clean, removal log)."""
    report: list[dict] = []

    def drop(mask, step, why):
        n = int(mask.sum())
        report.append({"step": step, "removed": n, "why": why})
        log.info("clean: %-22s removed %6d  (%s)", step, n, why)
        return df[~mask]

    report.append({"step": "input", "removed": 0, "why": f"{len(df)} rows"})
    df = df.copy()
    df["property_type"] = df["property_type"].str.strip()

    df = drop(df["mls_status"].fillna("Sold") != "Sold", "not sold", "closed but not a completed sale (e.g. Deal Fell Through)")
    df = drop(df["close_price"].isna() | df["sale_date"].isna(), "no price or date", "ClosePrice or sale date missing")
    df = drop(~df["property_type"].isin(KEEP_TYPES), "property type", "land, farm, mobile home, other")
    df = drop(df.duplicated("listing_key", keep="last"), "duplicate key", "same ListingKey twice")

    # The same home recorded as two sales a few weeks apart: a relist that sold
    # under a new number, or one sale entered twice. Keep the later record.
    df = df.sort_values(["sale_date", "close_date"])
    addr = (df["address"].fillna("").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
            + "|" + df["postal_code"].fillna("").str.upper().str.replace(" ", ""))
    has_addr = df["address"].notna()
    gap = df.groupby(addr)["sale_date"].diff(-1).abs().dt.days  # days to the next sale of this address
    df = drop(has_addr & (gap <= DUPLICATE_DAYS), "duplicate sale", f"same address sold again within {DUPLICATE_DAYS} days")

    df = drop((df["close_price"] < MIN_PRICE) | (df["close_price"] > MAX_PRICE), "price range",
              f"price under ${MIN_PRICE:,} or over ${MAX_PRICE:,} (typos, parking spots, portfolios)")

    sqft = df["sqft_range"].map(range_mid)
    lppsf = np.log(df["close_price"] / sqft.where(sqft > 0))
    med = lppsf.groupby(df["property_type"]).transform("median")
    mad = (lppsf - med).abs().groupby(df["property_type"]).transform("median") * 1.4826
    z = (lppsf - med) / mad.replace(0, np.nan)
    df = drop(z.abs() > OUTLIER_MAD, "price outlier", f"$/sqft more than {OUTLIER_MAD} robust SDs from its property type")

    report.append({"step": "output", "removed": 0, "why": f"{len(df)} rows"})
    return df.reset_index(drop=True), report


# --------------------------------------------------------------------------
# Property and location features
# --------------------------------------------------------------------------

def property_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["type_group"] = np.where(out["property_type"].isin(CONDO_TYPES), "condo", "freehold")
    out["sqft_mid"] = out["sqft_range"].map(range_mid)
    out["beds_total"] = out["beds_above"] + out["beds_below"].fillna(0)
    ft = np.where(out["lot_units"].fillna("Feet").str.startswith("Met"), 3.28084, 1.0)
    w, d = out["lot_width"] * ft, out["lot_depth"] * ft
    acres = out["lot_units"].fillna("").str.startswith("Acre")
    out["lot_width"] = w.where(~acres)
    out["lot_depth"] = d.where(~acres)
    out["lot_area"] = (w * d).where(~acres & (w > 0) & (d > 0))
    is_condo = out["type_group"] == "condo"
    for c in ("lot_area", "lot_width", "lot_depth"):
        out.loc[is_condo, c] = np.nan
    out["age_band"] = out["approx_age"]
    out["age_years"] = out["approx_age"].map(lambda s: range_mid(s, top_bump=1.2))
    out["basement_kind"] = out["basement"].map(_basement_kind)
    out["basement_finished"] = out["basement"].map(_finished)
    out["has_basement"] = out["basement"].map(
        lambda s: np.nan if s is None or s is pd.NA else float("None" not in str(s)))
    out["has_ac"] = out["cooling"].map(lambda s: np.nan if s is None or s is pd.NA else float("Central Air" in str(s)))
    out["tax_years_old"] = out["sale_date"].dt.year - out["tax_year"]
    out["foundation_kind"] = out["foundation"].map(_foundation)
    out["exterior_kind"] = out["exterior"].map(_basement_kind)  # sorted, comma-joined materials
    for src, dst in (("fireplace", "fireplace_yn"), ("family_room", "family_room_yn"), ("central_vac", "central_vac_yn")):
        out[dst] = out[src].map(lambda v: np.nan if v is None or v is pd.NA else float(v == "Y"))
    out["pool_inground"] = _has(out["pool"], "Inground")
    out["septic"] = _has(out["sewer"], "Septic")
    out["well_water"] = _has(out["water"], "Well")
    out["condo_fee"] = out["condo_fee"].where(is_condo)
    out["sale_month"] = out["sale_date"].dt.month
    out["t_months"] = (out["sale_date"] - EPOCH).dt.days / 30.44
    out["log_price"] = np.log(out["close_price"])
    out["ppsf"] = out["close_price"] / out["sqft_mid"].where(out["sqft_mid"] > 0)
    return out


def location_features(df: pd.DataFrame, stations: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.copy()
    out["dist_downtown_km"] = geo.downtown_km(out["lat"], out["lng"])
    out["dist_station_km"] = geo.nearest_station_km(out["lat"], out["lng"], out["sale_date"], stations)
    return out


# --------------------------------------------------------------------------
# Market features (strictly before each row's sale_date)
# --------------------------------------------------------------------------

def _window_median(q_keys, q_dates, h: pd.DataFrame, key: str, value: str, days: int, min_n: int):
    """Median of h[value] over rows with the same key and date in [d - days, d)."""
    med = np.full(len(q_keys), np.nan)
    cnt = np.zeros(len(q_keys))
    h = h.dropna(subset=[value]).sort_values("sale_date")
    groups = {k: (g["sale_date"].to_numpy(), g[value].to_numpy()) for k, g in h.groupby(key)}
    span = np.timedelta64(days, "D")
    for i, (k, d) in enumerate(zip(q_keys, q_dates)):
        g = groups.get(k)
        if g is None:
            continue
        dates, vals = g
        lo = np.searchsorted(dates, d - span, "left")
        hi = np.searchsorted(dates, d, "left")
        cnt[i] = hi - lo
        if hi - lo >= min_n:
            med[i] = np.median(vals[lo:hi])
    return med, cnt


def _comps(q: pd.DataFrame, h: pd.DataFrame):
    """Nearest COMP_K sales of the same property type in the COMP_DAYS before each sale."""
    n = len(q)
    price, ppsf, dist, cnt = (np.full(n, np.nan) for _ in range(4))
    h = h.dropna(subset=["lat", "lng"]).sort_values("sale_date").reset_index(drop=True)
    h_dates = h["sale_date"].to_numpy()
    h_rad = np.radians(h[["lat", "lng"]].to_numpy())
    h_type = h["property_type"].to_numpy()
    h_group = h["type_group"].to_numpy()
    h_price = h["close_price"].to_numpy()
    h_ppsf = h["ppsf"].to_numpy()
    span = np.timedelta64(COMP_DAYS, "D")

    q = q.reset_index(drop=True)
    has_pt = q["lat"].notna() & q["lng"].notna()
    for d, idx in q[has_pt].groupby("sale_date").groups.items():
        d = np.datetime64(d)
        lo, hi = np.searchsorted(h_dates, d - span, "left"), np.searchsorted(h_dates, d, "left")
        if hi - lo < COMP_MIN:
            continue
        window = np.arange(lo, hi)
        rows = q.loc[idx]
        for (ptype, grp), sub in rows.groupby(["property_type", "type_group"]):
            cand = window[h_type[window] == ptype]
            if len(cand) < COMP_K:
                cand = window[h_group[window] == grp]
            if len(cand) < COMP_MIN:
                continue
            k = min(COMP_K, len(cand))
            tree = BallTree(h_rad[cand], metric="haversine")
            dd, ii = tree.query(np.radians(sub[["lat", "lng"]].to_numpy()), k=k)
            picked = cand[ii]
            at = sub.index.to_numpy()
            price[at] = np.median(h_price[picked], axis=1)
            with warnings.catch_warnings():  # all-NaN rows (no comp has a size) stay NaN
                warnings.simplefilter("ignore", RuntimeWarning)
                ppsf[at] = np.nanmedian(h_ppsf[picked], axis=1)
            dist[at] = dd.mean(axis=1) * geo.EARTH_KM
            cnt[at] = k
    return price, ppsf, dist, cnt


def market_features(q: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Add area and comp features to q using history sales strictly before each q row."""
    out = q.reset_index(drop=True).copy()
    h = history
    dates = out["sale_date"].to_numpy()
    region_key = (out["city_region"].fillna("?") + "|" + out["type_group"]).to_numpy()
    hh = h.assign(_key=h["city_region"].fillna("?") + "|" + h["type_group"])
    out["area_ppsf_90d"], out["area_n_90d"] = _window_median(region_key, dates, hh, "_key", "ppsf", AREA_DAYS, AREA_MIN)
    out["city_ppsf_90d"], _ = _window_median(out["type_group"].to_numpy(), dates, h, "type_group", "ppsf", AREA_DAYS, AREA_MIN)
    out["comp_price_med"], out["comp_ppsf_med"], out["comp_dist_km"], out["comp_n"] = _comps(out, h)
    return out


# --------------------------------------------------------------------------

def build(df: pd.DataFrame, history: pd.DataFrame | None = None, stations: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cleaned sales (or query homes) -> model-ready frame.

    history: past sales for market features (defaults to df itself, which is
    what training wants; each row still only sees earlier sales).
    """
    x = location_features(property_features(df), stations)
    hist = x if history is None else location_features(property_features(history), stations)
    return market_features(x, hist)


def baseline_estimate(x: pd.DataFrame) -> np.ndarray:
    """The comps method: nearby $/sqft times size, or the comps' median price."""
    est = x["comp_ppsf_med"] * x["sqft_mid"]
    est = est.where(est.notna(), x["comp_price_med"])
    est = est.where(est.notna(), x["area_ppsf_90d"] * x["sqft_mid"])
    return est.to_numpy()


# --------------------------------------------------------------------------
# End units (townhouses): the description says so, or the lot is clearly wider
# than the narrowest townhouse lots on the same street (the interior units).
# On Ottawa sales the two agree about 3 times in 4.
# --------------------------------------------------------------------------

TOWN_TYPES = {"Att/Row/Townhouse", "Condo Townhouse"}
END_RX = r"\bend[- ]unit|\bend[- ]town|\bcorner[- ]unit|\bend[- ]row|\bend of (?:the )?row"
END_WIDER_FT = 4.0
STREET_MIN = 4  # townhouse lots on a street before its "narrowest" means anything


def street_key(address: pd.Series) -> pd.Series:
    """'4-560 Prominence Way' -> 'PROMINENCE WAY' (number and unit dropped)."""
    return (address.fillna("").str.upper().str.replace(r"^[\dA-Z-]*\d+[A-Z]?\s+", "", regex=True)
            .str.replace(r"[^A-Z0-9 ]", "", regex=True).str.strip())


def usable_width(w: pd.Series) -> pd.Series:
    return pd.to_numeric(w, errors="coerce").where(lambda v: v.between(10, 80))


def end_unit_features(x: pd.DataFrame) -> pd.DataFrame:
    """end_unit (1/0 for townhouses, missing for other types). Lot widths are
    facts about homes, not prices, so using every sale on the street is fine."""
    out = x.copy()
    town = out["property_type"].isin(TOWN_TYPES)
    said = out["remarks"].fillna("").str.lower().str.contains(END_RX, regex=True) if "remarks" in out \
        else pd.Series(False, index=out.index)
    w = usable_width(out["lot_width"])
    key = street_key(out["address"]) + "|" + out["property_type"].astype(str)
    grp = w.where(town).groupby(key)
    p25 = grp.transform(lambda v: v.quantile(0.25))
    n = grp.transform("count")
    wide = (w - p25 >= END_WIDER_FT) & (n >= STREET_MIN)
    out["end_unit"] = np.where(town, (said | wide).astype(float), np.nan)
    return out


# --------------------------------------------------------------------------
# Condo floor, from the unit number: 1204 -> 12, 305 -> 3, PH2 -> penthouse.
# Ottawa buildings number units floor + two digits; 1-2 digit units are
# usually townhouse-style or ground-level, so they stay unknown.
# --------------------------------------------------------------------------

def unit_of(address: pd.Series) -> pd.Series:
    """'1204-160 George St' -> '1204' (data._address puts the unit first)."""
    return address.fillna("").str.extract(r"^([0-9A-Za-z]+)-\d", expand=False)


def floor_of(unit: pd.Series) -> pd.DataFrame:
    u = unit.fillna("").astype(str).str.upper().str.strip()
    ph = u.str.startswith("PH")
    digits = pd.to_numeric(u.str.extract(r"^(\d{3,4})$", expand=False), errors="coerce")
    floor = (digits // 100).where(digits >= 100)
    return pd.DataFrame({"floor": floor.where(floor.between(1, 45)), "penthouse": ph.astype(float)}, index=unit.index)


def condo_floor_features(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    apt = out["property_type"] == "Condo Apartment"
    f = floor_of(unit_of(out["address"]))
    out["floor"] = f["floor"].where(apt)
    out["penthouse"] = f["penthouse"].where(apt)
    return out


# --------------------------------------------------------------------------
# Extra information: descriptions and photos (research feature sets)
# --------------------------------------------------------------------------

PHOTO_PCA = 16


def photo_columns() -> list[str]:
    from . import photos

    return ["n_photos", *photos.ZS_NAMES, *[f"img_pc_{i}" for i in range(PHOTO_PCA)]]


def add_extras(x: pd.DataFrame, photo_dir: str | None = None) -> pd.DataFrame:
    """Description features (always, when remarks exist) and photo features
    (when photo_dir holds the output of src.photos). Both are unsupervised:
    nothing here sees a price."""
    from . import text

    out = x.copy()
    if "remarks" in out:
        upd = text.update_features(out["remarks"], out["sale_date"].dt.year)
        out[upd.columns] = upd.to_numpy()
        summ = text.summary_features(out["remarks"])
        out[summ.columns] = summ.to_numpy()
    if photo_dir:
        from pathlib import Path

        from sklearn.decomposition import PCA

        from . import photos

        keys, n, emb, zs = photos.load(Path(photo_dir))
        cols = photo_columns()
        p = pd.DataFrame(np.nan, index=range(len(keys)), columns=cols)
        p["n_photos"] = n
        p[photos.ZS_NAMES] = zs
        ok = np.isfinite(emb.astype(float)).all(axis=1) & (n > 0)
        if ok.sum() > PHOTO_PCA:
            p.loc[ok, cols[-PHOTO_PCA:]] = PCA(PHOTO_PCA, random_state=42).fit_transform(emb[ok].astype(float))
        p["listing_key"] = keys
        out = out.drop(columns=[c for c in cols if c in out]).merge(p, on="listing_key", how="left")
    return out
