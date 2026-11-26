"""A homeowner's answers -> one model-ready row.

The answers use the keys and values of the questions on
mertmetin.ca/home-valuation (hv_sqft, hv_upd_roof, ...), so the site can
forward its form as is. MLS sub-types ("Att/Row/Townhouse") and the site's
own home types ("Townhome") are both accepted.

The production model is LightGBM on the property, location and market
features plus the update features (roof, windows, furnace...). In training
those come from what the listing description says; here they come from the
owner's answers. Descriptions and photos made the research model a little
better, but a homeowner has neither, so the served model never uses them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from . import features as F
from . import text

FEATURE_SET = "homeowner"
FEATURES = F.FEATURES + text.UPDATE_COLS + ["floor", "penthouse"]

SITE_TYPES = {
    "Detached": "Detached", "Semi-detached": "Semi-Detached", "Semi-Detached": "Semi-Detached",
    "Townhome": "Att/Row/Townhouse", "Townhouse": "Att/Row/Townhouse", "Condo": "Condo Apartment",
}
# "When was it last updated?" -> years ago (the middle of the band)
UPDATE_YEARS = {"lt5": 2.5, "5-10": 7.5, "10-20": 15.0, "gt20": 25.0}
UPDATE_ITEMS = ["roof", "windows", "furnace", "ac", "kitchen", "bath", "flooring"]
BASEMENT = {"Finished": "Full, Finished", "Partially Finished": "Full, Partially Finished",
            "Unfinished": "Full, Unfinished", "None": "None"}
FOUNDATION = {"Poured Concrete": "Poured Concrete", "Concrete Block": "Concrete Block", "Stone": "Stone"}
CONDITION_TLC = {"Needs work"}
CONDITION_TURNKEY = {"Excellent", "Newly renovated"}
CHECKS = ["hv_central_air", "hv_fireplace", "hv_pool", "hv_suite", "hv_septic", "hv_well"]


def _num(v) -> float:
    if v is None or v == "":
        return np.nan
    try:
        return float(str(v).rstrip("+").replace(",", "").replace("$", ""))
    except ValueError:
        return np.nan


def _ok(a: dict, key: str) -> str | None:
    v = a.get(key)
    if v is None:
        return None
    v = str(v).strip()
    return None if v in ("", "unknown") else v


def _to_ft(a: dict) -> float:
    return 3.28084 if str(a.get("lot_units") or "").lower().startswith("met") else 1.0


def property_type(a: dict) -> str:
    t = (_ok(a, "property_type") or "").strip()
    t = SITE_TYPES.get(t, t)
    if t == "Att/Row/Townhouse" and _num(a.get("hv_condo_fee")) > 0:
        t = "Condo Townhouse"
    if t not in F.KEEP_TYPES:
        raise ValueError(f"unknown home type {a.get('property_type')!r}")
    return t


def canonical(a: dict, as_of: pd.Timestamp) -> dict:
    """Answers -> one row in the layout of data.COLUMNS (like a sale with no price)."""
    ptype = property_type(a)
    condo = ptype in F.CONDO_TYPES
    yes = lambda k: "Y" if a.get(k) not in (None, "", 0, "0", False) else "N"
    garage = _num(_ok(a, "hv_garage"))
    row = {
        "listing_key": "query", "mls_status": "Sold", "property_type": ptype,
        "sale_date": as_of, "close_date": as_of, "close_price": np.nan,
        "sqft_range": _ok(a, "hv_sqft"),
        "beds_above": _num(a.get("bedrooms")), "beds_below": _num(a.get("beds_below")),
        "baths": _num(a.get("bathrooms")),
        # From an MLS record when the site has one; fill_lot() covers the rest. Always in feet here.
        "lot_width": _num(a.get("lot_width")) * _to_ft(a), "lot_depth": _num(a.get("lot_depth")) * _to_ft(a),
        "lot_units": "Feet",
        "approx_age": _ok(a, "hv_age"),
        "basement": BASEMENT.get(_ok(a, "hv_basement") or "") if not condo else None,
        "heat_type": _ok(a, "hv_heat_type"), "heat_source": _ok(a, "hv_heat_source"),
        "foundation": FOUNDATION.get(_ok(a, "hv_foundation") or "") if not condo else None,
        "cooling": "Central Air" if yes("hv_central_air") == "Y" else "None",
        "fireplace": yes("hv_fireplace"),
        "pool": "Inground" if yes("hv_pool") == "Y" else "None",
        "kitchens": 2.0 if yes("hv_suite") == "Y" else 1.0,
        "sewer": "Septic" if yes("hv_septic") == "Y" else "Sewer",
        "water": "Well" if yes("hv_well") == "Y" else "Municipal",
        "tax_annual": _num(a.get("hv_tax")) if _num(a.get("hv_tax")) > 0 else np.nan,
        "tax_year": float(as_of.year - 1),
        "condo_fee": _num(a.get("hv_condo_fee")) if condo else np.nan,
        "lat": _num(a.get("lat")), "lng": _num(a.get("lng")),
        "city_region": a.get("city_region"),
    }
    if not np.isnan(garage):
        if ptype == "Condo Apartment":   # the site asks condo owners about parking spots
            row["parking_total"] = garage
            row["garage_spaces"] = garage
            row["garage_type"] = "Underground" if garage > 0 else "None"
        else:
            row["garage_spaces"] = garage
            row["garage_type"] = "Attached" if garage > 0 else "None"
    return row


def update_columns(a: dict, home_age: float) -> dict:
    """The owner's update answers in the form text.update_features gives a description.

    Not answered = a description that doesn't mention it (flag 0, age unknown).
    "Original" = as old as the home.
    """
    out = {c: np.nan for c in text.UPDATE_COLS}
    for item in text.ITEMS:
        out[f"upd_{item}"] = 0.0
    for item in UPDATE_ITEMS:
        v = _ok(a, f"hv_upd_{item}")
        years = UPDATE_YEARS.get(v) if v else None
        if v == "orig" and np.isfinite(home_age):
            years = max(home_age, 0.0)
        if years is not None:
            out[f"age_{item}"] = years
            out[f"upd_{item}"] = float(years <= 10)
    for k in text.CONDITION:
        out[f"txt_{k}"] = 0.0
    # "How would you describe it inside?" -> the description cues the model learned
    cond = _ok(a, "condition")
    if cond in CONDITION_TLC:
        out["txt_tlc"] = 1.0
    elif cond in CONDITION_TURNKEY:
        out["txt_turnkey"] = 1.0
    hwt = _ok(a, "hv_hwt")
    if hwt == "owned":
        out["txt_hwt_owned"] = 1.0
    elif hwt == "rental":
        out["txt_hwt_rental"] = 1.0
    return out


def nearest_region(hist: pd.DataFrame, lat: float, lng: float) -> str | None:
    """A home's neighbourhood (CityRegion) = that of the closest past sale."""
    h = hist.dropna(subset=["lat", "lng", "city_region"])
    if h.empty or not (np.isfinite(lat) and np.isfinite(lng)):
        return None
    d = (h["lat"] - lat) ** 2 + ((h["lng"] - lng) * np.cos(np.radians(lat))) ** 2
    return str(h["city_region"].iloc[int(np.argmin(d.to_numpy()))])


def typical_parking(hist: pd.DataFrame, ptype: str, garage: float) -> float:
    """The site asks about the garage, not total parking (garage + driveway),
    which MLS fills 99% of the time: use what similar homes have."""
    if "parking_total" not in hist:
        return np.nan
    h = hist[hist["property_type"] == ptype]
    if pd.notna(garage):
        same = h[h["garage_spaces"] == garage]
        if len(same) >= 20:
            h = same
    v = h["parking_total"].dropna()
    return float(v.median()) if len(v) else np.nan


LOT_NEAREST = 20      # same-type homes nearby that set a typical lot
LOT_MAX_KM = 1.0


def nearby(hist: pd.DataFrame, ptype: str, lat: float, lng: float,
           n: int = LOT_NEAREST, max_km: float = LOT_MAX_KM) -> pd.DataFrame:
    """The nearest past sales of the same home type, with their distance (km)."""
    if not {"lat", "lng"} <= set(hist) or not (np.isfinite(lat) and np.isfinite(lng)):
        return hist.iloc[:0]
    h = hist[hist["property_type"] == ptype].dropna(subset=["lat", "lng"])
    if h.empty:
        return h
    km = np.hypot(h["lat"] - lat, (h["lng"] - lng) * np.cos(np.radians(lat))) * 111.2
    h = h.assign(km=km.to_numpy())
    return h[h["km"] <= max_km].nsmallest(n, "km")


def nearby_lots(hist: pd.DataFrame, ptype: str, lat: float, lng: float) -> pd.DataFrame:
    """Lot widths and depths (feet) of the nearest same-type homes."""
    if not {"lot_width", "lot_depth"} <= set(hist):
        return hist.iloc[:0]
    h = hist[hist["property_type"] == ptype]
    h = h.assign(lot_width=F.usable_width(h["lot_width"]),
                 lot_depth=pd.to_numeric(h["lot_depth"], errors="coerce").where(lambda v: v.between(20, 400)))
    return nearby(h.dropna(subset=["lot_width"]), ptype, lat, lng)


# --------------------------------------------------------------------------
# Typical answers of similar homes nearby. Used (1) for a skipped or clearly
# wrong tax bill: a blank tax reads to the model like a new build, which pushes
# the estimate up; and (2) as the comparison in "what moved it": "your
# finished basement vs what homes like yours nearby have", not "vs no answer".
# --------------------------------------------------------------------------

TAX_NEAREST, TAX_MAX_KM, TAX_MIN = 30, 2.0, 5
TAX_TOO_LOW = 0.5   # under half the typical bill: probably one installment


def _mode(s: pd.Series):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else None


def typical(hist: pd.DataFrame, ptype: str, lat: float, lng: float, sqft: float) -> dict:
    out = {}
    near = nearby(hist, ptype, lat, lng, n=TAX_NEAREST, max_km=TAX_MAX_KM)
    if near.empty:
        return out
    size = pd.to_numeric(near["sqft_mid"], errors="coerce") if "sqft_mid" in near else pd.Series(np.nan, index=near.index)
    if "tax_annual" in near:
        tax = pd.to_numeric(near["tax_annual"], errors="coerce").where(lambda v: v > 500)
        per_sqft = (tax / size.where(size > 0)).dropna()
        if np.isfinite(sqft) and sqft > 0 and len(per_sqft) >= TAX_MIN:
            # Tax per square foot nearby times this home's size: a bigger home
            # among smaller neighbours still gets a bigger bill.
            out["tax"] = float(per_sqft.median() * sqft)
        elif tax.notna().sum() >= TAX_MIN:
            out["tax"] = float(tax.median())
    if np.isfinite(sqft) and sqft > 0:
        similar = near[(size / sqft).between(0.75, 1.33)]
        if len(similar) >= TAX_MIN:
            near = similar
    if "basement_kind" in near and near["basement_kind"].notna().sum() >= TAX_MIN:
        out["basement"] = {"basement_kind": _mode(near["basement_kind"]),
                           "basement_finished": float(near["basement_finished"].median()),
                           "has_basement": _mode(near["has_basement"])}
    if "garage_spaces" in near and near["garage_spaces"].notna().sum() >= TAX_MIN:
        g = _mode(near["garage_spaces"])
        out["garage"] = {"garage_spaces": g, "garage_type": _mode(near.get("garage_type", pd.Series(dtype=object))),
                         "parking_total": typical_parking(hist, ptype, g)}
    return out


def check_tax(row: dict, typ: dict) -> dict:
    """Fill a skipped tax bill, replace one under half the typical bill.
    Returns what was used, for the lead note."""
    entered = row.get("tax_annual")
    t = typ.get("tax")
    if t is None:
        return {"source": "yours" if np.isfinite(entered) else "none",
                "amount": int(entered) if np.isfinite(entered) else None}
    if not np.isfinite(entered):
        row["tax_annual"] = t
        return {"source": "typical", "amount": int(round(t, -1))}
    if entered < TAX_TOO_LOW * t:
        row["tax_annual"] = t
        return {"source": "replaced", "entered": int(entered), "amount": int(round(t, -1))}
    return {"source": "yours", "amount": int(entered)}


def fill_lot(row: dict, hist: pd.DataFrame) -> bool:
    """The site doesn't ask for the lot: use the typical lot of the nearest
    same-type homes. On recent sales this brings website estimates for
    detached and semi-detached homes from 7.8% to 6.7% median error (6.5%
    with the real lot). Returns whether it filled anything."""
    if row["property_type"] in F.CONDO_TYPES:
        return False
    if np.isfinite(row.get("lot_width", np.nan)):
        return False
    near = nearby_lots(hist, row["property_type"], row["lat"], row["lng"])
    if len(near) < 3:
        return False
    row["lot_width"] = float(near["lot_width"].median())
    row["lot_units"] = "Feet"
    if not np.isfinite(row.get("lot_depth", np.nan)) and near["lot_depth"].notna().sum() >= 3:
        row["lot_depth"] = float(near["lot_depth"].median())
    return True


def build(a: dict, hist: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """One model-ready row. hist: past sales already run through property and
    location features (the nightly snapshot)."""
    as_of = pd.Timestamp(as_of or pd.Timestamp.now().normalize())
    row = canonical(a, as_of)
    if not row.get("city_region"):
        row["city_region"] = nearest_region(hist, row["lat"], row["lng"])
    filled = fill_lot(row, hist)
    typ = typical(hist, row["property_type"], row["lat"], row["lng"], F.range_mid(row["sqft_range"]))
    tax = check_tax(row, typ)
    df = data._normalise(pd.DataFrame([row]))
    x = F.location_features(F.property_features(df))
    x.attrs["lot_filled"] = filled
    x.attrs["typical"] = typ
    x.attrs["tax"] = tax
    f = F.floor_of(pd.Series([str(a.get("unit") or "")]))
    apt = row["property_type"] == "Condo Apartment"
    x["floor"] = float(f["floor"].iloc[0]) if apt else np.nan
    x["penthouse"] = float(f["penthouse"].iloc[0]) if apt else np.nan
    x = F.market_features(x, hist[hist["sale_date"] < as_of])
    if pd.isna(x["parking_total"].iloc[0]):
        x["parking_total"] = typical_parking(hist, x["property_type"].iloc[0], x["garage_spaces"].iloc[0])
    upd = update_columns(a, float(x["age_years"].iloc[0]))
    for c, v in upd.items():
        x[c] = v
    return x


# --------------------------------------------------------------------------
# What each answer is worth: the estimate with that answer swapped for the
# usual one ("Your new roof adds about $12,000"). For tax, basement and garage
# the usual answer is what similar homes nearby have; for updates, condition
# and extras it is "not mentioned", as in most listings. Market features don't
# depend on these answers, so variants only change the answer's own columns.
# --------------------------------------------------------------------------

def _unanswered(x: pd.DataFrame, key: str) -> pd.DataFrame | None:
    v = x.copy()
    if key.startswith("hv_upd_"):
        item = key[len("hv_upd_"):]
        v[f"upd_{item}"], v[f"age_{item}"] = 0.0, np.nan
    elif key == "condition":
        v["txt_tlc"], v["txt_turnkey"] = 0.0, 0.0
    elif key == "hv_hwt":
        v["txt_hwt_owned"], v["txt_hwt_rental"] = 0.0, 0.0
    elif key == "hv_basement":
        b = x.attrs.get("typical", {}).get("basement")
        if not b:
            return None
        for c, val in b.items():
            v[c] = val
    elif key == "hv_garage":
        g = x.attrs.get("typical", {}).get("garage")
        if not g:
            return None
        for c, val in g.items():
            v[c] = val
    elif key == "hv_foundation":
        v["foundation_kind"] = None
    elif key == "hv_age":
        v[["age_years", "age_band"]] = [np.nan, None]
    elif key == "hv_tax":
        t = x.attrs.get("typical", {}).get("tax")
        if t is None:
            return None
        v["tax_annual"] = t
    elif key == "hv_central_air":
        v["has_ac"] = 0.0
    elif key == "hv_fireplace":
        v["fireplace_yn"] = 0.0
    elif key == "hv_pool":
        v["pool_inground"] = 0.0
    elif key == "hv_suite":
        v["kitchens"] = 1.0
    else:
        return None
    return v


EXPLAIN_KEYS = [f"hv_upd_{i}" for i in UPDATE_ITEMS] + [
    "condition", "hv_hwt", "hv_basement", "hv_garage", "hv_foundation", "hv_age", "hv_tax"] + CHECKS[:4]


def answered(a: dict) -> list[str]:
    out = []
    for k in EXPLAIN_KEYS:
        v = a.get(k)
        if k in CHECKS:
            if v not in (None, "", 0, "0", False):
                out.append(k)
        elif _ok(a, k) is not None:
            out.append(k)
    return out


def variants(x: pd.DataFrame, a: dict) -> tuple[list[str], pd.DataFrame]:
    keys, rows = [], []
    for k in answered(a):
        v = _unanswered(x, k)
        if v is not None:
            keys.append(k)
            rows.append(v)
    return keys, (pd.concat(rows, ignore_index=True) if rows else x.iloc[:0])
