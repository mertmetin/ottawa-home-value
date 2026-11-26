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
FEATURES = F.FEATURES + text.UPDATE_COLS

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
        "lot_width": _num(a.get("lot_width")), "lot_depth": _num(a.get("lot_depth")), "lot_units": a.get("lot_units"),
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


def build(a: dict, hist: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """One model-ready row. hist: past sales already run through property and
    location features (the nightly snapshot)."""
    as_of = pd.Timestamp(as_of or pd.Timestamp.now().normalize())
    row = canonical(a, as_of)
    if not row.get("city_region"):
        row["city_region"] = nearest_region(hist, row["lat"], row["lng"])
    df = data._normalise(pd.DataFrame([row]))
    x = F.location_features(F.property_features(df))
    x = F.market_features(x, hist[hist["sale_date"] < as_of])
    if pd.isna(x["parking_total"].iloc[0]):
        x["parking_total"] = typical_parking(hist, x["property_type"].iloc[0], x["garage_spaces"].iloc[0])
    x.attrs["parking_unknown_garage"] = typical_parking(hist, x["property_type"].iloc[0], np.nan)
    upd = update_columns(a, float(x["age_years"].iloc[0]))
    for c, v in upd.items():
        x[c] = v
    return x


# --------------------------------------------------------------------------
# What each answer is worth: the estimate without it, for "Your new roof adds
# about $12,000". Market features don't depend on these answers, so variants
# only change the answer's own columns.
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
        v[["basement_kind", "basement_finished", "has_basement"]] = [None, np.nan, np.nan]
    elif key == "hv_garage":
        v[["garage_spaces", "garage_type"]] = [np.nan, None]
        v["parking_total"] = x.attrs.get("parking_unknown_garage", np.nan)
    elif key == "hv_foundation":
        v["foundation_kind"] = None
    elif key == "hv_age":
        v[["age_years", "age_band"]] = [np.nan, None]
    elif key == "hv_tax":
        v[["tax_annual", "tax_years_old"]] = [np.nan, np.nan]
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
