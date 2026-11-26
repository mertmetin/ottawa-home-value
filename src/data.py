"""Load sold listings into one canonical table.

Two sources produce the same columns:

* ``load_csv`` reads a CSV in the canonical layout (``data/sample.csv`` is a
  synthetic example anyone can run).
* ``load_db`` reads the ``sold_listings`` archive of the listing engine
  (Postgres, PropTx VOW data). That data is licensed: it is read on the
  server and never written into this repository.

Canonical columns (one row per sale):

    listing_key, mls_status, property_type, sale_date, close_date, close_price,
    sqft_range, beds_above, beds_below, baths, lot_width, lot_depth, lot_units,
    approx_age, basement, garage_type, parking_total, style, heat_type, cooling,
    tax_annual, tax_year, condo_fee, lat, lng, city_region, address, postal_code

``sale_date`` is the day the sale went firm (PropTx PurchaseContractDate),
falling back to the closing day when the firm date is missing. Closing can be
months after the sale, so market features are keyed on ``sale_date``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "listing_key", "mls_status", "property_type", "sale_date", "close_date", "close_price",
    "sqft_range", "beds_above", "beds_below", "baths", "lot_width", "lot_depth", "lot_units",
    "approx_age", "basement", "garage_type", "parking_total", "style", "heat_type", "cooling",
    "tax_annual", "tax_year", "condo_fee", "lat", "lng", "city_region", "address", "postal_code",
]

NUMERIC = ["close_price", "beds_above", "beds_below", "baths", "lot_width", "lot_depth",
           "parking_total", "tax_annual", "tax_year", "condo_fee", "lat", "lng"]


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    for c in COLUMNS:
        if c not in df:
            df[c] = np.nan
    df = df[COLUMNS].copy()
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("sale_date", "close_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["sale_date"] = df["sale_date"].fillna(df["close_date"])
    text = [c for c in COLUMNS if c not in NUMERIC and c not in ("sale_date", "close_date")]
    for c in text:
        df[c] = df[c].astype("string").str.strip().replace("", pd.NA)
    return df.reset_index(drop=True)


def load_csv(path: str | os.PathLike) -> pd.DataFrame:
    return _normalise(pd.read_csv(path, dtype=str))


def _joined(v) -> str | None:
    """PropTx sends some fields as JSON arrays (["Full", "Finished"])."""
    if v is None:
        return None
    if isinstance(v, list):
        return ", ".join(str(x) for x in v if x not in (None, "")) or None
    return str(v)


def _address(r: dict) -> str | None:
    parts = [r.get("StreetNumber"), r.get("StreetName"), r.get("StreetSuffix"), r.get("StreetDirSuffix")]
    s = " ".join(str(p) for p in parts if p)
    if not s:
        return None
    unit = r.get("UnitNumber")
    return f"{unit}-{s}" if unit else s


def from_record(row: dict) -> dict:
    """Map one sold_listings row (columns + the RESO record) to canonical fields."""
    r = row["record"]
    if isinstance(r, str):
        r = json.loads(r)
    return {
        "listing_key": row["listing_key"],
        "mls_status": row.get("mls_status") or r.get("MlsStatus"),
        "property_type": (row.get("property_sub_type") or r.get("PropertySubType") or "").strip() or None,
        "sale_date": row.get("contract_date") or r.get("PurchaseContractDate"),
        "close_date": row.get("close_date") or r.get("CloseDate"),
        "close_price": row.get("close_price") or r.get("ClosePrice"),
        "sqft_range": r.get("LivingAreaRange"),
        "beds_above": r.get("BedroomsAboveGrade"),
        "beds_below": r.get("BedroomsBelowGrade"),
        "baths": r.get("BathroomsTotalInteger"),
        "lot_width": r.get("LotWidth"),
        "lot_depth": r.get("LotDepth"),
        "lot_units": r.get("LotSizeUnits"),
        "approx_age": r.get("ApproximateAge"),
        "basement": _joined(r.get("Basement")),
        "garage_type": r.get("GarageType"),
        "parking_total": r.get("ParkingTotal"),
        "style": _joined(r.get("ArchitecturalStyle")),
        "heat_type": r.get("HeatType"),
        "cooling": _joined(r.get("Cooling")),
        "tax_annual": r.get("TaxAnnualAmount"),
        "tax_year": r.get("TaxYear"),
        "condo_fee": r.get("AssociationFee"),
        "lat": row.get("lat"),
        "lng": row.get("lng"),
        "city_region": r.get("CityRegion"),
        "address": _address(r),
        "postal_code": r.get("PostalCode"),
    }


def load_db(dsn: str | None = None, county: str = "Ottawa") -> pd.DataFrame:
    """Read the listing engine's sold archive. DSN from DATABASE_URL by default."""
    import psycopg  # optional dependency: pip install .[db]

    dsn = dsn or os.environ["DATABASE_URL"]
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT listing_key, mls_status, property_sub_type, close_price::float8 AS close_price,
                      contract_date, close_date, lat, lng, record
               FROM sold_listings WHERE county = %s""",
            (county,),
        )
        names = [d.name for d in cur.description]
        rows = [from_record(dict(zip(names, t))) for t in cur.fetchall()]
    return _normalise(pd.DataFrame(rows))


def load(source: str) -> pd.DataFrame:
    """``db`` reads DATABASE_URL; anything else is a CSV path."""
    if source == "db":
        return load_db()
    return load_csv(Path(source))
