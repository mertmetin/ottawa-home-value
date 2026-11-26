"""Generate data/sample.csv: SYNTHETIC Ottawa sales in the canonical layout.

No real listing, address or price is in this file. Neighbourhood names are
public district names; everything else (streets, keys, prices, features) is
drawn from a made-up pricing model with realistic shapes: seasonal volume, a
soft market, location premiums, size and age effects, missing fields at
roughly the real fill rates, and a few planted duplicates and typos so the
cleaning step has something to remove.

    python scripts/make_sample.py            # 6000 rows -> data/sample.csv
    python scripts/make_sample.py --rows 20000 --out /tmp/big.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# name, lat, lng, log-price premium, share of condos
REGIONS = [
    ("4001 - Lower Town/Byward Market", 45.4300, -75.6930, 0.05, 0.85),
    ("4103 - Ottawa Centre", 45.4170, -75.7000, 0.08, 0.80),
    ("4003 - Sandy Hill", 45.4235, -75.6800, 0.06, 0.55),
    ("4401 - Glebe", 45.4010, -75.6880, 0.30, 0.25),
    ("5003 - Westboro/Hampton Park", 45.3920, -75.7520, 0.28, 0.35),
    ("3402 - Vanier", 45.4380, -75.6620, -0.08, 0.35),
    ("3701 - Alta Vista", 45.3880, -75.6600, 0.10, 0.20),
    ("2602 - Riverside South/Gloucester Glen", 45.2770, -75.6900, 0.02, 0.10),
    ("2605 - Blossom Park/Kemp Park/Findlay Creek", 45.3200, -75.6150, -0.02, 0.12),
    ("2013 - Mer Bleue/Bradley Estates/Anderson Park", 45.4300, -75.5350, -0.03, 0.10),
    ("1118 - Avalon/Notting Gate/Fallingbrook", 45.4600, -75.4950, -0.04, 0.12),
    ("1105 - Fallingbrook/Pineview", 45.4740, -75.5140, -0.03, 0.12),
    ("7711 - Barrhaven - Half Moon Bay", 45.2630, -75.7570, -0.03, 0.08),
    ("7704 - Barrhaven - Heritage Park", 45.2780, -75.7380, -0.02, 0.10),
    ("9010 - Kanata - Emerald Meadows/Trailwest", 45.2990, -75.8950, -0.01, 0.10),
    ("9008 - Kanata - Morgan's Grant/South March", 45.3420, -75.9290, 0.04, 0.08),
    ("8211 - Stittsville (North)", 45.2700, -75.9190, 0.00, 0.06),
    ("7301 - Nepean - Centrepointe", 45.3440, -75.7650, 0.03, 0.30),
    ("4601 - Hintonburg/Mechanicsville", 45.4050, -75.7300, 0.15, 0.45),
    ("4801 - Old Ottawa South", 45.3930, -75.6830, 0.25, 0.15),
    ("2201 - Orleans - Convent Glen", 45.4690, -75.5450, -0.02, 0.15),
    ("2501 - Hunt Club", 45.3420, -75.6800, 0.00, 0.25),
]

# property type -> (share of freehold or condo sales, median price at 1500 sqft, typical sqft)
FREEHOLD = {
    "Detached": (0.58, 790_000, 1900), "Att/Row/Townhouse": (0.24, 590_000, 1550),
    "Semi-Detached": (0.10, 640_000, 1450), "Link": (0.02, 690_000, 1600),
    "Duplex": (0.04, 820_000, 2100), "Triplex": (0.02, 930_000, 2800),
}
CONDO = {
    "Condo Apartment": (0.64, 520_000, 850), "Condo Townhouse": (0.32, 470_000, 1250),
    "Common Element Condo": (0.04, 500_000, 1400),
}
FREEHOLD_BINS = [0, 700, 1100, 1500, 2000, 2500, 3000, 3500, 5000]
CONDO_BINS = [0, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000]
AGES = ["New", "0-5", "6-15", "16-30", "31-50", "51-99", "100+"]
STYLES_FH = ["2-Storey", "Bungalow", "3-Storey", "1 1/2 Storey", "Bungalow-Raised", "Split Level"]
STREETS = ["Maple", "Cedar", "Birch", "Willow", "Spruce", "Elm", "Aspen", "Juniper", "Tamarack", "Hemlock",
           "Larch", "Poplar", "Sumac", "Alder", "Basswood", "Hawthorn", "Linden", "Sycamore"]
SUFFIXES = ["St", "Ave", "Cres", "Dr", "Way", "Crt", "Pl", "Rd"]


def _bin_label(v: float, bins: list[int]) -> str:
    if v < bins[1]:
        return f"< {bins[1]}"
    for lo, hi in zip(bins[1:], bins[2:]):
        if v < hi:
            return f"{lo}-{hi - (1 if bins is CONDO_BINS else 0)}"
    return f"{bins[-1]} +"


def market_index(t_months: np.ndarray) -> np.ndarray:
    """Log price level: a soft market drifting down ~3% over two years, spring bump."""
    return -0.015 * t_months / 12 + 0.025 * np.sin(2 * np.pi * (t_months - 1.5) / 12)


def generate(rows: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2024-12-01"), pd.Timestamp("2026-11-20")
    days = pd.date_range(start, end, freq="D")
    season = 1 + 0.45 * np.sin(2 * np.pi * (days.day_of_year.to_numpy() - 60) / 365)  # spring peak
    weekday = np.where(days.day_of_week.to_numpy() >= 5, 0.35, 1.0)
    w = season * weekday
    sale_dates = rng.choice(days, size=rows, p=w / w.sum())

    reg_idx = rng.integers(0, len(REGIONS), size=rows)
    recs = []
    for i in range(rows):
        name, rlat, rlng, prem, condo_share = REGIONS[reg_idx[i]]
        is_condo = rng.random() < condo_share
        table = CONDO if is_condo else FREEHOLD
        types = list(table)
        ptype = rng.choice(types, p=np.array([table[t][0] for t in types]) / sum(table[t][0] for t in types))
        _, p1500, typical = table[ptype]
        sqft = float(np.clip(rng.lognormal(np.log(typical), 0.25), 450, 6000))
        lat, lng = rlat + rng.normal(0, 0.008), rlng + rng.normal(0, 0.011)
        age_band = rng.choice(AGES, p=[0.05, 0.10, 0.17, 0.25, 0.23, 0.17, 0.03])
        age = {"New": 0, "0-5": 3, "6-15": 10, "16-30": 23, "31-50": 40, "51-99": 70, "100+": 110}[age_band]
        beds_above = int(np.clip(round(sqft / 600 + rng.normal(0, 0.6)), 1, 6))
        beds_below = int(rng.random() < (0.0 if is_condo else 0.3))
        baths = int(np.clip(round(1 + sqft / 1100 + rng.normal(0, 0.5)), 1, 6))
        if is_condo:
            basement = "None" if ptype == "Condo Apartment" else rng.choice(["Full, Finished", "Finished", "None", "Full, Unfinished"])
            garage = rng.choice(["Underground", "Surface", "None", "Attached"], p=[0.5, 0.2, 0.15, 0.15]) \
                if ptype == "Condo Apartment" else rng.choice(["Attached", "Surface", "None"], p=[0.6, 0.3, 0.1])
            style = "Apartment" if ptype == "Condo Apartment" else rng.choice(["2-Storey", "Stacked Townhouse", "3-Storey"])
            lot_w = lot_d = None
            parking = int(rng.random() < 0.8)
        else:
            basement = rng.choice(["Full, Finished", "Finished", "Full, Unfinished", "Partially Finished", "Full", "Crawl Space"],
                                  p=[0.35, 0.2, 0.18, 0.12, 0.1, 0.05])
            garage = rng.choice(["Attached", "Detached", "None", "Built-In", "Carport"], p=[0.62, 0.12, 0.18, 0.05, 0.03])
            style = rng.choice(STYLES_FH, p=[0.55, 0.2, 0.08, 0.07, 0.05, 0.05])
            lot_w = round(float(np.clip(rng.normal(36 if ptype == "Detached" else 22, 8), 14, 120)), 2)
            lot_d = round(float(np.clip(rng.normal(105, 15), 60, 250)), 2)
            parking = int(np.clip(round(rng.normal(3 if ptype == "Detached" else 2, 1)), 1, 8))

        # Hidden condition (renovations, upkeep): moves the price, and the
        # description and photos reveal it only partly, as in real listings.
        condition = rng.normal(0, 1)
        sale_year = pd.Timestamp(sale_dates[i]).year
        roof_year = int(sale_year - rng.integers(0, 25))
        furnace_year = int(sale_year - rng.integers(0, 25))
        roof_new = float(sale_year - roof_year <= 5 and not is_condo)
        furnace_new = float(sale_year - furnace_year <= 5)
        detached = ptype in ("Detached", "Link", "Duplex", "Triplex")
        foundation = None if ptype == "Condo Apartment" else (
            rng.choice(["Concrete Block", "Stone"], p=[0.8, 0.2]) if age >= 40 and rng.random() < 0.5
            else rng.choice(["Poured Concrete", "Concrete"]))
        heat_source = "Electric" if ptype == "Condo Apartment" and rng.random() < 0.4 else (
            "Oil" if age >= 40 and rng.random() < 0.08 else rng.choice(["Gas", "Propane", "Electric"], p=[0.94, 0.03, 0.03]))
        garage_spaces = (2.0 if detached and rng.random() < 0.55 else 1.0) if garage in ("Attached", "Built-In", "Detached") else 0.0
        kitchens = 2 if not is_condo and rng.random() < 0.04 else 1
        fireplace = "Y" if rng.random() < (0.55 if not is_condo else 0.15) else "N"
        pool = "Inground" if detached and rng.random() < 0.05 else "None"
        rural = not is_condo and rng.random() < 0.05
        dist_dt = np.hypot((lat - 45.4236) * 111, (lng + 75.7009) * 78)
        t = (pd.Timestamp(sale_dates[i]) - pd.Timestamp("2024-01-01")).days / 30.44
        lp = (np.log(p1500) + (0.62 if is_condo else 0.72) * np.log(sqft / 1500) + prem
              + 0.05 * beds_below + 0.035 * (baths - 2)
              + (0.05 if "Finished" in basement and "Unfinished" not in basement else 0)
              + (0.03 if garage in ("Attached", "Built-In") else 0)
              + (0.07 * np.log((lot_w * lot_d) / 4000) if lot_w else 0)
              - 0.0025 * min(age, 60) + (0.06 if age >= 100 else 0)
              - 0.006 * min(dist_dt, 25)
              + market_index(np.array([t]))[0]
              + 0.06 * condition + 0.02 * roof_new + 0.015 * furnace_new
              - (0.04 if heat_source == "Oil" else 0) - (0.02 if foundation == "Concrete Block" else 0)
              + 0.035 * (garage_spaces >= 2) + 0.04 * (kitchens == 2) + 0.01 * (fireplace == "Y")
              + 0.025 * (pool == "Inground") - 0.03 * rural
              + rng.normal(0, 0.055))
        price = float(np.round(np.exp(lp), -3))
        assessed = price * rng.normal(0.72, 0.05)
        tax_year = int(rng.choice([pd.Timestamp(sale_dates[i]).year - 1, pd.Timestamp(sale_dates[i]).year], p=[0.7, 0.3]))
        bits = [rng.choice(["Welcome home!", "Lovely family home.", "Great location.", "Bright and spacious."])]
        if condition > 0.9:
            bits.append(rng.choice(["Fully renovated kitchen and bathrooms.", "Move-in ready, shows beautifully.",
                                    "Updated kitchen with quartz counters."]))
        elif condition < -1.0:
            bits.append(rng.choice(["Needs some TLC.", "Handyman special, sold as-is.", "Original kitchen."]))
        if roof_new and rng.random() < 0.6:
            bits.append(f"New roof {roof_year}.")
        if furnace_new and rng.random() < 0.5:
            bits.append(f"Furnace & A/C ({furnace_year}).")
        bits.append(rng.choice(["HWT rental.", "Owned hot water tank.", ""]))
        recs.append({
            "remarks": " ".join(b for b in bits if b),
            "foundation": foundation,
            "heat_source": heat_source,
            "garage_spaces": garage_spaces,
            "kitchens": kitchens,
            "rooms_above": beds_above + int(rng.integers(2, 5)),
            "fireplace": fireplace,
            "family_room": "Y" if rng.random() < 0.6 else "N",
            "central_vac": "Y" if rng.random() < 0.08 else "N",
            "pool": pool,
            "sewer": "Septic" if rural else "Sewer",
            "water": "Well" if rural else "Municipal",
            "exterior": None if is_condo else rng.choice(["Brick, Vinyl Siding", "Brick", "Vinyl Siding", "Stone, Vinyl Siding"]),
            "_condition": condition,
            "listing_key": f"S{100000 + i}",
            "mls_status": "Sold",
            "property_type": ptype,
            "sale_date": pd.Timestamp(sale_dates[i]).date().isoformat(),
            "close_date": (pd.Timestamp(sale_dates[i]) + pd.Timedelta(days=int(rng.integers(20, 120)))).date().isoformat(),
            "close_price": price,
            "sqft_range": _bin_label(sqft, CONDO_BINS if is_condo else FREEHOLD_BINS),
            "beds_above": beds_above,
            "beds_below": beds_below,
            "baths": baths,
            "lot_width": lot_w,
            "lot_depth": lot_d,
            "lot_units": None if is_condo else "Feet",
            "approx_age": age_band,
            "basement": basement,
            "garage_type": garage,
            "parking_total": parking,
            "style": style,
            "heat_type": rng.choice(["Forced Air", "Baseboard", "Heat Pump", "Radiant"],
                                    p=[0.45, 0.35, 0.15, 0.05] if is_condo else [0.88, 0.04, 0.06, 0.02]),
            "cooling": rng.choice(["Central Air", "None", "Wall Unit(s)"], p=[0.8, 0.12, 0.08]),
            "tax_annual": round(assessed * 0.0108 * (1.03 ** (tax_year - 2024)), 2),
            "tax_year": tax_year,
            "condo_fee": round(sqft * rng.normal(0.58, 0.12), 2) if is_condo else None,
            "lat": round(lat, 6),
            "lng": round(lng, 6),
            "city_region": name,
            "address": f"{int(rng.integers(1, 2500))} {rng.choice(STREETS)} {rng.choice(SUFFIXES)}"
                       + (f" #{int(rng.integers(101, 2400))}" if ptype == "Condo Apartment" else ""),
            "postal_code": f"K{rng.integers(1, 5)}{'ABCEGHJKLMNPRSTVXY'[rng.integers(0, 18)]} {rng.integers(1, 9)}"
                           f"{'ABCEGHJKLMNPRSTVXY'[rng.integers(0, 18)]}{rng.integers(1, 9)}",
        })
    df = pd.DataFrame(recs)

    # Missing fields at roughly the real fill rates.
    def blank(col, frac, mask=None):
        m = rng.random(len(df)) < frac
        if mask is not None:
            m &= mask
        df.loc[m, col] = None

    blank("approx_age", 0.75)
    blank("beds_below", 0.80)
    blank("sqft_range", 0.05)
    blank("lot_width", 0.25)
    df.loc[df["lot_width"].isna(), "lot_depth"] = None
    blank("tax_annual", 0.08)
    blank("condo_fee", 0.35)
    blank("garage_type", 0.05)
    blank("basement", 0.05)
    blank("lat", 0.02)
    df.loc[df["lat"].isna(), "lng"] = None

    # Planted problems for the cleaning step.
    n = len(df)
    dup = df.sample(frac=0.01, random_state=seed).copy()  # same home recorded again weeks later
    dup["listing_key"] = [f"D{j}" for j in range(len(dup))]
    dup["sale_date"] = (pd.to_datetime(dup["sale_date"]) + pd.Timedelta(days=21)).dt.date.astype(str)
    typo = rng.choice(n, size=int(0.004 * n), replace=False)
    df.loc[typo, "close_price"] = df.loc[typo, "close_price"] * rng.choice([0.1, 10], size=len(typo))
    df.loc[rng.choice(n, size=int(0.01 * n), replace=False), "mls_status"] = "Deal Fell Through"
    land = rng.choice(n, size=int(0.01 * n), replace=False)
    df.loc[land, "property_type"] = "Vacant Land"
    df = pd.concat([df, dup], ignore_index=True).sort_values("sale_date").reset_index(drop=True)
    return df


def synthetic_photos(df: pd.DataFrame, out: Path, seed: int = 7) -> None:
    """Fake src.photos output: condition scores that track the hidden condition
    with noise, random embeddings. Exercises the photo path without images."""
    import sys

    sys.path.insert(0, str(ROOT))
    from src import photos

    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("chunk_*.npz"):
        f.unlink()
    n = len(df)
    cond = df["_condition"].to_numpy()
    zs = rng.normal(0, 1.5, size=(n, len(photos.ZS_NAMES)))
    for j, name in enumerate(photos.ZS_NAMES):
        if any(k in name for k in ("kitchen", "bath", "living", "luxury")):
            zs[:, j] += 1.2 * cond
        if "needs_work" in name:
            zs[:, j] -= 1.2 * cond
    emb = rng.normal(0, 1, size=(n, 512))
    emb[:, 0] += cond
    np.savez(out / "chunk_00000.npz", keys=df["listing_key"].to_numpy().astype(str), n=np.full(n, 8),
             emb=emb.astype(np.float16), zs=zs.astype(np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "data" / "sample.csv"))
    a = ap.parse_args()
    df = generate(a.rows, a.seed)
    synthetic_photos(df, ROOT / "data" / "sample_photos", a.seed)
    df = df.drop(columns="_condition")
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} synthetic rows to {a.out}")


if __name__ == "__main__":
    main()
