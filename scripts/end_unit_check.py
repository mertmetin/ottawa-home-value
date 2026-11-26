"""Research record (2026-11-26): townhouse accuracy with website-style inputs,
with and without an end-unit flag, and the lot-size fill. Result on real
sales: the lot fill helps (detached + semi 7.8% -> 6.7%), the end-unit flag
doesn't (townhouses 4.41% -> 4.63%), so only the lot fill is served.

    python -m scripts.end_unit_check --data db
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import evaluate, homeowner, pipeline, train_gbm  # noqa: E402
from src import features as F  # noqa: E402


def website_lots(test: pd.DataFrame, hist: pd.DataFrame, use_end: bool) -> pd.DataFrame:
    """Drop each test home's own lot and fill it the way the API does."""
    out = test.copy()
    for i, r in test.iterrows():
        row = {"property_type": r["property_type"], "lat": r["lat"], "lng": r["lng"],
               "lot_width": np.nan, "lot_depth": np.nan}
        homeowner.fill_lot(row, hist)
        out.at[i, "lot_width"], out.at[i, "lot_depth"] = row["lot_width"], row["lot_depth"]
    w, d = out["lot_width"], out["lot_depth"]
    out["lot_area"] = (w * d).where((w > 0) & (d > 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="db")
    ap.add_argument("--rounds", type=int, default=5000)
    a = ap.parse_args()
    x, _ = pipeline.prepare(a.data)
    x = F.end_unit_features(x)
    train, test, _ = evaluate.splits(x)["recent"]
    towns = test[test["property_type"].isin(F.TOWN_TYPES)]
    old = list(homeowner.FEATURES)
    g_old = train_gbm.fit(train, a.rounds, features=old)
    g_new = train_gbm.fit(train, a.rounds, features=old + ["end_unit"])
    y = towns["close_price"].to_numpy()
    rows = {
        "before: MLS lot (research)": g_old.predict(towns),
        "before: website today (no lot)": g_old.predict(towns.assign(lot_width=np.nan, lot_depth=np.nan, lot_area=np.nan)),
        "before: website (lot from nearby)": g_old.predict(website_lots(towns, train, False)),
        "after: MLS lot + end unit": g_new.predict(towns),
        "after: website + end-unit answer": g_new.predict(website_lots(towns, train, True)),
        "after: website, end unit not answered": g_new.predict(website_lots(towns, train, False)),
    }
    print(f"townhouses in the last 3 months: {len(towns)} ({int(towns['end_unit'].sum())} end units)")
    for name, p in rows.items():
        m = evaluate.metrics(y, p["estimate"].to_numpy())
        e = towns["end_unit"].to_numpy() == 1
        me = evaluate.metrics(y[e], p["estimate"].to_numpy()[e])
        bias_end = np.median((p["estimate"].to_numpy()[e] - y[e]) / y[e]) * 100
        print(f"{name:40s} MdAPE {m['mdape']:.2f}%  within10 {m['within_10']:.1f}%  | end units MdAPE {me['mdape']:.2f}% bias {bias_end:+.1f}%")
    # Detached and semis too: the lot fill helps every house type, not just towns.
    free = test[test["property_type"].isin(["Detached", "Semi-Detached"])]
    yf = free["close_price"].to_numpy()
    for name, q in {"no lot (today)": free.assign(lot_width=np.nan, lot_depth=np.nan, lot_area=np.nan),
                    "lot from nearby": website_lots(free, train, False), "MLS lot": free}.items():
        print(f"detached + semi, {name:16s} MdAPE {evaluate.metrics(yf, g_old.predict(q)['estimate'].to_numpy())['mdape']:.2f}%")
    allp = g_new.predict(test)
    print("all homes, after (MLS inputs): MdAPE %.2f%%" % evaluate.metrics(test["close_price"].to_numpy(), allp["estimate"].to_numpy())["mdape"])


if __name__ == "__main__":
    main()
