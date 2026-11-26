"""Research: does a feature hold up across time, not just in one test window?

Rolling origin: for each of WINDOWS consecutive 2-month windows, train on every
sale before the window and test on the window. Baseline = the served feature
set; candidate = baseline + the feature(s). A feature is worth shipping when it
helps in most windows, not just on average.

    python -m scripts.rolling_check --data db
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import evaluate, homeowner, pipeline, train_gbm  # noqa: E402
from scripts.upgrades_check import FINISHES  # noqa: E402

WINDOWS, MONTHS = 5, 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="db")
    ap.add_argument("--rounds", type=int, default=5000)
    ap.add_argument("--no-tax", action="store_true", help="compare against the model without property tax")
    a = ap.parse_args()
    x, _ = pipeline.prepare(a.data)
    r = x["remarks"].fillna("").str.lower()
    x["no_rear"] = r.str.contains(FINISHES["no_rear"], regex=True).astype(float).where(x["remarks"].notna())
    x["premium_lot"] = r.str.contains(FINISHES["premium_lot"], regex=True).astype(float).where(x["remarks"].notna())
    variants = {"+ no rear neighbours": ["no_rear"], "+ no rear + premium lot": ["no_rear", "premium_lot"]}
    drops = {}
    if a.no_tax:
        # Without the tax features, as the website would be if it stopped asking.
        variants = {}
        drops = {"- property tax": ["tax_annual", "tax_years_old"]}
    last = x["sale_date"].max()
    rows = []
    for w in range(WINDOWS):
        end = last - pd.DateOffset(months=MONTHS * w)
        start = end - pd.DateOffset(months=MONTHS)
        train, test = x[x["sale_date"] <= start], x[(x["sale_date"] > start) & (x["sale_date"] <= end)]
        y = test["close_price"].to_numpy()
        fh = (test["type_group"] == "freehold").to_numpy()
        new = fh & (test["age_years"].to_numpy() <= 15)
        flag = fh & (test["no_rear"].to_numpy() == 1)
        preds = {"baseline": train_gbm.fit(train, a.rounds, features=homeowner.FEATURES).predict(test)["estimate"].to_numpy()}
        for name, extra in variants.items():
            preds[name] = train_gbm.fit(train, a.rounds, features=homeowner.FEATURES + extra).predict(test)["estimate"].to_numpy()
        for name, gone in drops.items():
            feats = [f for f in homeowner.FEATURES if f not in gone]
            preds[name] = train_gbm.fit(train, a.rounds, features=feats).predict(test)["estimate"].to_numpy()
        for name, p in preds.items():
            row = {"window": f"{start.date()}..{end.date()}", "model": name}
            for seg, m in {"all": np.ones(len(y), bool), "freehold": fh, "freehold<=15y": new, "no-rear houses": flag,
                           "condo": ~fh}.items():
                row[seg] = evaluate.metrics(y[m], p[m]).get("mdape", np.nan)
                if seg == "no-rear houses":
                    row["no-rear bias"] = float(np.median((p[m] - y[m]) / y[m]) * 100) if m.sum() else np.nan
                    row["no-rear n"] = int(m.sum())
            rows.append(row)
        print(f"window {start.date()}..{end.date()}: train {len(train)}, test {len(test)}", flush=True)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))
    base = df[df.model == "baseline"].set_index("window")
    for name in [*variants, *drops]:
        v = df[df.model == name].set_index("window")
        for seg in ["all", "freehold", "condo", "freehold<=15y"]:
            d = (v[seg] - base[seg])
            print(f"{name:26s} {seg:15s} mean change {d.mean():+.2f} pts, better in {(d < 0).sum()}/{len(d)} windows")


if __name__ == "__main__":
    main()
