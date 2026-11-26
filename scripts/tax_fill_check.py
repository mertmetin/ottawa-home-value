"""Research check (2026-11-26): what a skipped or wrong tax answer does to
website estimates, before and after filling in the typical bill nearby.

    python -m scripts.tax_fill_check --data db
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import evaluate, homeowner, pipeline, train_gbm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="db")
    ap.add_argument("--rounds", type=int, default=5000)
    a = ap.parse_args()
    x, _ = pipeline.prepare(a.data)
    train, test, _ = evaluate.splits(x)["recent"]
    test = test[test["tax_annual"].notna()].copy()
    g = train_gbm.fit(train, a.rounds, features=homeowner.FEATURES)
    hist = train[pipeline.SNAPSHOT_COLS]
    typ = np.array([homeowner.typical(hist, r.property_type, r.lat, r.lng, r.sqft_mid).get("tax", np.nan)
                    for r in test.itertuples()])
    y = test["close_price"].to_numpy()

    def score(name, tax):
        p = g.predict(test.assign(tax_annual=tax))["estimate"].to_numpy()
        m = evaluate.metrics(y, p)
        print(f"{name:44s} MdAPE {m['mdape']:.2f}%  within10 {m['within_10']:.1f}%  bias {np.median((p - y) / y) * 100:+.1f}%")

    real = test["tax_annual"].to_numpy()
    print(f"recent sales with a tax figure: {len(test)}; typical bill found for {np.isfinite(typ).mean():.0%}")
    print(f"typical vs real bill: median gap {np.nanmedian(np.abs(typ - real) / real) * 100:.0f}%")
    score("real tax bill", real)
    score("skipped, before (blank)", np.full(len(test), np.nan))
    score("skipped, after (typical nearby)", np.where(np.isfinite(typ), typ, np.nan))
    quarter = real / 4
    score("one quarterly installment, before", quarter)
    score("one quarterly installment, after (replaced)", np.where(np.isfinite(typ) & (quarter < 0.5 * typ), typ, quarter))


if __name__ == "__main__":
    main()
