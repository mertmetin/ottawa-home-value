"""Research: do builder upgrades / premium finishes in the description move
the price, and does a feature for them make estimates more accurate,
for newer homes vs older ones?

    python -m scripts.upgrades_check --data db
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import evaluate, homeowner, pipeline, train_gbm  # noqa: E402
from src import features as F  # noqa: E402

FINISHES = {
    "builder_upg": r"builder[- ]upgrade|upgrades? (?:from|by|through) (?:the )?builder|in (?:builder )?upgrades|upgraded (?:by|from) (?:the )?builder|thousands in upgrades|upgrades? galore|tons of upgrades|loaded with upgrades",
    "quartz": r"\bquartz\b|\bgranite\b",
    "ceilings9": r"\b(?:9|nine|10|ten)\s?(?:ft|foot|feet|')\s?ceil",
    "hardwood": r"\bhardwood\b|engineered (?:wood|hardwood)",
    "no_rear": r"no (?:rear|back(?:yard)?) neighbou?rs|backs? (?:on)?to (?:a |the )?(?:park|green ?space|forest|woods|pond|ravine|conservation|trees|ncc|golf)|backing onto (?:a |the )?(?:park|green ?space|forest|woods|pond|ravine|conservation|trees|ncc|golf)",
    "premium_lot": r"premium lot|pie[- ]shaped lot|oversized lot|corner lot",
    "walkout": r"walk[- ]?out (?:basement|lower)",
    "pot_lights": r"pot lights",
}


def flags(remarks: pd.Series) -> pd.DataFrame:
    r = remarks.fillna("").str.lower()
    out = pd.DataFrame({f"fin_{k}": r.str.contains(rx, regex=True).astype(float) for k, rx in FINISHES.items()}, index=r.index)
    out["fin_count"] = out.sum(axis=1)
    out.loc[remarks.isna()] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="db")
    ap.add_argument("--rounds", type=int, default=5000)
    a = ap.parse_args()
    x, _ = pipeline.prepare(a.data)
    f = flags(x["remarks"])
    x[f.columns] = f.to_numpy()
    houses = x[x["type_group"] == "freehold"].copy()
    houses["age_grp"] = np.select([houses["age_years"] <= 15, houses["age_years"] > 15], ["new (<=15 yrs)", "older"], "age unknown")
    q = houses["sale_date"].dt.to_period("Q").astype(str)
    g = houses.groupby([houses["property_type"], houses["city_region"], houses["sqft_range"], q])["log_price"]
    houses["resid"] = houses["log_price"] - g.transform("median")
    houses["n"] = g.transform("size")
    s = houses[houses["n"] >= 4]
    print("Freehold houses: share mentioning each, and premium vs same type/area/size/quarter")
    for col in [*[f"fin_{k}" for k in FINISHES]]:
        line = f"  {col[4:]:12s} mention {houses[col].mean():5.0%} |"
        for grp in ["new (<=15 yrs)", "older"]:
            t = s[s["age_grp"] == grp]
            yes, no = t[t[col] == 1]["resid"], t[t[col] == 0]["resid"]
            if len(yes) >= 30:
                line += f" {grp}: {np.expm1(yes.median() - no.median()) * 100:+.1f}% (n={len(yes)})"
        print(line)
    for k in (0, 1, 2, 3):
        t = s[s["fin_count"].clip(upper=3) == k]
        print(f"  {k}{'+' if k == 3 else ' '} finishes mentioned: {np.expm1(t['resid'].median()) * 100:+.1f}% (n={len(t)})")

    train, test, _ = evaluate.splits(x)["recent"]
    cols = list(f.columns)
    base = train_gbm.fit(train, a.rounds, features=homeowner.FEATURES)
    plus = train_gbm.fit(train, a.rounds, features=homeowner.FEATURES + cols)
    y = test["close_price"].to_numpy()
    pb, pp = base.predict(test)["estimate"].to_numpy(), plus.predict(test)["estimate"].to_numpy()
    age = test["age_years"].to_numpy()
    fh = (test["type_group"] == "freehold").to_numpy()
    for name, m in {"all homes": np.ones(len(test), bool), "freehold": fh,
                    "freehold <=15 yrs": fh & (age <= 15), "freehold older": fh & (age > 15),
                    "freehold, finishes mentioned": fh & (test["fin_count"].to_numpy() >= 2)}.items():
        mb, mp = evaluate.metrics(y[m], pb[m]), evaluate.metrics(y[m], pp[m])
        print(f"{name:30s} n={m.sum():5d}  without {mb['mdape']:.2f}%  with finishes {mp['mdape']:.2f}%")


if __name__ == "__main__":
    main()
