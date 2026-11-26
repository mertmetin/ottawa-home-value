"""Condo apartments: does the floor (from the unit number) help?

    python -m scripts.floor_check --data db
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import evaluate, homeowner, pipeline, train_gbm  # noqa: E402
from src import features as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="db")
    ap.add_argument("--rounds", type=int, default=5000)
    a = ap.parse_args()
    x, _ = pipeline.prepare(a.data)
    x = F.condo_floor_features(x)
    apt = x[x["property_type"] == "Condo Apartment"]
    print(f"condo apartments {len(apt)}, floor known {apt['floor'].notna().mean():.0%}, penthouse {int(apt['penthouse'].sum())}")
    # Raw premium per floor band, same building (street address) and quarter
    b = apt.assign(bld=F.street_key(apt["address"].str.replace(r"^[0-9A-Za-z]+-", "", regex=True)) + apt["address"].str.extract(r"-(\d+)", expand=False).fillna(""),
                   q=apt["sale_date"].dt.to_period("Q").astype(str))
    g = b.groupby(["bld", "q"])["ppsf"]
    b = b.assign(rel=np.log(b["ppsf"]) - np.log(g.transform("median")), n=g.transform("count"))
    s = b[(b["n"] >= 3) & b["floor"].notna() & b["rel"].notna()]
    bands = s.groupby(pd_cut := s["floor"].clip(upper=25) // 5 * 5)["rel"].agg(["median", "count"])
    print("$/sqft vs same building, same quarter, by floor band (0 = floors 1-4):")
    for f, r in bands.iterrows():
        print(f"  floors {int(f)}-{int(f)+4}: {np.expm1(r['median'])*100:+.1f}%  ({int(r['count'])} sales)")
    train, test, _ = evaluate.splits(x)["recent"]
    t = test[test["property_type"] == "Condo Apartment"]
    y = t["close_price"].to_numpy()
    for name, feats in {"without floor": homeowner.FEATURES, "with floor": homeowner.FEATURES + ["floor", "penthouse"]}.items():
        g = train_gbm.fit(train, a.rounds, features=feats)
        p = g.predict(t)["estimate"].to_numpy()
        m = evaluate.metrics(y, p)
        hi = t["floor"].to_numpy() >= 10
        mh = evaluate.metrics(y[hi], p[hi])
        print(f"{name:14s} condo apt MdAPE {m['mdape']:.2f}% within10 {m['within_10']:.1f}% | floor 10+ ({hi.sum()}) MdAPE {mh['mdape']:.2f}% bias {np.median((p[hi]-y[hi])/y[hi])*100:+.1f}%")


if __name__ == "__main__":
    main()
