"""Production jobs for the served model.

    python -m src.pipeline retrain   # weekly: train a candidate, promote it only if it passes the gate
    python -m src.pipeline nightly   # nightly: new sales -> history snapshot, drift log, recalibrated range
    python -m src.pipeline status    # what is served, how it is doing
    python -m src.pipeline rollback  # serve the previous version again

--data db (the default) reads DATABASE_URL; a CSV path works for the sample.

Retrain and the promotion gate
------------------------------
The newest GATE_DAYS of sales are held out. A candidate is trained on the
sales before them, then the candidate and the served model are both scored on
the held-out sales that neither has seen (those after the served model's data
ends). When too few of those exist (the served model is recent), the served
model's recipe (its feature list) is retrained on the candidate's sales and
scored instead, so the comparison is like with like. The candidate is
promoted when

* its median error is no more than GATE_MDAPE_SLACK points worse,
* its share within 10% is no more than GATE_WITHIN10_SLACK points lower,
* its calibrated range covers at least GATE_MIN_COVERAGE of those sales, and
* its median error is under SANITY_MDAPE (a broken feed can't sneak through).

When it passes, the model that is saved and served is refit on every sale,
the held-out ones included, with the same settings. A failed gate leaves the
served model alone and records why in promotions.jsonl.

Nightly
-------
Sales that went firm since the served model's data ended are new to it. Each
night they are used to (1) log its accuracy and input drift, measured before
anything is refit, then (2) recalibrate its range on the newest
RECALIB_N sales its quantile models never saw. Range coverage decayed from 79%
to 72% over three months without this.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, drift, homeowner, registry as R, text, train_gbm
from . import features as F

log = logging.getLogger("pipeline")

GATE_DAYS = 45
GATE_MIN = 150
GATE_MDAPE_SLACK = 0.25      # percentage points
GATE_WITHIN10_SLACK = 2.0    # percentage points
GATE_MIN_COVERAGE = 70.0     # percent (the target is 80)
SANITY_MDAPE = 10.0
RECALIB_N = 2000
RECALIB_MIN = 500
RECENT_DAYS = 30
SNAPSHOT_COLS = ["sale_date", "lat", "lng", "city_region", "type_group", "property_type", "ppsf", "close_price",
                 "parking_total", "garage_spaces", "lot_width", "lot_depth", "tax_annual", "sqft_mid", "basement_kind", "basement_finished",
                 "has_basement", "garage_type"]


def models_dir() -> Path:
    return Path(os.environ.get("OHV_MODELS", "models"))


def logs_dir() -> Path:
    return Path(os.environ.get("OHV_LOGS", "logs"))


def snapshot_path() -> Path:
    return models_dir() / "history.pkl"


def code_version() -> str:
    if os.environ.get("OHV_GIT_SHA"):
        return os.environ["OHV_GIT_SHA"]
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent.parent, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def prepare(source: str) -> tuple[pd.DataFrame, list[dict]]:
    """All sales, cleaned, with every feature the served model uses."""
    sales, removed = F.clean(data.load(source))
    x = F.build(sales)
    upd = text.update_features(x["remarks"], x["sale_date"].dt.year)
    x[upd.columns] = upd.to_numpy()
    x = F.condo_floor_features(x)
    return x.sort_values("sale_date").reset_index(drop=True), removed


def write_snapshot(x: pd.DataFrame) -> None:
    """Past sales for the API's market features (licensed data: stays in models/ on the server)."""
    p = snapshot_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    x[SNAPSHOT_COLS].to_pickle(tmp)
    os.replace(tmp, p)


def fit(x: pd.DataFrame, rounds: int = 5000, features: list[str] | None = None) -> tuple[train_gbm.GBM, dict]:
    g = train_gbm.fit(x, rounds=rounds, features=features or homeowner.FEATURES)
    tr, va = train_gbm.split_valid(x)
    card = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "code": code_version(),
        "feature_set": homeowner.FEATURE_SET,
        "n_features": len(g.features),
        "n_train": int(len(x)),
        "trees": int(g.models["mid"].num_trees()),
        "data_start": str(x["sale_date"].min().date()),
        "data_end": str(x["sale_date"].max().date()),
        # The quantile models never saw sales from this day on, so they can calibrate the range.
        "calib_from": str(va["sale_date"].min().date()),
        "drift_ref": drift.reference(x),
    }
    return g, card


def score(g: train_gbm.GBM, x: pd.DataFrame) -> dict:
    return drift.accuracy(x["close_price"].to_numpy(), g.predict(x))


def gate(cand: dict, cur: dict | None) -> tuple[bool, list[str]]:
    why = []
    if cand.get("n", 0) == 0:
        return False, ["no held-out sales to score"]
    if cand["mdape"] > SANITY_MDAPE:
        why.append(f"median error {cand['mdape']}% is over the {SANITY_MDAPE}% sanity limit")
    if cand["coverage"] < GATE_MIN_COVERAGE:
        why.append(f"range covers {cand['coverage']}%, under {GATE_MIN_COVERAGE}%")
    if cur and cur.get("n", 0):
        if cand["mdape"] > cur["mdape"] + GATE_MDAPE_SLACK:
            why.append(f"median error {cand['mdape']}% vs {cur['mdape']}% served")
        if cand["within_10"] < cur["within_10"] - GATE_WITHIN10_SLACK:
            why.append(f"within 10% {cand['within_10']}% vs {cur['within_10']}% served")
    return not why, why


def retrain(source: str, force: bool = False, rounds: int = 5000) -> dict:
    reg = R.Registry(models_dir() / "registry")
    x, removed = prepare(source)
    write_snapshot(x)
    hold_start = x["sale_date"].max() - pd.Timedelta(days=GATE_DAYS)
    train, hold = x[x["sale_date"] < hold_start], x[x["sale_date"] >= hold_start]
    log.info("candidate: %d sales before %s, %d held out", len(train), hold_start.date(), len(hold))
    cand_g, _ = fit(train, rounds)

    cur = reg.current()
    gate_set, note, ref = hold, "all held-out sales", None
    if cur:
        unseen = hold[hold["sale_date"] > pd.Timestamp(cur.card["data_end"])]
        if len(unseen) >= GATE_MIN:
            gate_set, note = unseen, f"held-out sales after the served model's data ({cur.card['data_end']})"
            ref = cur.load()
        else:
            # The served model saw these sales, so scoring it on them flatters it.
            # Compare like with like: the served model's recipe (its features)
            # retrained on exactly the candidate's training sales.
            note = (f"all held-out sales vs the served model's features retrained on the same sales "
                    f"(only {len(unseen)} are new to the served model)")
            ref, _ = fit(train, rounds, features=cur.load().features)
    m_cand = score(cand_g, gate_set)
    m_cur = score(ref, gate_set) if ref is not None else None
    ok, why = gate(m_cand, m_cur)
    result = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "gate_set": note, "gate_n": int(len(gate_set)),
              "candidate": m_cand, "served": m_cur, "served_version": cur.name if cur else None,
              "passed": ok, "failures": why, "forced": bool(force and not ok)}
    log.info("gate: candidate %s | served %s -> %s %s", m_cand, m_cur, "PASS" if ok else "FAIL", why)

    if ok or force:
        g, card = fit(x, rounds)
        card["gate"] = result
        card["cleaning"] = removed
        v = reg.save(g, card)
        reg.promote(v.name, why="forced past a failed gate" if not ok else "passed the gate")
        result["version"] = v.name
        log.info("promoted %s (%d sales to %s)", v.name, card["n_train"], card["data_end"])
    else:
        R.append_jsonl(reg.root / "promotions.jsonl", {"event": "rejected", **result})
    return result


def nightly(source: str) -> dict:
    reg = R.Registry(models_dir() / "registry")
    x, _ = prepare(source)
    write_snapshot(x)
    cur = reg.current()
    row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "sales": int(len(x)),
           "last_sale": str(x["sale_date"].max().date())}
    if cur is None:
        row["note"] = "no served model yet: run retrain"
        R.append_jsonl(logs_dir() / "drift.jsonl", row)
        return row
    card, g = cur.card, cur.load()
    row["version"] = cur.name

    # 1. How the served model does on sales it never saw, before anything is refit.
    unseen = x[x["sale_date"] > pd.Timestamp(card["data_end"])]
    recent = unseen[unseen["sale_date"] > x["sale_date"].max() - pd.Timedelta(days=RECENT_DAYS)]
    row["unseen"] = score(g, unseen) if len(unseen) else {"n": 0}
    row["last_30_days"] = score(g, recent) if len(recent) else {"n": 0}
    trained_on = card.get("gate", {}).get("candidate", {})
    acc = row["unseen"]
    row["retrain_suggested"] = bool(acc.get("n", 0) >= GATE_MIN and (
        acc["coverage"] < GATE_MIN_COVERAGE
        or (trained_on.get("mdape") and acc["mdape"] > trained_on["mdape"] * 1.25)))

    # 2. Input drift: recent sales, and homes people asked about.
    ref = card.get("drift_ref", {})
    recent_sales = x[x["sale_date"] > x["sale_date"].max() - pd.Timedelta(days=90)]
    row["psi_sales_90d"] = drift.psi(ref, recent_sales)
    req = R.read_jsonl(logs_dir() / "requests.jsonl")
    if req:
        rq = pd.DataFrame([r["x"] for r in req if r.get("at", "") >= (pd.Timestamp.now() - pd.Timedelta(days=30)).isoformat()])
        row["requests_30d"] = int(len(rq))
        row["psi_requests_30d"] = drift.psi(ref, rq, min_n=30) if len(rq) else {}
    row["drift_flags"] = drift.flags({**row["psi_sales_90d"], **{f"requests:{k}": v for k, v in row.get("psi_requests_30d", {}).items()}})

    # 3. Recalibrate the range on the newest sales the quantile models never saw.
    fresh = x[x["sale_date"] >= pd.Timestamp(card["calib_from"])].tail(RECALIB_N)
    if len(fresh) >= RECALIB_MIN:
        calib = train_gbm.calibrate(g, fresh)
        cur.set_calib(calib, {"n": int(len(fresh)), "from": str(fresh["sale_date"].min().date()),
                              "to": str(fresh["sale_date"].max().date())})
        row["recalibrated"] = {"n": int(len(fresh)), "all": round(calib["all"], 4)}
    else:
        row["recalibrated"] = None
    R.append_jsonl(logs_dir() / "drift.jsonl", row)
    return row


def status() -> dict:
    reg = R.Registry(models_dir() / "registry")
    cur = reg.current()
    out = {"served": None, "versions": [v.name for v in reg.versions()]}
    if cur:
        c = cur.card
        out["served"] = {k: c.get(k) for k in ("version", "trained_at", "code", "n_train", "data_start", "data_end", "trees")}
        out["served"]["gate"] = {k: c.get("gate", {}).get(k) for k in ("candidate", "served", "passed")}
    d = R.read_jsonl(logs_dir() / "drift.jsonl")
    out["last_nightly"] = d[-1] if d else None
    out["last_decisions"] = R.read_jsonl(reg.root / "promotions.jsonl")[-3:]
    return out


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", choices=["retrain", "nightly", "status", "rollback"])
    ap.add_argument("--data", default="db", help="'db' for DATABASE_URL, or a CSV path")
    ap.add_argument("--force", action="store_true", help="retrain: promote even if the gate fails")
    a = ap.parse_args(argv)
    if a.job == "retrain":
        out = retrain(a.data, force=a.force)
    elif a.job == "nightly":
        out = nightly(a.data)
    elif a.job == "rollback":
        out = {"serving": R.Registry(models_dir() / "registry").rollback()}
    else:
        out = status()
    print(json.dumps(out, indent=2, default=str))
    if a.job == "retrain" and not out.get("passed") and not out.get("version"):
        sys.exit(3)  # gate failed: visible to cron / monitoring


if __name__ == "__main__":
    main()
