"""Drift: is the world the model sees still the one it learned?

Two kinds, both logged every night to logs/drift.jsonl:

* Accuracy on sales that went firm after the model's training data ended.
  These are the real test: the model never saw them. Error creeping up or the
  range covering fewer sales is the signal to retrain.
* Input drift (population stability index, PSI) of key features: recent sales,
  and the homes people asked about through the API, against the training
  data. PSI under 0.1 is stable, 0.1-0.25 is worth a look, over 0.25 has moved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC = ["sqft_mid", "age_years", "beds_total", "baths", "tax_annual", "lat", "lng", "comp_ppsf_med",
           "area_ppsf_90d"]
CATEGORICAL = ["property_type", "type_group"]
BINS = 10
PSI_WARN, PSI_ALERT = 0.1, 0.25


def reference(x: pd.DataFrame) -> dict:
    """Bin edges and shares of the training data, stored in the model card."""
    ref = {}
    for c in NUMERIC:
        v = pd.to_numeric(x[c], errors="coerce").dropna()
        if len(v) < BINS * 10:
            continue
        edges = np.unique(np.quantile(v, np.linspace(0, 1, BINS + 1))[1:-1])
        ref[c] = {"edges": edges.tolist(), "share": _shares_num(v, edges).tolist(),
                  "missing": float(pd.to_numeric(x[c], errors="coerce").isna().mean())}
    for c in CATEGORICAL:
        s = x[c].astype("object").fillna("?").value_counts(normalize=True)
        ref[c] = {"share": s.to_dict()}
    return ref


def _shares_num(v: pd.Series, edges) -> np.ndarray:
    idx = np.searchsorted(np.asarray(edges), v.to_numpy(), side="right")
    return np.bincount(idx, minlength=len(edges) + 1) / max(len(v), 1)


def _psi(expected, actual, eps: float = 1e-4) -> float:
    e = np.clip(np.asarray(expected, float), eps, None)
    a = np.clip(np.asarray(actual, float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi(ref: dict, x: pd.DataFrame, min_n: int = 50) -> dict:
    """PSI per feature of x against the reference (features with too few values are skipped)."""
    out = {}
    for c, r in ref.items():
        if c not in x:
            continue
        if "edges" in r:
            v = pd.to_numeric(x[c], errors="coerce").dropna()
            if len(v) < min_n:
                continue
            out[c] = round(_psi(r["share"], _shares_num(v, r["edges"])), 4)
        else:
            s = x[c].astype("object").fillna("?")
            if len(s) < min_n:
                continue
            got = s.value_counts(normalize=True)
            cats = sorted(set(r["share"]) | set(got.index))
            out[c] = round(_psi([r["share"].get(k, 0) for k in cats], [got.get(k, 0) for k in cats]), 4)
    return out


def flags(p: dict) -> dict:
    return {k: ("alert" if v > PSI_ALERT else "watch") for k, v in p.items() if v > PSI_WARN}


def accuracy(y: np.ndarray, pred: pd.DataFrame) -> dict:
    """Point and range accuracy of predictions against sale prices."""
    if len(y) == 0:
        return {"n": 0}
    est = pred["estimate"].to_numpy()
    ape = np.abs(est - y) / y
    return {
        "n": int(len(y)),
        "mdape": round(float(np.median(ape) * 100), 2),
        "within_10": round(float(np.mean(ape <= 0.10) * 100), 1),
        "bias": round(float(np.median((est - y) / y) * 100), 2),
        "coverage": round(float(np.mean((pred["lo"].to_numpy() <= y) & (y <= pred["hi"].to_numpy())) * 100), 1),
        "range_width": round(float(np.median((pred["hi"] - pred["lo"]).to_numpy() / est) * 100), 1),
    }
