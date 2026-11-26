"""LightGBM on log(price), plus 10th/90th percentile models for a range.

    python -m src.train_gbm --data data/sample.csv --out models/gbm
    python -m src.train_gbm --data db --out models/gbm      # on the server
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

try:
    # Load PyTorch's OpenMP runtime before LightGBM's. The other order makes
    # the first torch op segfault on macOS (two libomp copies in one process).
    import torch  # noqa: F401
except ImportError:
    pass
import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features as F

PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "cat_smooth": 10,
    "verbose": -1,
    "seed": 42,
    "num_threads": 4,
}
QUANTILES = {"lo": 0.1, "hi": 0.9}
COVERAGE = 0.8          # the range should contain the sale price this often
MIN_CALIB_GROUP = 100   # calibrate a group on its own when it has this many held-out sales
# Predicted-price bands for calibration (the true price is unknown at prediction time).
CALIB_BANDS = [0, 400_000, 600_000, 800_000, 1_200_000, np.inf]


@dataclass
class GBM:
    models: dict[str, lgb.Booster] = field(default_factory=dict)
    categories: dict[str, list[str]] = field(default_factory=dict)
    features: list[str] = field(default_factory=lambda: list(F.FEATURES))
    # Conformal widening of the 10-90% range, in log price, per type_group
    # ("all" is the fallback). Raw quantile models are overconfident on new
    # sales; this stretches the range until it covers COVERAGE of held-out sales.
    calib: dict[str, float] = field(default_factory=dict)

    @property
    def categorical(self) -> list[str]:
        return [c for c in F.CATEGORICAL if c in self.features]

    def frame(self, x: pd.DataFrame) -> pd.DataFrame:
        """Model input with categoricals pinned to the training categories."""
        out = x[self.features].copy()
        for c in self.features:
            if c not in F.CATEGORICAL:
                out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
        for c in self.categorical:
            v = out[c].astype("object")
            v = v.where(v.isin(self.categories[c]))  # unseen values become missing
            out[c] = pd.Categorical(v, categories=self.categories[c])
        return out

    def raw(self, x: pd.DataFrame) -> pd.DataFrame:
        """Log-price predictions before calibration, with lo <= mid <= hi."""
        X = self.frame(x)
        out = pd.DataFrame(index=x.index)
        out["mid"] = self.models["mid"].predict(X)
        for name in QUANTILES:
            out[name] = self.models[name].predict(X)
        # Quantile models are fit separately and can cross.
        out["lo"] = np.minimum(out["lo"], out["mid"])
        out["hi"] = np.maximum(out["hi"], out["mid"])
        return out

    def widen(self, x: pd.DataFrame, mid: np.ndarray) -> np.ndarray:
        """Widening per sale: its type group and predicted price band, falling
        back to the type group, then to everything."""
        tg = x["type_group"].astype("object").to_numpy() if "type_group" in x else np.full(len(x), "all", object)
        out = np.empty(len(x))
        for i, (t, k) in enumerate(zip(tg, calib_keys(tg, mid))):
            out[i] = self.calib.get(k, self.calib.get(t, self.calib.get("all", 0.0)))
        return out

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        r = self.raw(x)
        q = self.widen(x, r["mid"].to_numpy())
        return pd.DataFrame({"estimate": np.exp(r["mid"]), "lo": np.exp(r["lo"] - q), "hi": np.exp(r["hi"] + q)},
                            index=x.index)

    def importance(self) -> pd.Series:
        m = self.models["mid"]
        return pd.Series(m.feature_importance("gain"), index=m.feature_name()).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for k, m in self.models.items():
            m.save_model(str(path / f"{k}.lgb"))
        meta = {"categories": self.categories, "features": self.features, "calib": self.calib}
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> "GBM":
        g = cls(**json.loads((path / "meta.json").read_text()))
        for k in ["mid", *QUANTILES]:
            g.models[k] = lgb.Booster(model_file=str(path / f"{k}.lgb"))
        return g


def split_valid(train: pd.DataFrame, frac: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The most recent `frac` of the training sales, for early stopping."""
    cut = train["sale_date"].quantile(1 - frac)
    return train[train["sale_date"] < cut], train[train["sale_date"] >= cut]


def conformal(scores: np.ndarray, coverage: float = COVERAGE) -> float:
    """Split-conformal quantile of CQR scores: widen by this to reach coverage."""
    n = len(scores)
    if n == 0:
        return 0.0
    k = min(int(np.ceil((n + 1) * coverage)), n)  # the k-th smallest score
    return float(max(np.sort(scores)[k - 1], 0.0))


def calib_keys(type_group: np.ndarray, log_mid: np.ndarray) -> list[str]:
    band = np.digitize(np.exp(log_mid), CALIB_BANDS[1:-1])
    return [f"{t}|{b}" for t, b in zip(type_group, band)]


def calibrate(g: GBM, held_out: pd.DataFrame) -> dict[str, float]:
    """Conformalized quantile regression (Romano et al., 2019) on sales the
    quantile models never saw. Score = how far the sale fell outside the raw
    range (negative when inside)."""
    r = g.raw(held_out)
    y = held_out[F.TARGET].to_numpy()
    scores = np.maximum(r["lo"].to_numpy() - y, y - r["hi"].to_numpy())
    calib = {"all": conformal(scores)}
    tg = held_out["type_group"].astype("object").to_numpy()
    keys = np.array(calib_keys(tg, r["mid"].to_numpy()))
    for col in (tg, keys):
        for grp in np.unique(col):
            idx = np.flatnonzero(col == grp)
            if len(idx) >= MIN_CALIB_GROUP:
                calib[str(grp)] = conformal(scores[idx])
    return calib


def fit(train: pd.DataFrame, rounds: int = 5000, params: dict | None = None,
        features: list[str] | None = None) -> GBM:
    params = {**PARAMS, **(params or {})}
    feats = list(features or F.FEATURES)
    g = GBM(categories={c: sorted(train[c].dropna().astype(str).unique()) for c in F.CATEGORICAL if c in feats},
            features=feats)
    tr, va = split_valid(train)
    Xtr, Xva = g.frame(tr), g.frame(va)
    ytr, yva = tr[F.TARGET].to_numpy(), va[F.TARGET].to_numpy()

    def one(objective: str, alpha: float | None, best: int | None = None) -> lgb.Booster:
        p = {**params, "objective": objective}
        if alpha is not None:
            p["alpha"] = alpha
        dtr = lgb.Dataset(Xtr, ytr, categorical_feature=g.categorical, free_raw_data=False)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        if best is not None:  # quantile models reuse the point model's tree count
            return lgb.train(p, dtr, num_boost_round=best)
        return lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(200, verbose=False)])

    g.models["mid"] = one("regression", None)
    best = g.models["mid"].best_iteration or rounds
    for name, a in QUANTILES.items():
        g.models[name] = one("quantile", a, best)
    # The quantile models trained on tr only, so va is fresh for them. It is
    # also the newest slice of training, closest to the sales we'll predict.
    g.calib = calibrate(g, va)
    return g


def main():
    from . import data

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/sample.csv", help="CSV path, or 'db' for DATABASE_URL")
    ap.add_argument("--out", default="models/gbm")
    a = ap.parse_args()
    sales, _ = F.clean(data.load(a.data))
    x = F.build(sales)
    g = fit(x)
    g.save(Path(a.out))
    print(f"trained on {len(x)} sales, {g.models['mid'].num_trees()} trees -> {a.out}")


if __name__ == "__main__":
    main()
