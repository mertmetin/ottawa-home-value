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


@dataclass
class GBM:
    models: dict[str, lgb.Booster] = field(default_factory=dict)
    categories: dict[str, list[str]] = field(default_factory=dict)

    def frame(self, x: pd.DataFrame) -> pd.DataFrame:
        """Model input with categoricals pinned to the training categories."""
        out = x[F.FEATURES].copy()
        for c in F.NUMERIC:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
        for c in F.CATEGORICAL:
            v = out[c].astype("object")
            v = v.where(v.isin(self.categories[c]))  # unseen values become missing
            out[c] = pd.Categorical(v, categories=self.categories[c])
        return out

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        X = self.frame(x)
        out = pd.DataFrame(index=x.index)
        out["estimate"] = np.exp(self.models["mid"].predict(X))
        for name in QUANTILES:
            out[name] = np.exp(self.models[name].predict(X))
        # Quantile models are fit separately and can cross; keep lo <= estimate <= hi.
        out["lo"] = np.minimum(out["lo"], out["estimate"])
        out["hi"] = np.maximum(out["hi"], out["estimate"])
        return out

    def importance(self) -> pd.Series:
        m = self.models["mid"]
        return pd.Series(m.feature_importance("gain"), index=m.feature_name()).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for k, m in self.models.items():
            m.save_model(str(path / f"{k}.lgb"))
        (path / "categories.json").write_text(json.dumps(self.categories))

    @classmethod
    def load(cls, path: Path) -> "GBM":
        g = cls(categories=json.loads((path / "categories.json").read_text()))
        for k in ["mid", *QUANTILES]:
            g.models[k] = lgb.Booster(model_file=str(path / f"{k}.lgb"))
        return g


def split_valid(train: pd.DataFrame, frac: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The most recent `frac` of the training sales, for early stopping."""
    cut = train["sale_date"].quantile(1 - frac)
    return train[train["sale_date"] < cut], train[train["sale_date"] >= cut]


def fit(train: pd.DataFrame, rounds: int = 5000, params: dict | None = None) -> GBM:
    params = {**PARAMS, **(params or {})}
    g = GBM(categories={c: sorted(train[c].dropna().astype(str).unique()) for c in F.CATEGORICAL})
    tr, va = split_valid(train)
    Xtr, Xva = g.frame(tr), g.frame(va)
    ytr, yva = tr[F.TARGET].to_numpy(), va[F.TARGET].to_numpy()

    def one(objective: str, alpha: float | None, best: int | None = None) -> lgb.Booster:
        p = {**params, "objective": objective}
        if alpha is not None:
            p["alpha"] = alpha
        dtr = lgb.Dataset(Xtr, ytr, categorical_feature=F.CATEGORICAL, free_raw_data=False)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        if best is not None:  # quantile models reuse the point model's tree count
            return lgb.train(p, dtr, num_boost_round=best)
        return lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(200, verbose=False)])

    g.models["mid"] = one("regression", None)
    best = g.models["mid"].best_iteration or rounds
    for name, a in QUANTILES.items():
        g.models[name] = one("quantile", a, best)
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
