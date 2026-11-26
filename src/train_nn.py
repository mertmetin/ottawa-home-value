"""PyTorch MLP with learned embeddings for the categorical features.

Numeric features are log-scaled where skewed, median-imputed with a missing
flag, and standardised. Each categorical gets an embedding (rare values share
the "unknown" slot). The target is standardised log(price).

    python -m src.train_nn --data data/sample.csv --out models/nn
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import features as F
from .train_gbm import split_valid

LOG_COLS = {"sqft_mid", "lot_area", "lot_width", "lot_depth", "tax_annual", "condo_fee", "comp_price_med",
            "comp_ppsf_med", "area_ppsf_90d", "city_ppsf_90d", "comp_dist_km", "dist_downtown_km",
            "dist_station_km", "age_years", "area_n_90d"}
MIN_CAT_COUNT = 5


@dataclass
class Prep:
    median: dict[str, float] = field(default_factory=dict)
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)   # numeric columns that get a missing indicator
    vocab: dict[str, list[str]] = field(default_factory=dict)
    y_mean: float = 0.0
    y_std: float = 1.0

    @classmethod
    def fit(cls, x: pd.DataFrame) -> "Prep":
        p = cls()
        num = cls._num(x)
        for c in F.NUMERIC:
            v = num[c]
            p.median[c] = float(v.median()) if v.notna().any() else 0.0
            f = v.fillna(p.median[c])
            p.mean[c], p.std[c] = float(f.mean()), float(f.std() or 1.0)
            if v.isna().any():
                p.flags.append(c)
        for c in F.CATEGORICAL:
            vc = x[c].dropna().astype(str).value_counts()
            p.vocab[c] = sorted(vc[vc >= MIN_CAT_COUNT].index)
        p.y_mean, p.y_std = float(x[F.TARGET].mean()), float(x[F.TARGET].std())
        return p

    @staticmethod
    def _num(x: pd.DataFrame) -> pd.DataFrame:
        out = x[F.NUMERIC].apply(pd.to_numeric, errors="coerce").astype(float)
        for c in LOG_COLS:
            out[c] = np.log1p(out[c].clip(lower=0))
        return out

    def transform(self, x: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        num = self._num(x)
        cols = []
        for c in F.NUMERIC:
            v = num[c]
            cols.append(((v.fillna(self.median[c]) - self.mean[c]) / self.std[c]).to_numpy())
        for c in self.flags:
            cols.append(num[c].isna().to_numpy(float))
        xn = torch.tensor(np.column_stack(cols), dtype=torch.float32)
        idx = []
        for c in F.CATEGORICAL:
            lookup = {v: i + 1 for i, v in enumerate(self.vocab[c])}  # 0 = missing / rare / unseen
            idx.append(x[c].astype("object").map(lambda v: lookup.get(str(v), 0) if pd.notna(v) else 0).to_numpy())
        xc = torch.tensor(np.column_stack(idx), dtype=torch.long)
        return xn, xc

    def target(self, x: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(((x[F.TARGET] - self.y_mean) / self.y_std).to_numpy(), dtype=torch.float32)

    @property
    def n_numeric(self) -> int:
        return len(F.NUMERIC) + len(self.flags)


class Net(nn.Module):
    def __init__(self, n_numeric: int, cardinalities: list[int], hidden=(512, 256, 128), dropout=0.15):
        super().__init__()
        self.emb = nn.ModuleList(nn.Embedding(n + 1, min(16, (n + 2) // 2)) for n in cardinalities)
        width = n_numeric + sum(e.embedding_dim for e in self.emb)
        layers: list[nn.Module] = []
        for h in hidden:
            layers += [nn.Linear(width, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, xn, xc):
        e = [emb(xc[:, i]) for i, emb in enumerate(self.emb)]
        return self.mlp(torch.cat([xn, *e], dim=1)).squeeze(1)


@dataclass
class NN:
    prep: Prep
    net: Net

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        self.net.eval()
        xn, xc = self.prep.transform(x)
        with torch.no_grad():
            z = self.net(xn, xc).numpy()
        return pd.DataFrame({"estimate": np.exp(z * self.prep.y_std + self.prep.y_mean)}, index=x.index)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "net.pt")
        (path / "prep.json").write_text(json.dumps(self.prep.__dict__))

    @classmethod
    def load(cls, path: Path) -> "NN":
        prep = Prep(**json.loads((path / "prep.json").read_text()))
        net = Net(prep.n_numeric, [len(prep.vocab[c]) for c in F.CATEGORICAL])
        net.load_state_dict(torch.load(path / "net.pt", weights_only=True))
        return cls(prep, net)


def fit(train: pd.DataFrame, epochs: int = 300, batch: int = 256, lr: float = 2e-3,
        patience: int = 25, seed: int = 42, verbose: bool = False) -> NN:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    tr, va = split_valid(train)
    prep = Prep.fit(tr)
    xn, xc, y = *prep.transform(tr), prep.target(tr)
    vn, vc, vy = *prep.transform(va), prep.target(va)
    net = Net(prep.n_numeric, [len(prep.vocab[c]) for c in F.CATEGORICAL])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    loss_fn = nn.HuberLoss(delta=0.5)
    best, best_state, wait = np.inf, None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(y), generator=g)
        for i in range(0, len(y), batch):
            b = perm[i:i + batch]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss_fn(net(xn[b], xc[b]), y[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float((net(vn, vc) - vy).abs().mean()) * prep.y_std  # MAE in log price
        sched.step(val)
        if val < best - 1e-5:
            best, best_state, wait = val, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
        if verbose and ep % 10 == 0:
            print(f"epoch {ep:3d}  valid MAE(log) {val:.4f}")
    net.load_state_dict(best_state)
    return NN(prep, net)


def main():
    from . import data

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/sample.csv", help="CSV path, or 'db' for DATABASE_URL")
    ap.add_argument("--out", default="models/nn")
    a = ap.parse_args()
    sales, _ = F.clean(data.load(a.data))
    x = F.build(sales)
    m = fit(x, verbose=True)
    m.save(Path(a.out))
    print(f"trained on {len(x)} sales -> {a.out}")


if __name__ == "__main__":
    main()
