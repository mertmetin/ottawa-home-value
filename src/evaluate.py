"""Train every model on the same time-based splits and compare them.

    python -m src.evaluate --data data/sample.csv --out results/sample
    python -m src.evaluate --data db --out results/real          # on the server

Splits (both are "train on the past, test on the future"):

* recent: test on the last 3 months of sales, train on everything before.
* shift:  train on the first --shift-months months, test on everything after,
          to see how a model holds up as the market moves away from it.

Models: the comps baseline, LightGBM (with a 10-90% range), the MLP, and
TabPFN when it is installed and --models asks for it.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from . import features as F
from . import train_gbm, train_nn

log = logging.getLogger("evaluate")

PRICE_BANDS = [0, 400_000, 600_000, 800_000, 1_200_000, np.inf]
BAND_LABELS = ["< $400K", "$400-600K", "$600-800K", "$800K-1.2M", "> $1.2M"]
MODEL_NAMES = {"baseline": "Comps baseline", "gbm": "LightGBM", "nn": "Neural net (MLP)", "tabpfn": "TabPFN",
               "gbm_notax": "LightGBM, no taxes", "gbm_updates": "+ updates (roof, furnace...)",
               "gbm_text": "+ description", "gbm_photos": "+ photos", "gbm_all": "+ description + photos"}
NO_TAX = [f for f in F.FEATURES if f not in ("tax_annual", "tax_years_old")]


def variants() -> dict[str, list[str]]:
    """LightGBM feature sets: what each kind of extra information adds."""
    from . import text

    upd = F.FEATURES + text.UPDATE_COLS
    desc = upd + text.SUMMARY_COLS
    pho = F.photo_columns()
    return {"gbm_notax": NO_TAX, "gbm_updates": upd, "gbm_text": desc,
            "gbm_photos": F.FEATURES + pho, "gbm_all": desc + pho}
TABPFN_MAX_TRAIN = 10_000


# --------------------------------------------------------------------------
# Splits and metrics
# --------------------------------------------------------------------------

def splits(x: pd.DataFrame, test_months: int = 3, shift_months: int = 12) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, str]]:
    last, first = x["sale_date"].max(), x["sale_date"].min()
    cut = last - pd.DateOffset(months=test_months)
    shift_end = first + pd.DateOffset(months=shift_months)
    fmt = lambda d: d.strftime("%Y-%m-%d")
    return {
        "recent": (x[x["sale_date"] <= cut], x[x["sale_date"] > cut],
                   f"train {fmt(first)} to {fmt(cut)}, test after {fmt(cut)} (last {test_months} months)"),
        "shift": (x[x["sale_date"] < shift_end], x[x["sale_date"] >= shift_end],
                  f"train first {shift_months} months ({fmt(first)} to {fmt(shift_end)}), test everything after"),
    }


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) == 0:
        return {"n": 0}
    ape = np.abs(p - y) / y
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(p - y))),
        "mdape": float(np.median(ape) * 100),
        "within_5": float(np.mean(ape <= 0.05) * 100),
        "within_10": float(np.mean(ape <= 0.10) * 100),
        "bias": float(np.median((p - y) / y) * 100),  # + means the model tends to overestimate
    }


def breakdown(test: pd.DataFrame, preds: dict[str, np.ndarray], by: str, min_n: int = 20, top: int | None = None) -> pd.DataFrame:
    rows = []
    keys = test[by].astype("object").fillna("(unknown)")
    for key, idx in test.groupby(keys, observed=True).indices.items():
        if len(idx) < min_n:
            continue
        r = {by: key, "n": len(idx)}
        y = test["close_price"].to_numpy()[idx]
        for m, p in preds.items():
            r[m] = metrics(y, p[idx]).get("mdape", np.nan)
        rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if by == "price_band":
        out = out.set_index(by).reindex([b for b in BAND_LABELS if b in set(out[by])]).reset_index()
    else:
        out = out.sort_values("n", ascending=False)
    return out.head(top) if top else out


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def run_tabpfn(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    from tabpfn import TabPFNRegressor  # optional: pip install .[tabpfn]

    tr = train.sort_values("sale_date").tail(TABPFN_MAX_TRAIN)  # TabPFN's sample limit: keep the newest
    g = train_gbm.GBM(categories={c: sorted(tr[c].dropna().astype(str).unique()) for c in F.CATEGORICAL})  # encoder only

    def enc(x):
        X = g.frame(x)
        for c in F.CATEGORICAL:
            X[c] = X[c].cat.codes.replace(-1, np.nan).astype(float)
        return X.to_numpy(float)

    m = TabPFNRegressor(random_state=42)
    m.fit(enc(tr), tr[F.TARGET].to_numpy())
    return np.exp(m.predict(enc(test)))


def run_split(train: pd.DataFrame, test: pd.DataFrame, models: list[str]) -> tuple[dict, dict, dict]:
    preds, extra, timing = {}, {}, {}
    if "baseline" in models:
        preds["baseline"] = F.baseline_estimate(test)
    if "gbm" in models:
        t = time.time()
        g = train_gbm.fit(train)
        timing["gbm"] = time.time() - t
        out = g.predict(test)
        preds["gbm"] = out["estimate"].to_numpy()
        y = test["close_price"].to_numpy()
        raw = np.exp(g.raw(test))
        inside = (y >= out["lo"].to_numpy()) & (y <= out["hi"].to_numpy())
        width = ((out["hi"] - out["lo"]) / out["estimate"]).to_numpy()
        extra["gbm_range"] = {
            "coverage_raw": float(np.mean((y >= raw["lo"]) & (y <= raw["hi"])) * 100),
            "width_raw": float(np.median((raw["hi"] - raw["lo"]) / raw["mid"]) * 100),
            "coverage_10_90": float(np.mean(inside) * 100),
            "median_width_pct": float(np.median(width) * 100),
            "calib": {k: round(v, 4) for k, v in g.calib.items()},
        }
        by = {}
        for col in ("type_group", "price_band"):
            for key, idx in test.groupby(test[col].astype("object")).indices.items():
                if len(idx) >= 20:
                    by[f"{col}={key}"] = {"n": int(len(idx)), "coverage": float(inside[idx].mean() * 100),
                                          "width": float(np.median(width[idx]) * 100)}
        extra["gbm_range"]["by"] = by
        extra["gbm_importance"] = g.importance().head(15).round(1).to_dict()
        extra["gbm_trees"] = g.models["mid"].num_trees()
    for name, feats in variants().items():
        if name not in models:
            continue
        missing = [f for f in feats if f not in train]
        if missing:
            log.warning("%s skipped: no %s (photos need --photos)", name, missing[0])
            continue
        g2 = train_gbm.fit(train, features=feats)
        preds[name] = g2.predict(test)["estimate"].to_numpy()
        extra[f"{name}_importance"] = g2.importance().head(20).round(1).to_dict()
    if "nn" in models:
        t = time.time()
        n = train_nn.fit(train)
        timing["nn"] = time.time() - t
        preds["nn"] = n.predict(test)["estimate"].to_numpy()
    if "tabpfn" in models:
        try:
            t = time.time()
            preds["tabpfn"] = run_tabpfn(train, test)
            timing["tabpfn"] = time.time() - t
        except ImportError:
            log.warning("tabpfn not installed (pip install .[tabpfn]); skipped")
    return preds, extra, timing


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _fmt_table(df: pd.DataFrame, pct_cols: list[str]) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(MODEL_NAMES.get(c, c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in pct_cols:
                cells.append("" if pd.isna(v) else f"{v:.1f}%")
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(out: Path, source: str, cleaning: list[dict], results: dict) -> None:
    md = [f"# Results ({source})", ""]
    md += ["## Cleaning", "", "| Step | Removed | Why |", "|---|---|---|"]
    md += [f"| {c['step']} | {c['removed']:,} | {c['why']} |" for c in cleaning]
    for name, r in results.items():
        md += ["", f"## Split: {name}", "", r["desc"], f"Train {r['n_train']:,} sales, test {r['n_test']:,}.", ""]
        md += ["| Model | MAE | Median abs % error | Within 5% | Within 10% | Median bias |", "|---|---|---|---|---|---|"]
        for m, s in r["overall"].items():
            md.append(f"| {MODEL_NAMES[m]} | ${s['mae']:,.0f} | {s['mdape']:.1f}% | {s['within_5']:.1f}% | "
                      f"{s['within_10']:.1f}% | {s['bias']:+.1f}% |")
        if "gbm_range" in r["extra"]:
            gr = r["extra"]["gbm_range"]
            md += ["", "### LightGBM range (target: contains the sale price 80% of the time)", "",
                   "| | Contains sale price | Median width |", "|---|---|---|",
                   f"| Raw 10-90% quantile models | {gr['coverage_raw']:.1f}% | {gr['width_raw']:.1f}% |",
                   f"| Calibrated (conformal) | {gr['coverage_10_90']:.1f}% | {gr['median_width_pct']:.1f}% |"]
            md += [f"| ... {k.replace('type_group=', '').replace('price_band=', '')} ({v['n']:,}) | "
                   f"{v['coverage']:.1f}% | {v['width']:.1f}% |" for k, v in gr["by"].items()]
        for by, title in (("property_type", "By property type"), ("price_band", "By price band"),
                          ("city_region", "By neighbourhood (15 largest in the test set)")):
            t = r["breakdowns"][by]
            if t.empty:
                continue
            md += ["", f"### {title} (median abs % error)", "", _fmt_table(t, [m for m in r["overall"]])]
    (out / "results.md").write_text("\n".join(md) + "\n")


# Reference categorical palette, fixed order (dataviz skill, light mode).
SERIES = {"baseline": "#2a78d6", "gbm": "#eb6834", "nn": "#1baf7a", "tabpfn": "#eda100", "gbm_notax": "#e87ba4",
          "gbm_updates": "#008300", "gbm_text": "#6250d6", "gbm_photos": "#e34948", "gbm_all": "#52514e"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def charts(out: Path, results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "figure.facecolor": SURFACE})

    def grouped(ax, groups: list[str], series: dict[str, list[float]], fmt="{:.1f}%"):
        n = len(series)
        width = 0.8 / n
        x = np.arange(len(groups))
        for i, (m, vals) in enumerate(series.items()):
            pos = x - 0.4 + width * (i + 0.5)
            bars = ax.bar(pos, vals, width * 0.9, color=SERIES[m], label=MODEL_NAMES[m], edgecolor=SURFACE, linewidth=1)
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(x, groups)
        _style(ax)
        ax.legend(frameon=False, ncols=min(n, 4), loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8)

    # 1. Median abs % error, every model, both splits.
    names = list(results)
    models = list(next(iter(results.values()))["overall"])
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(models) + 2), 3.8))
    grouped(ax, [results[s]["label"] for s in names], {m: [results[s]["overall"][m]["mdape"] for s in names] for m in models})
    ax.set_ylabel("Median absolute % error (lower is better)")
    fig.tight_layout()
    fig.savefig(out / "error_by_model.png", dpi=160)
    plt.close(fig)

    # 2. Within 10% by price band, recent split.
    r = results.get("recent") or results[names[0]]
    t = r["breakdowns"]["price_band"]
    if not t.empty:
        fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(models) + 2), 3.8))
        grouped(ax, list(t["price_band"]), {m: list(t[m]) for m in models})
        ax.set_ylabel("Median absolute % error")
        fig.tight_layout()
        fig.savefig(out / "error_by_price_band.png", dpi=160)
        plt.close(fig)

    # 3. Predicted vs actual for the two learned models, recent split.
    learned = [m for m in ("gbm", "nn") if m in r["pred"]]
    if learned:
        fig, axes = plt.subplots(1, len(learned), figsize=(4.2 * len(learned), 4), sharey=True)
        axes = np.atleast_1d(axes)
        y = r["actual"] / 1000
        lim = (np.percentile(y, 0.5), np.percentile(y, 99.5))
        for ax, m in zip(axes, learned):
            p = r["pred"][m] / 1000
            ax.scatter(y, p, s=8, alpha=0.35, color=SERIES[m], linewidths=0)
            ax.plot(lim, lim, color=INK2, linewidth=1)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_title(f"{MODEL_NAMES[m]}: MdAPE {r['overall'][m]['mdape']:.1f}%", fontsize=10, color=INK)
            ax.set_xlabel("Sale price ($K)")
            _style(ax)
            ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        axes[0].set_ylabel("Estimate ($K)")
        fig.tight_layout()
        fig.savefig(out / "predicted_vs_actual.png", dpi=160)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/sample.csv", help="CSV path, or 'db' for DATABASE_URL")
    ap.add_argument("--out", default="results/sample")
    ap.add_argument("--models", default="baseline,gbm,nn,gbm_notax,gbm_updates,gbm_text,gbm_photos,gbm_all",
                    help="comma list of baseline,gbm,nn,tabpfn,gbm_notax,gbm_updates,gbm_text,gbm_photos,gbm_all")
    ap.add_argument("--photos", default=None, help="directory of src.photos output (data/sample_photos for the sample)")
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--shift-months", type=int, default=12)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = data.load(a.data)
    sales, cleaning = F.clean(raw)
    x = F.add_extras(F.build(sales), a.photos)
    x["price_band"] = pd.cut(x["close_price"], PRICE_BANDS, labels=BAND_LABELS, right=False).astype(str)

    results, summary = {}, {"source": a.data, "cleaning": cleaning, "splits": {}}
    for name, (train, test, desc) in splits(x, a.test_months, a.shift_months).items():
        log.info("split %s: %s (train %d, test %d)", name, desc, len(train), len(test))
        preds, extra, timing = run_split(train, test, models)
        y = test["close_price"].to_numpy()
        overall = {m: metrics(y, p) for m, p in preds.items()}
        for m, s in overall.items():
            log.info("  %-9s MdAPE %5.1f%%  within10 %5.1f%%  MAE $%s", m, s["mdape"], s["within_10"], f"{s['mae']:,.0f}")
        results[name] = {
            "desc": desc, "label": "Last 3 months" if name == "recent" else "Market shift",
            "n_train": len(train), "n_test": len(test), "overall": overall, "extra": extra,
            "breakdowns": {
                "property_type": breakdown(test, preds, "property_type"),
                "price_band": breakdown(test, preds, "price_band", min_n=10),
                "city_region": breakdown(test, preds, "city_region", top=15),
            },
            "pred": preds, "actual": y,
        }
        summary["splits"][name] = {"desc": desc, "n_train": len(train), "n_test": len(test),
                                   "overall": overall, "extra": extra, "train_seconds": timing,
                                   "breakdowns": {k: v.to_dict("records") for k, v in results[name]["breakdowns"].items()}}

    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    write_report(out, a.data, cleaning, results)
    charts(out, results)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
