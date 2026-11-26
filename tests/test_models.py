import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import data, evaluate, features as F, train_gbm, train_nn  # noqa: E402
import make_sample  # noqa: E402


@pytest.fixture(scope="module")
def frame():
    clean, _ = F.clean(data._normalise(make_sample.generate(2500, seed=5)))
    return F.build(clean)


def test_splits_are_time_ordered(frame):
    for train, test, _ in evaluate.splits(frame).values():
        assert len(train) and len(test)
        assert train["sale_date"].max() <= test["sale_date"].min()


def test_gbm_learns_and_round_trips(frame, tmp_path):
    train, test, _ = evaluate.splits(frame)["recent"]
    g = train_gbm.fit(train, rounds=400)
    p = g.predict(test)
    assert (p["lo"] <= p["estimate"]).all() and (p["estimate"] <= p["hi"]).all()
    m = evaluate.metrics(test["close_price"].to_numpy(), p["estimate"].to_numpy())
    base = evaluate.metrics(test["close_price"].to_numpy(), F.baseline_estimate(test))
    assert m["mdape"] < base["mdape"]
    g.save(tmp_path)
    again = train_gbm.GBM.load(tmp_path).predict(test)
    np.testing.assert_allclose(again["estimate"], p["estimate"], rtol=1e-6)


def test_nn_learns_and_round_trips(frame, tmp_path):
    train, test, _ = evaluate.splits(frame)["recent"]
    n = train_nn.fit(train, epochs=40, patience=10)
    p = n.predict(test)["estimate"].to_numpy()
    assert np.isfinite(p).all()
    assert evaluate.metrics(test["close_price"].to_numpy(), p)["mdape"] < 15
    n.save(tmp_path)
    np.testing.assert_allclose(train_nn.NN.load(tmp_path).predict(test)["estimate"], p, rtol=1e-5)


def test_unseen_category_does_not_crash(frame):
    train, test, _ = evaluate.splits(frame)["recent"]
    t = test.head(5).copy()
    t["city_region"] = "9999 - Nowhere"
    g = train_gbm.fit(train, rounds=50)
    assert np.isfinite(g.predict(t)["estimate"]).all()


def test_conformal_quantile():
    s = np.arange(1, 101) / 100  # 0.01 .. 1.00
    assert train_gbm.conformal(s, 0.8) == pytest.approx(0.81)  # ceil(101 * 0.8) / 100 -> 81st score
    assert train_gbm.conformal(-s, 0.8) == 0.0                  # already covering: never narrows
    assert train_gbm.conformal(np.array([]), 0.8) == 0.0


def test_calibration_widens_range_toward_target(frame):
    train, test, _ = evaluate.splits(frame)["shift"]
    g = train_gbm.fit(train, rounds=400)
    assert g.calib["all"] >= 0
    y = test["close_price"].to_numpy()
    raw = np.exp(g.raw(test))
    cal = g.predict(test)
    cov_raw = np.mean((y >= raw["lo"]) & (y <= raw["hi"]))
    cov_cal = np.mean((y >= cal["lo"]) & (y <= cal["hi"]))
    assert cov_cal >= cov_raw
    assert cov_cal > 0.7


def test_gbm_without_taxes(frame):
    train, test, _ = evaluate.splits(frame)["recent"]
    g = train_gbm.fit(train, rounds=200, features=evaluate.NO_TAX)
    assert "tax_annual" not in g.models["mid"].feature_name()
    assert np.isfinite(g.predict(test)["estimate"]).all()
