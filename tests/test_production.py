import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import drift, homeowner, pipeline, registry as R  # noqa: E402
import make_sample  # noqa: E402

ROUNDS = 300


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A sample CSV and empty model / log dirs, wired through the env vars."""
    d = tmp_path_factory.mktemp("prod")
    csv = d / "sales.csv"
    make_sample.generate(3000, seed=11).to_csv(csv, index=False)
    mp = pytest.MonkeyPatch()
    mp.setenv("OHV_MODELS", str(d / "models"))
    mp.setenv("OHV_LOGS", str(d / "logs"))
    yield {"csv": str(csv), "dir": d}
    mp.undo()


@pytest.fixture(scope="module")
def first(world):
    return pipeline.retrain(world["csv"], rounds=ROUNDS)


def reg():
    return R.Registry(pipeline.models_dir() / "registry")


def test_first_retrain_serves_a_version(first):
    assert first["passed"] and first["served"] is None
    cur = reg().current()
    assert cur.name == first["version"]
    card = cur.card
    assert card["feature_set"] == "homeowner" and card["gate"]["candidate"]["n"] > 0
    assert pipeline.snapshot_path().exists()
    # licensed fields stay out of the snapshot the API loads
    snap = pd.read_pickle(pipeline.snapshot_path())
    assert "remarks" not in snap and "address" not in snap


def test_gate_rules():
    good = {"n": 300, "mdape": 6.0, "within_10": 70.0, "coverage": 78.0}
    assert pipeline.gate(good, None)[0]
    assert pipeline.gate(good, {**good, "mdape": 5.9})[0]          # within the slack
    ok, why = pipeline.gate(good, {**good, "mdape": 5.5})
    assert not ok and "median error" in why[0]
    assert not pipeline.gate({**good, "coverage": 60.0}, None)[0]
    assert not pipeline.gate({**good, "mdape": 14.0}, None)[0]
    assert not pipeline.gate({"n": 0}, None)[0]


def test_failed_gate_keeps_the_served_model(first, world, monkeypatch):
    before = reg().current().name
    monkeypatch.setattr(pipeline, "gate", lambda c, s: (False, ["test says no"]))
    out = pipeline.retrain(world["csv"], rounds=ROUNDS)
    assert not out["passed"] and "version" not in out
    assert reg().current().name == before
    last = R.read_jsonl(reg().root / "promotions.jsonl")[-1]
    assert last["event"] == "rejected" and last["failures"] == ["test says no"]


def test_nightly_logs_drift_and_recalibrates(first, world):
    row = pipeline.nightly(world["csv"])
    assert row["version"] == reg().current().name
    assert set(row["psi_sales_90d"]) >= {"sqft_mid", "property_type"}
    cur = reg().current()
    assert (cur.path / "calib.json").exists() == (row["recalibrated"] is not None)
    logged = R.read_jsonl(pipeline.logs_dir() / "drift.jsonl")
    assert logged[-1]["version"] == row["version"]


def test_psi_sees_a_shift():
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"sqft_mid": rng.normal(1500, 300, 3000), "property_type": ["Detached"] * 3000})
    ref = drift.reference(a.assign(**{c: np.nan for c in drift.NUMERIC if c != "sqft_mid"}, type_group="freehold"))
    same = drift.psi(ref, a.sample(800, random_state=1))
    moved = drift.psi(ref, a.assign(sqft_mid=a["sqft_mid"] * 1.3, property_type="Condo Apartment"))
    assert same["sqft_mid"] < 0.05 and moved["sqft_mid"] > 0.25 and moved["property_type"] > 0.25


def test_answers_map_to_features():
    a = {"property_type": "Townhome", "hv_condo_fee": 350, "hv_upd_roof": "lt5", "hv_upd_windows": "orig",
         "hv_hwt": "rental", "hv_garage": "1", "hv_central_air": 1, "hv_sqft": "unknown"}
    assert homeowner.property_type(a) == "Condo Townhouse"
    assert homeowner.property_type({"property_type": "Condo"}) == "Condo Apartment"
    with pytest.raises(ValueError):
        homeowner.property_type({"property_type": "Castle"})
    row = homeowner.canonical(a, pd.Timestamp("2026-11-26"))
    assert row["sqft_range"] is None and row["garage_spaces"] == 1 and row["cooling"] == "Central Air"
    upd = homeowner.update_columns(a, home_age=22.0)
    assert upd["upd_roof"] == 1 and upd["age_roof"] == 2.5
    assert upd["upd_windows"] == 0 and upd["age_windows"] == 22.0
    assert upd["txt_hwt_rental"] == 1 and upd["upd_furnace"] == 0 and np.isnan(upd["age_furnace"])
    assert homeowner.answered(a) == ["hv_upd_roof", "hv_upd_windows", "hv_hwt", "hv_garage", "hv_central_air"]
    assert homeowner.update_columns({"condition": "Needs work"}, 10.0)["txt_tlc"] == 1
    assert homeowner.update_columns({"condition": "Newly renovated"}, 10.0)["txt_turnkey"] == 1
    assert homeowner.update_columns({"condition": "Good"}, 10.0)["txt_turnkey"] == 0


def test_api_estimates_and_reloads(first, world):
    from fastapi.testclient import TestClient

    from src import api

    snap = pd.read_pickle(pipeline.snapshot_path())
    town = snap[snap["property_type"] == "Att/Row/Townhouse"].iloc[-1]
    c = TestClient(api.app)
    h = c.get("/health").json()
    assert h["version"] == reg().current().name
    body = {"property_type": "Townhome", "lat": float(town["lat"]), "lng": float(town["lng"]), "bedrooms": "3",
            "bathrooms": 3, "hv_sqft": "1500-2000", "hv_age": "16-30", "hv_upd_roof": "lt5",
            "hv_upd_kitchen": "gt20", "hv_garage": "1", "hv_central_air": 1}
    r = c.post("/v1/estimate", json=body)
    assert r.status_code == 200, r.text
    e = r.json()
    assert e["low"] <= e["estimate"] <= e["high"] and 150_000 < e["estimate"] < 3_000_000
    assert e["city_region"] and e["comps"] > 0
    assert {x["key"] for x in e["answers"]} == {"hv_upd_roof", "hv_upd_kitchen", "hv_garage", "hv_central_air", "hv_age"}
    assert "close_price" not in json.dumps(e)
    bigger = c.post("/v1/estimate", json={**body, "hv_sqft": "2500-3000", "explain": False}).json()
    assert bigger["estimate"] > e["estimate"]
    assert c.post("/v1/estimate", json={**body, "property_type": "Castle"}).status_code == 422
    assert c.post("/v1/estimate", json={**body, "lat": 10}).status_code == 422
    logged = R.read_jsonl(pipeline.logs_dir() / "requests.jsonl")
    assert len(logged) >= 2 and "address" not in logged[-1]["x"] and logged[-1]["x"]["lat"] == round(body["lat"], 2)


def test_api_key(first, monkeypatch):
    from fastapi.testclient import TestClient

    from src import api

    monkeypatch.setenv("OHV_API_KEY", "s3cret")
    c = TestClient(api.app)
    body = {"property_type": "Detached", "lat": 45.3, "lng": -75.7}
    assert c.post("/v1/estimate", json=body).status_code == 401
    assert c.post("/v1/estimate", json=body, headers={"X-API-Key": "s3cret"}).status_code == 200


def test_rollback(first, world):
    r = reg()
    a = r.current().name
    out = pipeline.retrain(world["csv"], rounds=ROUNDS, force=True)
    assert r.current().name == out["version"] != a
    assert r.rollback() == a and r.current().name == a
