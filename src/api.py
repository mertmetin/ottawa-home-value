"""The estimate API: what is this home worth today?

    uvicorn src.api:app --host 0.0.0.0 --port 8000

POST /v1/estimate with the homeowner's answers (the keys of the questions on
mertmetin.ca/home-valuation) plus lat / lng:

    {"property_type": "Townhome", "lat": 45.27, "lng": -75.74, "bedrooms": 3, "bathrooms": 3,
     "hv_sqft": "1500-2000", "hv_age": "16-30", "hv_upd_roof": "lt5", "hv_central_air": 1}

->  {"estimate": 655000, "low": 590000, "high": 720000, "version": "v20261126-...",
     "answers": [{"key": "hv_upd_roof", "effect": 9000}, ...], ...}

`answers` says what each answer moved the estimate by (the estimate with that
answer left out, minus the estimate), biggest first.

The server's model and its history snapshot come from src.pipeline; when the
nightly job or a promotion changes them, the API picks them up within a
minute without a restart. Requests are logged for input-drift checks, without
the address (lat / lng rounded to about a kilometre).

Set OHV_API_KEY to require an X-API-Key header. Prices and comps never leave
the API: only the estimate, its range and counts.
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
import time

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from . import drift, homeowner, pipeline, registry as R

log = logging.getLogger("api")
RELOAD_SECONDS = 60


class Answers(BaseModel):
    model_config = ConfigDict(extra="allow")  # every hv_* answer passes through to homeowner.build

    property_type: str = Field(description="MLS sub-type or the site's type: Detached, Semi-detached, Townhome, Condo")
    lat: float = Field(ge=44.9, le=45.6)
    lng: float = Field(ge=-76.4, le=-75.2)
    bedrooms: str | float | None = None
    bathrooms: str | float | None = None
    city_region: str | None = None
    as_of: str | None = Field(None, description="value on this date (default today)")
    explain: bool = True


class Served:
    """The served model + history, reloaded when the pipeline changes them."""

    def __init__(self):
        self.lock = threading.Lock()
        self.stamp = -1.0
        self.checked = 0.0
        self.version = None
        self.g = None
        self.hist = None

    def get(self):
        now = time.time()
        if now - self.checked < RELOAD_SECONDS and self.g is not None:
            return self
        with self.lock:
            self.checked = now
            reg = R.Registry(pipeline.models_dir() / "registry")
            snap = pipeline.snapshot_path()
            stamp = reg.stamp() + (snap.stat().st_mtime if snap.exists() else 0)
            if stamp != self.stamp:
                cur = reg.current()
                if cur is None or not snap.exists():
                    raise HTTPException(503, "no model has been trained yet")
                self.g, self.version, self.card = cur.load(), cur.name, cur.card
                self.hist = pd.read_pickle(snap)
                self.stamp = stamp
                log.info("serving %s with %d past sales", self.version, len(self.hist))
        return self


served = Served()
app = FastAPI(title="Ottawa home value", version="1")


def auth(x_api_key: str | None = Header(None)):
    want = os.environ.get("OHV_API_KEY")
    if want and not hmac.compare_digest(x_api_key or "", want):
        raise HTTPException(401, "bad API key")


@app.get("/health")
def health():
    s = served.get()
    return {"ok": True, "version": s.version, "data_end": s.card.get("data_end"),
            "history_sales": int(len(s.hist)), "last_sale": str(s.hist["sale_date"].max().date())}


def _round(v: float, to: int = 1000) -> int:
    return int(round(float(v) / to) * to)


@app.post("/v1/estimate", dependencies=[Depends(auth)])
def estimate(body: Answers):
    s = served.get()
    a = body.model_dump()
    as_of = pd.Timestamp(body.as_of) if body.as_of else pd.Timestamp.now().normalize()
    try:
        x = homeowner.build(a, s.hist, as_of)
    except ValueError as e:
        raise HTTPException(422, str(e))
    p = s.g.predict(x).iloc[0]
    out = {
        "estimate": _round(p["estimate"]), "low": _round(p["lo"]), "high": _round(p["hi"]),
        "version": s.version, "property_type": x["property_type"].iloc[0],
        "city_region": x["city_region"].iloc[0] if pd.notna(x["city_region"].iloc[0]) else None,
        "comps": int(x["comp_n"].fillna(0).iloc[0]),
        "comps_km": round(float(x["comp_dist_km"].iloc[0]), 2) if pd.notna(x["comp_dist_km"].iloc[0]) else None,
        "as_of": str(as_of.date()),
    }
    if body.explain:
        keys, var = homeowner.variants(x, a)
        if keys:
            without = s.g.predict(var)["estimate"].to_numpy()
            eff = [{"key": k, "effect": _round(p["estimate"] - w, 500)} for k, w in zip(keys, without)]
            out["answers"] = sorted(eff, key=lambda e: -abs(e["effect"]))
        else:
            out["answers"] = []
    _log_request(s.version, x, out)
    return out


def _log_request(version: str, x: pd.DataFrame, out: dict) -> None:
    row = {}
    for c in drift.NUMERIC + drift.CATEGORICAL:
        v = x[c].iloc[0]
        if pd.isna(v):
            continue
        row[c] = round(float(v), 2) if c in ("lat", "lng") else (v if isinstance(v, str) else float(v))
    try:
        R.append_jsonl(pipeline.logs_dir() / "requests.jsonl", {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "version": version, "x": row,
            "estimate": out["estimate"], "width": round((out["high"] - out["low"]) / max(out["estimate"], 1), 3)})
    except OSError as e:
        log.warning("request log: %s", e)
