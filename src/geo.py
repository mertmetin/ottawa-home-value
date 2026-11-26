"""Distances: downtown Ottawa and the nearest O-Train station open at sale time."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EARTH_KM = 6371.0
# Parliament Hill.
DOWNTOWN = (45.4236, -75.7009)
STATIONS_CSV = Path(__file__).resolve().parent.parent / "data" / "otrain_stations.csv"


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_stations(path: Path = STATIONS_CSV) -> pd.DataFrame:
    s = pd.read_csv(path)
    s["opened"] = pd.to_datetime(s["opened"])
    return s


def downtown_km(lat, lng):
    return haversine_km(np.asarray(lat, float), np.asarray(lng, float), *DOWNTOWN)


def nearest_station_km(lat, lng, dates, stations: pd.DataFrame | None = None) -> np.ndarray:
    """Distance to the nearest station already open on each row's date.

    Line 2 was closed from 2020 and reopened with new stations on 2025-01-06,
    so a 2024 sale near South Keys did not have a train yet.
    """
    st = load_stations() if stations is None else stations
    lat = np.asarray(lat, float)
    lng = np.asarray(lng, float)
    dates = pd.to_datetime(pd.Series(dates)).to_numpy()
    d = haversine_km(lat[:, None], lng[:, None], st["lat"].to_numpy()[None, :], st["lng"].to_numpy()[None, :])
    is_open = st["opened"].to_numpy()[None, :] <= dates[:, None]
    d = np.where(is_open, d, np.inf)
    out = d.min(axis=1)
    out[~np.isfinite(out)] = np.nan
    return out
