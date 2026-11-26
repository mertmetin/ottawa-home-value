import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import data, features as F, geo  # noqa: E402
import make_sample  # noqa: E402


@pytest.fixture(scope="module")
def sample():
    return data._normalise(make_sample.generate(1500, seed=3))


@pytest.mark.parametrize("s,want", [
    ("1500-2000", 1750), ("< 700", 595), ("5000 +", 5500), ("New", 0), ("16-30", 23), ("100+", 110), (None, np.nan),
])
def test_range_mid(s, want):
    got = F.range_mid(s)
    assert (np.isnan(got) and np.isnan(want)) or got == pytest.approx(want)


def test_clean_removes_planted_problems_and_logs(sample):
    clean, report = F.clean(sample)
    steps = {r["step"]: r["removed"] for r in report}
    assert steps["not sold"] > 0 and steps["property type"] > 0 and steps["duplicate sale"] > 0
    assert steps["price range"] + steps["price outlier"] > 0
    assert set(clean["property_type"]) <= F.KEEP_TYPES
    assert (clean["mls_status"] == "Sold").all()
    assert sum(r["removed"] for r in report) == len(sample) - len(clean)


def test_market_features_use_only_earlier_sales(sample):
    clean, _ = F.clean(sample)
    x = F.build(clean)
    # Changing every sale on or after a date must not change features of rows before it.
    cut = x["sale_date"].quantile(0.6)
    tampered = clean.copy()
    later = tampered["sale_date"] >= cut
    tampered.loc[later, "close_price"] *= 3
    y = F.build(tampered)
    before = x["sale_date"] < cut
    cols = ["area_ppsf_90d", "city_ppsf_90d", "comp_price_med", "comp_ppsf_med"]
    pd.testing.assert_frame_equal(x.loc[before, cols].reset_index(drop=True), y.loc[before, cols].reset_index(drop=True))
    # And same-day sales are not used either: a row's own price never feeds its comps.
    one = clean.copy()
    one.loc[one.index[-1], "close_price"] *= 5
    z = F.build(one)
    assert z.iloc[-1]["comp_price_med"] == x.iloc[-1]["comp_price_med"]


def test_excluded_features_are_absent():
    banned = {"list_price", "original_list_price", "days_on_market", "list_to_sale"}
    assert not banned & set(F.FEATURES)


def test_station_distance_respects_opening_date():
    # Next to South Keys (Line 2, reopened 2025-01-06).
    lat, lng = 45.3545, -75.6475
    before = geo.nearest_station_km([lat], [lng], ["2024-12-15"])[0]
    after = geo.nearest_station_km([lat], [lng], ["2025-02-01"])[0]
    assert after < 0.1 < before


def test_db_record_mapping():
    row = {"listing_key": "X1", "mls_status": "Sold", "property_sub_type": "Semi-Detached ", "close_price": 650000.0,
           "contract_date": None, "close_date": "2026-01-30", "lat": 45.4, "lng": -75.7,
           "record": {"LivingAreaRange": "1100-1500", "Basement": ["Full", "Finished"], "StreetNumber": "12",
                      "StreetName": "Main", "StreetSuffix": "St", "UnitNumber": "3", "PurchaseContractDate": "2026-01-02",
                      "ArchitecturalStyle": ["2-Storey"], "CityRegion": "4401 - Glebe"}}
    r = data.from_record(row)
    assert r["property_type"] == "Semi-Detached" and r["basement"] == "Full, Finished"
    assert r["sale_date"] == "2026-01-02" and r["address"] == "3-12 Main St" and r["style"] == "2-Storey"


def test_update_features_from_descriptions():
    from src import text

    r = pd.Series(["Stunning! New roof 2021, furnace & A/C (2019). HWT rental. Priced at $649,900.",
                   "Handyman special, sold as-is. Original kitchen.",
                   "Fully renovated kitchen and bathrooms, newer windows. Owned hot water tank.",
                   None])
    f = text.update_features(r, pd.Series([2026] * 4))
    assert f.loc[0, "age_roof"] == 5 and f.loc[0, "age_furnace"] == 7 and f.loc[0, "age_ac"] == 7
    assert f.loc[0, "txt_hwt_rental"] == 1 and np.isnan(f.loc[0, "age_hwt"])  # no year borrowed across sentences
    assert f.loc[1, "txt_tlc"] == 1 and f.loc[1, "upd_kitchen"] == 0
    assert f.loc[2, "upd_kitchen"] == 1 and f.loc[2, "upd_bath"] == 1 and f.loc[2, "upd_windows"] == 1
    assert f.loc[2, "txt_hwt_owned"] == 1
    assert f.loc[3].isna().all()


def test_prices_never_reach_text_features():
    from src import text

    assert "649" not in text._strip_prices("listed at $649,900 or 650k")


def test_new_property_fields():
    df = pd.DataFrame({"foundation": ["Poured Concrete", "Concrete Block", "Stone", None],
                       "pool": ["Inground, Salt", "None", None, "Above Ground"],
                       "sewer": ["Septic", "Sewer", None, "Sewer"]})
    assert [F._foundation(v) for v in df["foundation"]] == ["concrete", "block", "stone", None]
    assert F._has(df["pool"], "Inground").tolist()[:2] == [1.0, 0.0]
    assert F._has(df["sewer"], "Septic").tolist()[0] == 1.0


def test_extras_merge_photos(sample, tmp_path):
    import make_sample

    clean, _ = F.clean(sample)
    raw = make_sample.generate(300, seed=3)
    make_sample.synthetic_photos(raw, tmp_path)
    x = F.add_extras(F.build(clean.head(300)), str(tmp_path))
    assert len(x) == 300
    assert {"zs_kitchen_updated_mean", "img_pc_0", "upd_roof", "txt_svd_0"} <= set(x.columns)
