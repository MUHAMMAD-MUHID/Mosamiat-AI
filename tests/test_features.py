import json
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from test_collect import CITY, response_payload

from mosamiat.backfill import METRICS, backfill, digest
from mosamiat.collect import CollectionError
from mosamiat.features.pipeline import build, load_definitions, transform
from mosamiat.features.target_builder import build_targets
from mosamiat.features.time_features import encode_wind

CONFIG = Path("configs/features.toml")


@pytest.fixture
def values():
    index = pd.date_range("2025-01-01", periods=80, freq="h", tz="UTC")
    frame = pd.DataFrame({metric: np.arange(80, dtype=float) for metric in METRICS}, index=index)
    frame["rain_probability"] = np.nan
    return frame


def test_current_and_future_values_cannot_change_features(values):
    definitions = load_definitions(CONFIG)
    original = transform(values, definitions, CITY.timezone)
    changed = values.copy()
    changed.iloc[40:] = 999
    altered = transform(changed, definitions, CITY.timezone)
    pd.testing.assert_frame_equal(original.iloc[:41], altered.iloc[:41])
    # Inference uses the same function on the prefix; no future labels required.
    pd.testing.assert_frame_equal(
        original.iloc[:41], transform(values.iloc[:41], definitions, CITY.timezone)
    )
    row = original.iloc[30]
    assert row.temperature_c_lag_1h == 29
    assert row.temperature_c_lag_3h == 27
    assert row.temperature_c_lag_24h == 6
    assert row.temperature_c_mean_3h == 28
    assert row.temperature_c_variance_3h == pytest.approx(2 / 3)
    assert row.temperature_c_change_1h == 1
    assert len(original) == len(values)
    assert original.iloc[-1].temperature_c_lag_1h == 78


def test_missingness_and_age_do_not_fill_or_peek_ahead(values):
    values.loc[values.index[3:5], "temperature_c"] = np.nan
    frame = transform(values, load_definitions(CONFIG), CITY.timezone)
    assert frame.iloc[4].temperature_c_missing
    assert frame.iloc[4].temperature_c_age_hours == 2
    assert frame.iloc[5].temperature_c_age_hours == 3
    assert np.isnan(frame.iloc[5].temperature_c_mean_3h)
    assert frame.rain_probability_lag_1h.isna().all()
    assert frame.rain_probability_age_hours.isna().all()
    assert np.isnan(frame.iloc[0].temperature_c_age_hours)
    assert np.isnan(frame.iloc[23].temperature_c_mean_24h)


def test_local_calendar_and_circular_wind(values):
    values["wind_direction_deg"] = 0.0
    values.iloc[1, values.columns.get_loc("wind_direction_deg")] = 360
    wind = encode_wind(values)
    assert np.allclose(wind.iloc[:2].wind_direction_sin, 0, atol=1e-12)
    assert np.allclose(wind.iloc[:2].wind_direction_cos, 1)
    frame = transform(values, load_definitions(CONFIG), CITY.timezone)
    assert frame.iloc[0].hour == 5  # Midnight UTC is 05:00 in configured city.
    assert frame.iloc[0].day_of_week == 2
    assert not any(name.startswith("wind_direction_deg") for name in frame)


def test_all_horizons_align_and_latest_rows_keep_missing_labels(values):
    targets = build_targets(values, list(range(1, 25)))
    first = targets[targets.reference_at_utc == values.index[30]]
    assert list(first.horizon_hours) == list(range(1, 25))
    assert list(first.target_temperature_c) == list(range(31, 55))
    assert (first.target_at_utc - first.reference_at_utc).dt.total_seconds().tolist() == [
        hour * 3600 for hour in range(1, 25)
    ]
    last = targets[targets.reference_at_utc == values.index[-1]]
    assert len(last) == 24
    assert last.target_temperature_c_missing.all()
    assert targets.target_rain_probability_missing.all()


@pytest.mark.parametrize("kind", ["gap", "duplicate", "naive", "local", "infinite", "off_hour"])
def test_invalid_hourly_input_is_rejected(values, kind):
    if kind == "gap":
        values = values.drop(values.index[4])
    elif kind == "duplicate":
        values = pd.concat([values.iloc[:1], values])
    elif kind == "naive":
        values.index = values.index.tz_localize(None)
    elif kind == "local":
        values.index = values.index.tz_convert(CITY.timezone)
    elif kind == "infinite":
        values.iloc[0, 0] = np.inf
    else:
        values.index += pd.Timedelta(minutes=1)
    with pytest.raises(CollectionError, match="continuous hourly"):
        transform(values, load_definitions(CONFIG), CITY.timezone)


def test_build_validates_sources_and_publishes_repeatable_reference_artifacts(tmp_path):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=response_payload(request))
        )
    ) as client:
        dataset = backfill(CITY, tmp_path / "source", client, date(2025, 1, 1), date(2025, 1, 2))
    destination = tmp_path / "features"
    build(dataset, destination, CITY, CONFIG)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["feature_rows"] == 48
    assert manifest["target_rows"] == 48 * 24
    assert manifest["is_ground_truth"] is False
    assert manifest["operational_point_in_time_safe"] is False
    assert manifest["source_missing"]["rain_probability"] == 48
    hashes = manifest["sha256"]
    assert all(digest(destination / name) == checksum for name, checksum in hashes.items())
    build(dataset, destination, CITY, CONFIG)
    assert all(digest(destination / name) == checksum for name, checksum in hashes.items())
    with pytest.raises(CollectionError, match="overwrite"):
        build(dataset, dataset, CITY, CONFIG)
    (dataset / "hourly.csv").write_text("corrupted")
    with pytest.raises(CollectionError, match="integrity"):
        build(dataset, destination, CITY, CONFIG)


@pytest.mark.parametrize("content", ["lags_hours = [0]", "lags_hours = [3, 1]", "schema_version=2"])
def test_invalid_feature_definitions_fail(tmp_path, content):
    config = tmp_path / "features.toml"
    config.write_text(
        "schema_version=1\n" + content if "schema_version" not in content else content
    )
    with pytest.raises(CollectionError):
        load_definitions(config)
