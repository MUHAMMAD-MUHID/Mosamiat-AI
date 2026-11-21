import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from test_collect import CITY, response_payload

from mosamiat.backfill import METRICS, backfill, digest
from mosamiat.collect import CollectionError
from mosamiat.eda.analysis import analyze, circular_summary
from mosamiat.eda.pipeline import build
from mosamiat.features.pipeline import build as build_features
from mosamiat.reference import ReferenceDataset, load_reference


def reference(values):
    """Small source records with honest provider metadata for numerical tests."""
    rows = (
        values.rename_axis("valid_at_utc")
        .reset_index()
        .melt(id_vars="valid_at_utc", var_name="metric", value_name="value")
    )
    rows["source"] = "open-meteo"
    rows["product"] = "test-model"
    rows["data_kind"] = "reanalysis"
    rows["unit"] = rows.metric.map(METRICS)
    rows["aqi_scale"] = np.where(rows.metric.eq("aqi_value"), "US_AQI", None)
    rows["retrieved_at_utc"] = pd.Timestamp("2026-11-20T00:00:00Z")
    return ReferenceDataset(rows, values, ())


@pytest.fixture
def values():
    index = pd.date_range("2025-01-01", periods=8, freq="h", tz="UTC")
    frame = pd.DataFrame({metric: np.arange(8, dtype=float) for metric in METRICS}, index=index)
    frame["rain_probability"] = np.nan
    return frame


def test_coverage_reindexes_gaps_without_filling_values(values):
    values.loc[values.index[[2, 4]], "temperature_c"] = np.nan
    values = values.drop(values.index[3])
    result = analyze(reference(values), CITY.timezone)
    assert result.summary["expected_hours"] == 8
    assert result.summary["source_hours"] == 7
    assert result.summary["missing_timestamps"] == 1
    assert result.metrics.loc["temperature_c", "longest_gap_hours"] == 3
    assert result.metrics.loc["temperature_c", "missing_pct"] == 37.5
    assert result.metrics.loc["rain_probability", "missing_pct"] == 100
    assert result.metrics.loc["rain_probability", "longest_gap_hours"] == 8
    assert result.values.loc[pd.Timestamp("2025-01-01T03:00:00Z")].isna().all()
    assert result.summary["is_ground_truth"] is False


def test_declared_range_includes_missing_edge_timestamps(values):
    dataset = replace(reference(values.iloc[1:-1]), expected_index=values.index)
    result = analyze(dataset, CITY.timezone)
    assert result.summary["missing_timestamps"] == 2
    assert result.summary["expected_hours"] == 8
    assert result.metrics.loc["temperature_c", "missing"] == 2


def test_circular_wind_wraps_and_excludes_calm_hours(values):
    values["wind_direction_deg"] = [359, 1, 180, *([np.nan] * 5)]
    values["wind_speed_mps"] = [2, 2, 0, *([np.nan] * 5)]
    result = analyze(reference(values), CITY.timezone)
    wind = result.summary["wind"]
    assert wind["count"] == 2
    assert wind["calm_hours"] == 1
    mean = wind["mean_direction_deg"]
    assert min(mean, 360 - mean) < 1e-8
    assert wind["resultant_length"] == pytest.approx(np.cos(np.deg2rad(1)))
    assert np.isnan(result.metrics.loc["wind_direction_deg", "mean"])
    assert "wind_direction_deg" not in result.correlations
    assert "wind_direction_sin" in result.correlations
    assert result.correlation_counts.loc["wind_direction_sin", "wind_direction_sin"] == 2
    assert circular_summary(pd.Series([0, 180]))["mean_direction_deg"] is None


def test_pairwise_support_and_constant_series_are_explicit(values):
    values["temperature_c"] = [1, 2, 3, *([np.nan] * 5)]
    values["humidity_pct"] = [2, 4, np.nan, 8, *([np.nan] * 4)]
    values["uv_index"] = 1.0
    result = analyze(reference(values), CITY.timezone)
    assert result.correlation_counts.loc["temperature_c", "humidity_pct"] == 2
    assert np.isnan(result.correlations.loc["temperature_c", "humidity_pct"])
    assert result.correlations.uv_index.isna().all()
    assert not result.metrics.loc["uv_index", "iqr_available"]
    assert result.hourly_profile.loc[5, "temperature_c_count"] == 1
    assert result.hourly_profile.loc[0, "temperature_c_count"] == 0


def test_iqr_review_flags_preserve_legitimate_extremes(values):
    values["temperature_c"] = [0, 1, 2, 3, 4, 100, np.nan, np.nan]
    values["precipitation_mm"] = [0, 0, 0, 0, 0, 0, 0, 1]
    result = analyze(reference(values), CITY.timezone)
    assert result.metrics.loc["temperature_c", "iqr_review_count"] == 1
    assert result.metrics.loc["temperature_c", "iqr_upper"] == 7.5
    assert result.anomalies.value.tolist() == [100]
    assert result.values.temperature_c.max() == 100
    assert not result.metrics.loc["precipitation_mm", "iqr_available"]
    assert result.summary["rain"]["zero_fraction_of_present"] == 7 / 8
    assert result.summary["rain"]["observed_sum_mm"] == 1
    assert not result.daily_precipitation.complete_local_day.any()


def test_all_missing_rain_is_not_zero_rain(values):
    values["precipitation_mm"] = np.nan
    result = analyze(reference(values), CITY.timezone)
    assert result.summary["rain"]["observed_sum_mm"] is None
    assert result.summary["rain"]["zero_fraction_of_present"] is None
    assert result.daily_precipitation.observed_sum_mm.isna().all()
    assert any("rain probability" in warning for warning in result.summary["warnings"])


def test_daily_precipitation_handles_a_25_hour_local_day(values):
    index = pd.date_range("2026-11-01T04:00:00Z", periods=25, freq="h")
    values = values.reindex(index).fillna(1.0)
    result = analyze(reference(values), "America/New_York")
    day = result.daily_precipitation.iloc[0]
    assert len(result.daily_precipitation) == 1
    assert day.expected_day_hours == 25
    assert day.observed_sum_mm == 25
    assert day.complete_local_day


def test_report_renders_all_metrics_and_binds_artifact_hashes(tmp_path):
    pytest.importorskip("matplotlib")
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=response_payload(request))
        )
    ) as client:
        dataset = backfill(CITY, tmp_path / "source", client, date(2025, 1, 1), date(2025, 1, 2))
    output = build(dataset, tmp_path / "eda", CITY)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["expected_hours"] == 48
    assert len(manifest["charts"]) == 8
    assert all(digest(output / name) == checksum for name, checksum in manifest["sha256"].items())
    report = (output / "report.html").read_text(encoding="utf-8")
    assert report.count("data:image/png;base64,") == 8
    assert set(pd.read_csv(output / "metrics.csv").metric) == set(METRICS)
    assert not (output / ".lock").exists()
    with pytest.raises(CollectionError, match="overwrite"):
        build(dataset, dataset, CITY)
    (dataset / "hourly.csv").write_text("corrupt")
    with pytest.raises(CollectionError, match="integrity"):
        build(dataset, output, CITY)


def test_missing_edge_window_is_audited_and_feature_generation_rejects_it(tmp_path):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=response_payload(request))
        )
    ) as client:
        dataset = backfill(CITY, tmp_path / "source", client, date(2025, 1, 31), date(2025, 2, 1))
    manifest = json.loads((dataset / "manifest.json").read_text())
    manifest["windows"] = manifest["windows"][1:]
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    result = analyze(load_reference(dataset, CITY), CITY.timezone)
    assert result.summary["expected_hours"] == 48
    assert result.summary["missing_timestamps"] == 24
    with pytest.raises(CollectionError, match="coverage differs"):
        build_features(dataset, tmp_path / "features", CITY, Path("configs/features.toml"))


def test_render_failure_marks_report_failed_and_releases_lock(values, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    dataset = tmp_path / "source"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}")
    monkeypatch.setattr("mosamiat.eda.pipeline.load_reference", lambda *_: reference(values))

    def fail(*_):
        raise RuntimeError("simulated rendering failure")

    monkeypatch.setattr("mosamiat.eda.plots.render_charts", fail)
    output = tmp_path / "eda"
    with pytest.raises(RuntimeError, match="rendering failure"):
        build(dataset, output, CITY)
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    assert not (output / ".lock").exists()
