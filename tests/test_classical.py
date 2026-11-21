import json
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from test_collect import CITY, response_payload

from mosamiat.backfill import backfill, digest
from mosamiat.collect import CollectionError
from mosamiat.modeling.baseline import FORECAST_METRICS
from mosamiat.modeling.classical import evaluate
from mosamiat.modeling.rain_labels import ENDPOINT, events
from mosamiat.reference import load_reference


def test_complete_horizon_targets_purging_and_wind_wrap(monkeypatch):
    forest = pytest.importorskip("mosamiat.modeling.forest")
    monkeypatch.setitem(forest.PARAMETERS, "n_estimators", 3)
    times = pd.date_range("2025-01-01", periods=100, freq="h", tz="UTC")
    values = pd.DataFrame({m: np.arange(100, dtype=float) for m in FORECAST_METRICS}, index=times)
    values.wind_direction_deg = np.where(np.arange(100) % 2, 359.0, 1.0)
    values.wind_speed_mps = 3.0
    features = pd.DataFrame(
        {"lag": np.arange(100, dtype=float), "holdout_only": np.nan}, index=times
    )
    features.iloc[80:, 1] = 1
    features.iloc[10, 0] = np.nan
    rain = pd.Series(0.0, index=times)
    fitted = forest.fit(features, values, rain, times[80], warmup=3)
    assert fitted.columns == ["lag"]
    assert fitted.imputer.statistics_[0] == features.lag.iloc[3:56].median()
    assert fitted.training_counts["temperature_c"] == 80 - 3 - 24
    output = fitted.predict(features.iloc[80:81])
    assert output.target_at_utc.tolist() == list(pd.date_range(times[81], periods=24, freq="h"))
    assert output.rain_probability.eq(0).all()
    assert ((output.wind_direction_deg < 2) | (output.wind_direction_deg > 358)).all()
    # With y(t)=t, every tree leaf has exactly 1 between adjacent horizon predictions.
    np.testing.assert_allclose(np.diff(output.temperature_c), 1)
    values.iloc[80:] = 800
    features.iloc[80:] = 700
    rain.iloc[80:] = 1
    changed = forest.fit(features, values, rain, times[80], warmup=3)
    pd.testing.assert_frame_equal(
        fitted.predict(features.tail(1)), changed.predict(features.tail(1))
    )
    with pytest.raises(CollectionError, match="Insufficient training"):
        forest.fit(features, values, rain, times[20], warmup=3)
    with pytest.raises(CollectionError, match="continuous UTC"):
        forest.fit(
            features.drop(times[10]), values.drop(times[10]), rain.drop(times[10]), times[80], 3
        )


def test_snapshot_to_saved_forest_with_verified_rain_labels(tmp_path, monkeypatch):
    forest = pytest.importorskip("mosamiat.modeling.forest")
    import joblib

    monkeypatch.setitem(forest.PARAMETERS, "n_estimators", 2)
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=response_payload(request))
        )
    ) as client:
        dataset = backfill(CITY, tmp_path / "data", client, date(2025, 1, 1), date(2025, 1, 8))
    reference = load_reference(dataset, CITY)
    path = reference.paths[0] / "weather.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    payload = raw["response"].copy()
    hours = payload["hourly"]["time"]
    rain = [0.1 if n % 7 == 0 else 0.0 for n in range(len(hours))]
    rain[-2] = None
    payload.update(
        hourly={"time": hours, "rain": rain}, hourly_units={"time": "unixtime", "rain": "mm"}
    )
    chunk = dict(
        endpoint=ENDPOINT,
        weather_sha256=digest(path),
        parameters=raw["parameters"] | {"hourly": "rain"},
        retrieved_at_utc=raw["retrieved_at_utc"],
        response=payload,
    )
    document = dict(schema_version=1, windows=[chunk])
    labels = tmp_path / "rain.json"
    labels.write_text(json.dumps(document))
    rain_series = events(document, reference, CITY)
    assert rain_series.iloc[0] == 1 and rain_series.iloc[1] == 0 and pd.isna(rain_series.iloc[-2])
    out = tmp_path / "scores"
    cfg = Path("configs/features.toml")
    manifest = evaluate(dataset, out, CITY, labels, cfg)
    assert manifest["status"] == "complete" and not manifest["is_ground_truth"]
    assert all(digest(out / name) == sha for name, sha in manifest["sha256"].items())
    assert set(joblib.load(out / "model.joblib").estimators) == set(FORECAST_METRICS)
    cases = pd.read_csv(out / "cases.csv")
    wet = cases[cases.metric.eq("rain_probability") & cases.eligible]
    assert len(wet) > 0 and wet.random_forest.between(0, 1).all()
    assert set(wet.actual) == {0, 1} and set(wet.seasonal_naive) == {0, 1}
    assert cases.loc[cases.eligible, "random_forest"].notna().all()
    assert len(pd.read_csv(out / "metrics.csv")) == 2 * 2 * 8 * 24
    chunk["response"]["latitude"] += 1
    with pytest.raises(CollectionError, match="grid"):
        events(document, reference, CITY)
