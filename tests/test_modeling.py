import json
from datetime import date

import httpx
import numpy as np
import pandas as pd
import pytest
from test_collect import CITY, response_payload

from mosamiat.backfill import backfill, digest
from mosamiat.collect import CollectionError
from mosamiat.features.target_builder import build_targets
from mosamiat.modeling.baseline import FORECAST_METRICS, baseline_cases
from mosamiat.modeling.evaluate import evaluate
from mosamiat.modeling.metrics import metrics, score_predictions
from mosamiat.modeling.splits import split_targets, validate_split_order


def partitions(values):
    times = values.index
    return split_targets(
        build_targets(values, list(range(1, 25))),
        times[80],
        times[140],
        times[-1] + pd.Timedelta(hours=1),
    )


@pytest.fixture
def values():
    times = pd.date_range("2025-01-01", periods=200, freq="h", tz="UTC")
    return pd.DataFrame({name: np.arange(200.0) for name in FORECAST_METRICS}, index=times)


def test_splits_purge_overlapping_labels_and_reject_random_partition(values):
    folds = partitions(values)
    assert folds["train"].target_at_utc.max() == values.index[79]
    assert folds["validation"].reference_at_utc.min() == values.index[80]
    assert folds["validation"].target_at_utc.max() == values.index[139]
    assert folds["test"].reference_at_utc.min() == values.index[140]
    shuffled = pd.concat(folds.values()).sample(frac=1, random_state=7)
    with pytest.raises(CollectionError, match="overlap"):
        validate_split_order(
            dict(
                train=shuffled.iloc[:1000],
                validation=shuffled.iloc[1000:2000],
                test=shuffled.iloc[2000:],
            )
        )


def test_seasonal_forecast_repeats_only_completed_hours(values):
    cases = baseline_cases(values, partitions(values)).reset_index()
    first = cases[(cases.reference_at_utc == values.index[80]) & (cases.metric == "temperature_c")]
    assert first.horizon_hours.tolist() == list(range(1, 25))
    assert first.seasonal_naive.tolist() == [*range(57, 80), 56]
    assert first.actual.tolist() == list(range(81, 105))
    assert (cases.source_at_utc < cases.reference_at_utc).all()


def test_comparison_mask_and_candidate_key_alignment(values):
    values.loc[values.index[81], "wind_speed_mps"] = 0
    values.loc[values.index[57], "temperature_c"] = np.nan
    cases = baseline_cases(values, partitions(values))
    first = cases.sort_index().xs(("validation", values.index[80], values.index[81], 1))
    assert not first.loc["wind_direction_deg", "eligible"]
    assert not first.loc["temperature_c", "eligible"]
    assert not cases.xs("rain_probability", level="metric").eligible.any()
    score = score_predictions(cases, cases.seasonal_naive)
    candidate = cases.loc[cases.eligible, "actual"].sample(frac=1, random_state=2)
    perfect = score_predictions(cases, candidate)
    pd.testing.assert_series_equal(score.scored, perfect.scored)
    assert perfect.rmse.dropna().eq(0).all()
    with pytest.raises(CollectionError, match="Missing predictions"):
        score_predictions(cases, candidate.iloc[1:])


def test_scores_on_known_arrays():
    result = metrics("temperature_c", [1, 2, 3], [2, 2, 2])
    assert result == pytest.approx(dict(rmse=np.sqrt(2 / 3), mae=2 / 3, r2=0))
    assert np.isnan(metrics("temperature_c", [3, 3], [1, 2])["r2"])
    assert np.isnan(metrics("precipitation_mm", [0.1] * 3, [1] * 3)["r2"])
    assert metrics("wind_direction_deg", [359, 2], [1, 358]) == {"circular_mae_deg": 3}
    assert metrics("rain_probability", [0, 1], [0.5, 0.75]) == {"brier": 0.15625}
    assert metrics("uv_index", [], []) == {}
    for actual, predicted in (([0.5], [0]), ([1], [1.1]), ([1], [np.nan])):
        with pytest.raises(CollectionError):
            metrics("rain_probability", actual, predicted)


def test_evaluation_publishes_auditable_short_sample(tmp_path):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=response_payload(request))
        )
    ) as client:
        dataset = backfill(CITY, tmp_path / "source", client, date(2025, 1, 1), date(2025, 1, 2))
    output = tmp_path / "evaluation"
    manifest = evaluate(dataset, output, CITY)
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert manifest["status"] == "complete" and not manifest["operational_point_in_time_safe"]
    assert all(digest(output / name) == checksum for name, checksum in manifest["sha256"].items())
    summary = pd.read_csv(output / "metrics.csv")
    assert len(summary) == 2 * 24 * len(FORECAST_METRICS)
    assert summary.scored.sum() == manifest["scored_metric_cases"] > 0
    assert summary[summary.horizon_hours == 24].scored.eq(0).all()
    assert summary[summary.metric == "rain_probability"].brier.isna().all()
    assert evaluate(dataset, output, CITY)["sha256"] == manifest["sha256"]
    with pytest.raises(CollectionError, match="boundary"):
        evaluate(dataset, output, CITY, validation_start="2025-01-01 12:00")
    with pytest.raises(CollectionError, match="overwrite"):
        evaluate(dataset, dataset, CITY)
