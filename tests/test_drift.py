from datetime import timedelta
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from test_collect import CITY

from mosamiat.agents.drift import error_checks, input_checks
from mosamiat.agents.ledger import Ledger
from mosamiat.agents.retraining import MAX_TRAINING_ATTEMPTS, checkpoint, fit_candidate, run_drift
from mosamiat.modeling.inference import MANIFEST


def test_input_drift_and_incomplete_window():
    hours = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
    values = pd.DataFrame({"temperature_c": 10.0, "wind_direction_deg": 359.0}, index=hours)
    limits = {"temperature_c": {"lower": 10, "upper": 20}}
    assert input_checks(values, limits)[0]["status"] == "within_limits"
    values.temperature_c += 11
    assert input_checks(values, limits)[0]["status"] == "drift"
    values.iloc[-1, 0] = np.nan
    assert input_checks(values, limits)[0]["status"] == "insufficient"


def test_model_drift_uses_matched_mature_targets_and_correct_scales():
    hours = pd.date_range("2026-01-01", periods=25, freq="h", tz="UTC")
    truth = pd.DataFrame(
        {"temperature_c": 0.0, "wind_direction_deg": 359.0, "rain_probability": 1.0}, index=hours
    )
    limits = {
        key: dict(model="seasonal_naive", measure=metric, upper=upper)
        for key, metric, upper in [
            ("temperature_c", "rmse", 2),
            ("wind_direction_deg", "circular_mae_deg", 3),
            ("rain_probability", "brier", 0.3),
        ]
    }
    runs = [
        dict(
            issued_at_utc=(t - timedelta(hours=24)).isoformat(),
            evaluation_sha256=MANIFEST,
            forecasts=[
                dict(
                    horizon_hours=24,
                    target_at_utc=t.isoformat(),
                    metric=k,
                    model_name="seasonal_naive",
                    model_version=1,
                    value=v,
                )
                for k, v in zip(limits, [4, 1, 0.5], strict=True)
            ],
        )
        for t in hours
    ]
    checks = error_checks(runs, truth, hours[0], hours[-1], limits)
    assert [c["status"] for c in checks] == ["drift", "within_limits", "within_limits"]
    assert [c["score"] for c in checks] == [4.0, 2.0, 0.25]
    assert all(c["last"] == hours[-2].isoformat() and c["count"] == 24 for c in checks)
    runs[0]["forecasts"][0]["model_version"] = 2
    assert error_checks(runs, truth, hours[0], hours[-1], limits)[0]["status"] == "insufficient"


def test_checkpoint_and_worker_reuse_one_candidate(tmp_path, monkeypatch):
    ledger = Ledger(tmp_path / "ledger.sqlite3")

    def decide(hour):
        run = dict(run_id=hour, issued_at_utc=hour, city_id="islamabad")
        return checkpoint(ledger, run, lambda: dict(run_id=hour, drift_count=1))

    first = decide("2026-01-15T10:00:00+00:00")
    same = decide("2026-01-15T11:00:00+00:00")
    assert same["action"] == "cooldown" and same["request_id"] == first["request_id"]
    assert decide("2026-01-15T10:00:00+00:00") == first
    assert decide("2026-01-16T10:00:00+00:00")["request_id"] != first["request_id"]

    def train(store, city, stamp, work, output):
        assert stamp == pd.Timestamp("2026-01-15T10:00:00Z")
        output.mkdir()
        return "sha"

    trainer = Mock(side_effect=train)
    monkeypatch.setattr("mosamiat.agents.retraining.train", trainer)
    monkeypatch.setattr("mosamiat.agents.retraining.verify_training", Mock(return_value="sha"))
    store = Mock()
    assert fit_candidate(CITY, first["request_id"], tmp_path, store) == fit_candidate(
        CITY, same["request_id"], tmp_path, store
    )
    trainer.assert_called_once()


def test_retraining_failure_is_reported_without_losing_the_drift_decision(
    tmp_path, monkeypatch, capsys
):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    payload = {"run_id": "hour", "city_id": CITY.city_id, "issued_at_utc": "2026-11-27T08:00:00Z"}
    ledger.log("hour", "forecast_completed", "Readback verified")
    assess = Mock(return_value={"run_id": "hour", "drift_count": 1})
    publish = Mock()
    monkeypatch.setattr("mosamiat.agents.retraining.evaluate", assess)
    monkeypatch.setattr("mosamiat.agents.retraining.publish_drift", publish)
    result = run_drift(CITY, payload, ledger, Mock(), trainer=Mock(side_effect=RuntimeError))
    assert result["training_status"] == "failed"
    assert result["drift_count"] == 1
    assert publish.call_args.args[1] == result
    assert [event["status"] for event in ledger.history(ledger.path, "hour")] == [
        "forecast_completed",
        "drift_assessed",
        "retraining_failed",
        "drift_verified",
    ]
    assert "RuntimeError" in capsys.readouterr().err


@pytest.mark.parametrize("succeeds", [False, True])
def test_drift_reuses_success_or_caps_attempts_after_worker_restarts(tmp_path, monkeypatch, succeeds):
    monkeypatch.setattr(
        "mosamiat.agents.retraining.evaluate",
        lambda city, payload, *args: dict(run_id=payload["run_id"], drift_count=1),
    )
    monkeypatch.setattr("mosamiat.agents.retraining.publish_drift", Mock())
    trainer = Mock(return_value="verified-candidate", side_effect=None if succeeds else RuntimeError)
    path = tmp_path / "ledger.sqlite3"
    results = []
    for hour in range(10, 10 + MAX_TRAINING_ATTEMPTS + 2):
        payload = dict(
            run_id=f"hour-{hour}",
            city_id=CITY.city_id,
            issued_at_utc=f"2026-11-27T{hour}:00:00Z",
        )
        results.append(run_drift(CITY, payload, Ledger(path), Mock(), trainer=trainer))
    assert len({result["request_id"] for result in results}) == 1
    if succeeds:
        trainer.assert_called_once()
        assert all(result["candidate_sha256"] == "verified-candidate" for result in results)
    else:
        assert trainer.call_count == MAX_TRAINING_ATTEMPTS
        assert [result["training_status"] for result in results] == (
            ["failed"] * MAX_TRAINING_ATTEMPTS + ["attempt_limit_reached"] * 2
        )
