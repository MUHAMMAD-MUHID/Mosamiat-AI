from datetime import timedelta
from unittest.mock import Mock

import numpy as np
import pandas as pd
from test_collect import CITY

from mosamiat.agents.drift import error_checks, input_checks
from mosamiat.agents.ledger import Ledger
from mosamiat.agents.retraining import checkpoint, fit_candidate
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
