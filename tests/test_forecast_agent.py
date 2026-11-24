import json
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import numpy as np
import pandas as pd
import pytest
from test_collect import CITY, response_payload

from mosamiat.collect import CollectionError, collect
from mosamiat.features.pipeline import load_definitions
from mosamiat.modeling.baseline import FORECAST_METRICS
from mosamiat.modeling.inference import MODEL, recent_history


def test_live_graph_replays_prepared_forecast_after_publication_failure(tmp_path, monkeypatch):
    graph = pytest.importorskip("mosamiat.agents.graph")
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=response_payload(r)))
    ) as client:
        path = collect(CITY, tmp_path, client)
    now = pd.Timestamp.now(tz="UTC")
    reference = now.floor("h")
    history, _ = recent_history(path, CITY, now)
    assert history.index[-1] == reference - pd.Timedelta(hours=1)
    with pytest.raises(CollectionError, match="recent"):
        recent_history(path, CITY, now + pd.Timedelta(hours=1))

    scores = pd.DataFrame(
        [
            dict(model=m, split=s, horizon_hours=h, metric=k, rmse=e, circular_mae_deg=e)
            for h in range(1, 25)
            for k in FORECAST_METRICS
            for s in ("test", "validation")
            for m, e in (
                ("seasonal_naive", 1),
                ("random_forest", 0 if s == "test" or k == "aqi_value" else 1),
            )
        ]
    )
    forest = MagicMock()
    index = pd.MultiIndex.from_product(
        [[reference], range(1, 25)], names=["reference_at_utc", "horizon_hours"]
    )
    forest.predict.return_value = pd.DataFrame(
        {k: np.ones(24) for k in FORECAST_METRICS}, index=index
    ).assign(target_at_utc=pd.date_range(reference + pd.Timedelta(hours=1), periods=24, freq="h"))
    manifest = dict(
        city_id=CITY.city_id,
        timezone=CITY.timezone,
        rain_training_event_rate=0.1,
        feature_definitions=load_definitions(Path("configs/features.toml")),
        features=["temperature_c_lag_1h", "hour"],
    )
    loader = MagicMock(return_value=(forest, manifest, scores))
    monkeypatch.setattr(graph, "load_forecast_bundle", loader)
    project = MagicMock()
    group = (
        project.return_value.get_feature_store.return_value.get_or_create_feature_group.return_value
    )
    group.primary_key = ["forecast_id"]
    group.insert.return_value = (None, None)
    group.event_time = "target_at_utc"
    read = group.select_all.return_value.filter.return_value.read
    read.side_effect = ConnectionError("Readback interrupted")
    ledger = graph.Ledger(tmp_path / "ledger.sqlite3")
    publish_inputs = MagicMock()
    app = graph.build_graph(CITY, ledger, project, lambda: now, publish_inputs=publish_inputs)
    args = dict(run_id="hourly", reference_at_utc=reference.isoformat(), snapshot=str(path))
    with pytest.raises(ConnectionError):
        app.invoke(args)
    assert read.call_count == 3
    failures = [
        e["detail"] for e in ledger.history(ledger.path, "hourly") if e["status"] == "node_failed"
    ]
    assert failures == ["save_forecasts: ConnectionError"] * 3
    saved = group.insert.call_args.args[0]
    read.side_effect = [httpx.ReadTimeout("interrupted readback"), saved.copy()]
    bulletins = MagicMock()
    bulletins.insert.return_value = (None, None)
    bulletins.primary_key = ["run_id"]
    bulletins.event_time = "issued_at_utc"
    bulletins.select_all.return_value.filter.return_value.read.side_effect = lambda **_: (
        bulletins.insert.call_args.args[0].copy()
    )
    drift_group = MagicMock()
    drift_group.insert.return_value = (None, None)
    drift_group.primary_key = ["run_id"]
    drift_group.event_time = "issued_at_utc"
    drift_group.select_all.return_value.filter.return_value.read.side_effect = lambda **_: (
        drift_group.insert.call_args.args[0].copy()
    )
    monkeypatch.setattr(
        "mosamiat.agents.retraining.fit_candidate", MagicMock(return_value="test-model")
    )
    groups = {
        "mosamiat_forecast_ledger": group,
        "mosamiat_weather_bulletins": bulletins,
        "mosamiat_drift_log": drift_group,
    }
    project.return_value.get_feature_store.return_value.get_or_create_feature_group.side_effect = (
        lambda **kw: groups[kw["name"]]
    )
    assert app.invoke(dict(run_id="hourly", retry=True))["status"] == "verified"
    assert read.call_count == 5
    events = ledger.history(ledger.path, "hourly")
    assert [e["detail"] for e in events if e["status"] == "node_started"][-6:] == [
        "forecast",
        "save_forecasts",
        "save_forecasts",
        "alert",
        "weather_report",
        "drift",
    ]
    assert bulletins.insert.call_args.args[0].iloc[0].report_trigger == "daily"
    assert drift_group.insert.call_args.args[0].iloc[0].run_id == "hourly"
    loader.assert_called_once()
    assert publish_inputs.call_args.args[1]["input_features"] == (
        forest.predict.call_args.args[0][manifest["features"]].iloc[0].to_dict()
    )
    assert "input_features" not in saved
    assert saved.forecast_id.nunique() == len(saved) == 192
    assert saved.target_at_utc.nunique() == 24
    assert saved.loc[saved.metric.eq("aqi_value"), "model_name"].eq(MODEL).all()
    assert saved.loc[saved.metric.eq("temperature_c"), "model_name"].eq("seasonal_naive").all()
    assert saved.loc[saved.metric.eq("rain_probability"), "value"].eq(0.1).all()
    pd.testing.assert_frame_equal(saved, group.insert.call_args.args[0])
    read.side_effect = None
    read.return_value = saved.iloc[:-1].copy()
    with pytest.raises(CollectionError, match="readback"):
        app.invoke(dict(run_id="hourly", retry=True))
    assert read.call_count == 6


def test_history_command_reads_existing_ledger_without_running_graph(tmp_path, capsys):
    graph = pytest.importorskip("mosamiat.agents.graph")
    path = tmp_path / "ledger.sqlite3"
    command = ["--history", "example", "--ledger", str(path)]
    assert graph.main(command) == 1
    assert not path.exists()
    assert json.loads(capsys.readouterr().out) == []
    ledger = graph.Ledger(path)
    ledger.log("example", "verified", "192 remote forecast rows matched")
    ledger.log("other", "started", "hourly forecast")
    assert graph.main(command) == 0
    assert json.loads(capsys.readouterr().out) == ledger.history(path, "example")
