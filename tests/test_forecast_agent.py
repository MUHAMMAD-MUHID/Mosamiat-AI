import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import numpy as np
import pandas as pd
import pytest
import requests
from test_collect import CITY, response_payload

from mosamiat.agents.ledger import Ledger, MaterializationError, _write_verified
from mosamiat.collect import CollectionError, collect
from mosamiat.features.pipeline import load_definitions
from mosamiat.modeling.baseline import FORECAST_METRICS
from mosamiat.modeling.inference import MODEL, recent_history


@pytest.mark.parametrize("use_active", [False, True])
@pytest.mark.parametrize("bulletin_fails", [False, True])
def test_live_graph_replays_prepared_forecast_after_publication_failure(
    tmp_path, monkeypatch, capsys, use_active, bulletin_fails
):
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
    if use_active:
        candidate = {"name": "mosamiat_islamabad_lstm", "version": 42, "family": "lstm"}
        policy = {"models": {"temperature_c:1": candidate}, "error_limits": {}}
        monkeypatch.setattr(graph, "read_policy", MagicMock(return_value=(policy, "a" * 64)))
        monkeypatch.setattr(graph, "load_candidate", MagicMock(return_value=object()))
        monkeypatch.setattr(
            graph,
            "candidate_predictions",
            MagicMock(return_value=forest.predict.return_value.xs(reference, level=0)),
        )
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
    checkpoint = MagicMock()
    app = graph.build_graph(
        CITY,
        ledger,
        project,
        lambda: now,
        publish_inputs=publish_inputs,
        checkpoint_forecast=lambda: checkpoint(
            [item["status"] for item in ledger.history(ledger.path, "hourly")]
        ),
    )
    args = dict(run_id="hourly", reference_at_utc=reference.isoformat(), snapshot=str(path))
    with pytest.raises(MaterializationError, match="operation=readback.*error=ConnectionError"):
        app.invoke(args)
    checkpoint.assert_not_called()
    assert "forecast_completed" not in [
        event["status"] for event in ledger.history(ledger.path, "hourly")
    ]
    assert read.call_count == 3
    failures = [
        e["detail"] for e in ledger.history(ledger.path, "hourly") if e["status"] == "node_failed"
    ]
    assert len(failures) == 3
    assert all(item.startswith("save_forecasts: operation=readback") for item in failures)
    saved = group.insert.call_args.args[0]
    read.side_effect = [httpx.ReadTimeout("interrupted readback"), saved.copy()]
    bulletins = MagicMock()
    bulletins.insert.return_value = (None, None)
    bulletins.primary_key = ["run_id"]
    bulletins.event_time = "issued_at_utc"
    bulletin_read = bulletins.select_all.return_value.filter.return_value.read
    bulletin_read.side_effect = lambda **_: (
        bulletins.insert.call_args.args[0].copy() if bulletins.insert.called else pd.DataFrame()
    )
    if bulletin_fails:
        bulletins.id = 43
        job = bulletins.materialization_job
        failed = SimpleNamespace(id=309209, success=False)
        job.run.return_value = failed
        job.get_executions.side_effect = [[], [failed]]
        bulletins.insert.side_effect = lambda *args, **kwargs: (job.run(), None)
    drift_group = MagicMock()
    drift_group.insert.return_value = (None, None)
    drift_group.primary_key = ["run_id"]
    drift_group.event_time = "issued_at_utc"
    drift_group.select_all.return_value.filter.return_value.read.side_effect = lambda **_: (
        drift_group.insert.call_args.args[0].copy()
    )
    monkeypatch.setattr(
        "mosamiat.agents.retraining.fit_candidate",
        MagicMock(side_effect=RuntimeError) if use_active else MagicMock(return_value="test-model"),
    )
    if use_active:
        monkeypatch.setattr(
            "mosamiat.agents.retraining.evaluate",
            lambda city, payload, *args: dict(
                run_id=payload["run_id"],
                issued_at_utc=payload["issued_at_utc"],
                checks=[],
                drift_count=1,
            ),
        )
    groups = {
        "mosamiat_forecast_ledger": group,
        "mosamiat_weather_bulletins": bulletins,
        "mosamiat_drift_log": drift_group,
    }
    project.return_value.get_feature_store.return_value.get_or_create_feature_group.side_effect = (
        lambda **kw: groups[kw["name"]]
    )
    state = app.invoke(dict(run_id="hourly", retry=True))
    assert state["status"] == "verified"
    assert state["bulletin_status"] == ("failed" if bulletin_fails else "verified")
    checkpoint.assert_called_once()
    assert checkpoint.call_args.args[0][-1] == "forecast_completed"
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
    assert "daily" in bulletins.insert.call_args.args[0].iloc[0].report_trigger.split("+")
    assert ("bulletin_failed" in [e["status"] for e in events]) is bulletin_fails
    err = capsys.readouterr().err
    if bulletin_fails:
        assert not state["bulletin"]
        assert "::warning::Weather bulletin unavailable" in err
        assert "feature_group=mosamiat_weather_bulletins execution_id=309209" in err
    assert drift_group.insert.call_args.args[0].iloc[0].run_id == "hourly"
    if use_active:
        assert state["drift_result"]["training_status"] == "failed"
        assert drift_group.insert.call_args.args[0].iloc[0].training_status == "failed"
    loader.assert_called_once()
    assert publish_inputs.call_args.args[1]["input_features"] == (
        forest.predict.call_args.args[0][manifest["features"]].iloc[0].to_dict()
    )
    assert "input_features" not in saved
    assert saved.forecast_id.nunique() == len(saved) == 192
    assert saved.target_at_utc.nunique() == 24
    assert saved.loc[saved.metric.eq("aqi_value"), "model_name"].eq(MODEL).all()
    temperatures = saved.loc[saved.metric.eq("temperature_c")]
    assert (
        temperatures.loc[temperatures.horizon_hours.gt(1), "model_name"].eq("seasonal_naive").all()
    )
    assert temperatures.loc[temperatures.horizon_hours.eq(1), "model_name"].item() == (
        "mosamiat_islamabad_lstm" if use_active else "seasonal_naive"
    )
    if use_active:
        assert saved.evaluation_sha256.eq("a" * 64).all()
    assert saved.loc[saved.metric.eq("rain_probability"), "value"].eq(0.1).all()
    pd.testing.assert_frame_equal(saved, group.insert.call_args.args[0])
    read.side_effect = None
    read.return_value = saved.iloc[:-1].copy()
    with pytest.raises(CollectionError, match="readback"):
        app.invoke(dict(run_id="hourly", retry=True))
    assert read.call_count == 6
    group.insert.assert_called_once()


def test_forecast_write_reconciles_partial_and_complete_remote_rows():
    group = MagicMock(primary_key=["forecast_id"], event_time="target_at_utc")
    store = MagicMock()
    store.get_or_create_feature_group.return_value = group
    frame = pd.DataFrame(
        {
            "forecast_id": ["a", "b"],
            "run_id": ["same", "same"],
            "target_at_utc": pd.date_range("2026-11-27", periods=2, freq="h", tz="UTC"),
            "value": [1.5, 2.5],
        }
    )
    read = group.select_all.return_value.filter.return_value.read
    group.insert.return_value = (None, None)
    read.side_effect = [frame.iloc[:1].copy(), frame.copy(), frame.copy()]
    args = (store, frame, "forecast", "forecast_id", "target_at_utc", ["target_at_utc"], "forecast")
    _write_verified(*args, reconcile=True)
    _write_verified(*args, reconcile=True)
    group.insert.assert_called_once()
    assert group.insert.call_args.args[0].forecast_id.tolist() == ["b"]
    assert group.insert.call_args.kwargs["operation"] == "upsert"
    read.side_effect = [frame.assign(value=[5.5, 2.5])]
    with pytest.raises(CollectionError, match="readback differs"):
        _write_verified(*args, reconcile=True)
    group.insert.assert_called_once()


@pytest.fixture
def materialization():
    group = MagicMock(id=7, primary_key=["forecast_id"], event_time="target_at_utc")
    store = MagicMock()
    store.get_or_create_feature_group.return_value = group
    frame = pd.DataFrame({
        "forecast_id": ["a", "b"],
        "run_id": ["hour", "hour"],
        "target_at_utc": pd.date_range("2026-11-27", periods=2, freq="h", tz="UTC"),
        "value": [1.5, 2.5],
    })
    job = group.materialization_job
    group.insert.side_effect = lambda *args, **kwargs: (job.run(args="sdk offsets"), None)
    read = group.select_all.return_value.filter.return_value.read
    args = (store, frame, "forecast", "forecast_id", "target_at_utc", ["target_at_utc"], "forecast")
    return args, frame, group, job, read


@pytest.mark.parametrize("remote_state", ["complete", "partial", "missing", "conflicting"])
def test_ambiguous_submission_reconciles_and_never_reissues_without_id(
    tmp_path, materialization, remote_state
):
    args, frame, group, job, read = materialization
    job.get_executions.return_value = []
    response = requests.Response()
    response.status_code = 503
    response.url = "https://service.test/?key=api-secret"
    job.run.side_effect = requests.HTTPError("authorization=secret-value", response=response)
    actual = {
        "complete": frame, "partial": frame.iloc[:1], "missing": frame.iloc[:0],
        "conflicting": frame.assign(value=[8.0, 2.5]),
    }[remote_state]
    read.return_value = actual.copy()
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    if remote_state == "complete":
        pd.testing.assert_frame_equal(_write_verified(*args, ledger=ledger), frame)
    else:
        with pytest.raises(MaterializationError, match="http=503 category=5xx") as failed:
            _write_verified(*args, ledger=ledger)
        assert "operation=" in str(failed.value)
        assert "execution_id=unknown" in str(failed.value)
        assert "api-secret" not in str(failed.value) and "secret-value" not in str(failed.value)
        assert failed.value.__suppress_context__
        reopened = Ledger(ledger.path)
        with pytest.raises(MaterializationError, match="outcome unknown|readback differs"):
            _write_verified(*args, ledger=reopened, reconcile=True)
        read.return_value = frame.copy()
        pd.testing.assert_frame_equal(_write_verified(*args, ledger=reopened), frame)
    group.insert.assert_called_once()
    job.run.assert_called_once_with(args="sdk offsets")
    job.get_executions.assert_called_once()
    assert ledger.materialization("hour", "forecast") == (None,)


def test_replay_reads_after_known_terminal_execution_and_only_writes_missing_rows(
    tmp_path, materialization
):
    args, frame, group, job, read = materialization
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    ledger.record_materialization("hour", "forecast", 12)
    failed = SimpleNamespace(id=12, success=False)
    sent = SimpleNamespace(id=13, success=True)
    unrelated = SimpleNamespace(id=14, success=False)
    job.run.return_value = sent
    job.get_executions.side_effect = [
        [unrelated, failed], [unrelated, failed], [unrelated, sent, failed],
    ]
    read.side_effect = [frame.iloc[:0], frame.iloc[:1], frame.copy(), frame.copy()]
    pd.testing.assert_frame_equal(_write_verified(*args, ledger=ledger), frame)
    assert group.insert.call_args.args[0].forecast_id.tolist() == ["b"]
    assert ledger.materialization("hour", "forecast") == (13,)
    pd.testing.assert_frame_equal(_write_verified(*args, ledger=ledger), frame)
    group.insert.assert_called_once()


@pytest.mark.parametrize("success", [True, False])
def test_tracks_only_submitted_execution_and_requires_full_readback(
    monkeypatch, tmp_path, materialization, success
):
    args, frame, group, job, read = materialization
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    monkeypatch.setattr("mosamiat.agents.ledger.time.sleep", lambda _: None)
    execution = SimpleNamespace(id=12, success=success)
    unrelated = SimpleNamespace(id=20, success=not success)
    job.run.return_value = execution
    job.get_executions.side_effect = [[], [unrelated], [unrelated, execution]]
    read.return_value = frame.iloc[:1]
    reason = "readback is incomplete" if success else "materialization failed"
    with pytest.raises(MaterializationError, match=f"execution_id=12.*{reason}"):
        _write_verified(*args, ledger=ledger)
    assert read.call_count == int(success)
    group.insert.assert_called_once()
    assert ledger.materialization("hour", "forecast") == (12,)


def test_unidentified_bootstrap_uses_only_row_verification(monkeypatch, materialization):
    args, frame, group, job, read = materialization
    group.id = None
    group.insert.side_effect = None
    group.insert.return_value = (job, None)
    read.side_effect = [frame.iloc[:0], frame.copy()]
    monkeypatch.setattr("mosamiat.agents.ledger.time.sleep", lambda _: None)
    pd.testing.assert_frame_equal(_write_verified(*args), frame)
    job.get_executions.assert_not_called()


def test_launch_tracking_ignores_other_calls_on_the_same_job(materialization):
    from mosamiat.hopsworks_compat import insert_with_execution

    _, frame, group, job, _ = materialization
    launch = job.run
    launch.side_effect = [14, 12]

    def insert(*args, **kwargs):
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(job.run).result() == 14
        assert job.run() == 12
        return job, None

    group.insert.side_effect = insert
    observe = MagicMock(side_effect=lambda run: run())
    assert insert_with_execution(group, frame, job, observe) == (job, None)
    observe.assert_called_once_with(launch)
    assert job.run is launch


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
