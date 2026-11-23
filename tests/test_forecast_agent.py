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
    )
    loader = MagicMock(return_value=(forest, manifest, scores))
    monkeypatch.setattr(graph, "load_forecast_bundle", loader)
    project = MagicMock()
    group = (
        project.return_value.get_feature_store.return_value.get_or_create_feature_group.return_value
    )
    group.primary_key = ["forecast_id"]
    group.event_time = "target_at_utc"
    read = group.select_all.return_value.filter.return_value.read
    read.side_effect = ConnectionError("Readback interrupted")
    ledger = graph.Ledger(tmp_path / "ledger.sqlite3")
    app = graph.build_graph(CITY, ledger, project, lambda: now)
    args = dict(run_id="hourly", reference_at_utc=reference.isoformat(), snapshot=str(path))
    with pytest.raises(ConnectionError):
        app.invoke(args)
    saved = group.insert.call_args.args[0]
    read.side_effect = None
    read.return_value = saved.copy()
    bulletins = MagicMock()
    bulletins.primary_key = ["run_id"]
    bulletins.event_time = "issued_at_utc"
    bulletins.select_all.return_value.filter.return_value.read.side_effect = lambda: (
        bulletins.insert.call_args.args[0].copy()
    )
    project.return_value.get_feature_store.return_value.get_or_create_feature_group.side_effect = (
        lambda **kw: group if kw["name"] == "mosamiat_forecast_ledger" else bulletins
    )
    assert app.invoke(dict(run_id="hourly", retry=True))["status"] == "verified"
    assert bulletins.insert.call_args.args[0].iloc[0].report_trigger == "daily"
    loader.assert_called_once()
    assert saved.forecast_id.nunique() == len(saved) == 192
    assert saved.target_at_utc.nunique() == 24
    assert saved.loc[saved.metric.eq("aqi_value"), "model_name"].eq(MODEL).all()
    assert saved.loc[saved.metric.eq("temperature_c"), "model_name"].eq("seasonal_naive").all()
    assert saved.loc[saved.metric.eq("rain_probability"), "value"].eq(0.1).all()
    pd.testing.assert_frame_equal(saved, group.insert.call_args.args[0])
    read.return_value = saved.iloc[:-1].copy()
    with pytest.raises(CollectionError, match="readback"):
        app.invoke(dict(run_id="hourly", retry=True))
