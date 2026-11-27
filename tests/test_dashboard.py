from hashlib import sha256
from zipfile import ZipFile

import httpx
import pandas as pd
import pytest
from test_collect import CITY
from test_weather_bulletin import payload

from mosamiat import dashboard
from mosamiat.agents.alerts import evaluate, forecast_hours
from mosamiat.agents.ledger import Ledger
from mosamiat.collect import CollectionError


def issue(ledger, hour, *, completed=True):
    data = payload(f"2026-11-24T{hour}:00Z") | {"city_id": CITY.city_id}
    ledger.prepare(data)
    ledger.bulletin(CITY, data, [])
    if completed:
        ledger.log(data["run_id"], "completed", "All four agents finished")
    return data


def test_outlook_uses_latest_completed_issue_and_dates_earlier_report(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    first = issue(ledger, "07")
    second = issue(ledger, "08")
    issue(ledger, "09", completed=False)
    result = dashboard.read_outlook(ledger.path, CITY)
    assert result.run_id == second["run_id"]
    assert result.report_issued == pd.Timestamp(first["issued_at_utc"])
    assert result.hours.shape == (24, 8)
    assert result.hours.index[0].hour == 14
    assert result.hours["rain_probability"].eq(10).all()


def test_outlook_uses_verified_forecast_before_bulletin_and_after_retraining_failure(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    first = issue(ledger, "07")
    data = payload("2026-11-24T08:00Z") | {"city_id": CITY.city_id}
    ledger.prepare(data)
    ledger.log(data["run_id"], "forecast_completed", "Remote forecast readback matched")
    ledger.log(data["run_id"], "retraining_failed", "RuntimeError")
    issue(ledger, "09", completed=False)
    result = dashboard.read_outlook(ledger.path, CITY)
    assert result.run_id == data["run_id"]
    assert result.report_issued == pd.Timestamp(first["issued_at_utc"])
    assert result.hours.shape == (24, 8)


def test_cloud_download_checksum_and_offline_fallback(tmp_path, monkeypatch):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    data = issue(ledger, "08")
    bundle = tmp_path / "fixture.zip"
    with ZipFile(bundle, "w") as zipped:
        zipped.write(ledger.path, "ledger.sqlite3")
    content = bundle.read_bytes()
    name = f"ledger-0000000001-{sha256(content).hexdigest()}.zip"
    cache = tmp_path / "cache" / CITY.city_id
    cache.mkdir(parents=True)

    def reply(request):
        assert request.method == "GET"
        if "getProjectInfo" in request.url.path:
            return httpx.Response(200, json={"projectId": 42})
        if request.url.params.get("action") == "listing":
            return httpx.Response(200, json={"items": [{"attributes": {"name": name}}]})
        return httpx.Response(200, content=content)

    with httpx.Client(
        base_url="https://example.test/", transport=httpx.MockTransport(reply)
    ) as client:
        result = dashboard._fetch_checkpoint(client, "city", CITY, cache)
    assert result.run_id == data["run_id"]

    def offline(*args):
        raise ConnectionError

    monkeypatch.setattr(dashboard, "_cloud_outlook", offline)
    result, source = dashboard.load_outlook(CITY, cache.parent, tmp_path / "absent")
    assert source == "Cached Hopsworks checkpoint" and result.run_id == data["run_id"]
    (cache / name).write_bytes(b"damaged")
    with pytest.raises(CollectionError, match="No verified forecast"):
        dashboard.load_outlook(CITY, cache.parent, tmp_path / "absent")


def test_desk_unit_switching_window_and_hour_comparison(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    issue(ledger, "07")
    outlook = dashboard.read_outlook(ledger.path, CITY)
    outlook.alerts = evaluate(forecast_hours(payload("2025-01-01T05:00Z", aqi_value=152)))
    outlook.hours.iloc[12, outlook.hours.columns.get_loc("temperature_c")] = 30
    monkeypatch.setattr(dashboard, "load_outlook", lambda city: (outlook, "Local verified ledger"))
    app = testing.AppTest.from_file("streamlit_app.py", default_timeout=15).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Overview", "Hourly explorer", "Alerts", "Bulletin"]
    app.radio[0].set_value("°F")
    app.radio[1].set_value("km/h")
    app.get("button_group")[0].set_value(6)
    app.run()
    assert not app.exception
    table, compare = (frame.value for frame in app.dataframe)
    assert len(table) == 6 and table["Temperature (°F)"].eq(68).all()
    assert table["Wind speed (km/h)"].eq(10.8).all()
    assert table["Rain chance (%)"].eq(10).all()
    assert compare.iloc[0].tolist() == ["Temperature", "68.0 °F", "86.0 °F"]
    assert len(compare) == 8 and outlook.hours.temperature_c.iloc[0] == 20

    selectors = {box.label: box for box in app.selectbox}
    selectors["Alert status"].set_value("Ended")
    selectors["Severity"].set_value("Elevated")
    selectors["Guidance for"].set_value("Outdoor workers")
    app.run()
    assert not app.exception
    assert any(item.value == outlook.alerts[0]["workers"] for item in app.markdown)
    next(box for box in app.selectbox if box.label == "Severity").set_value("Advisory").run()
    assert any(item.value == "No alerts match these filters." for item in app.info)
