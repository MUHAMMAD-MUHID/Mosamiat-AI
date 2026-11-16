import json
from datetime import UTC, date, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path

import httpx
import pandas as pd
import pytest

from mosamiat.collect import (
    PROBABILITY,
    WEATHER,
    City,
    CollectionError,
    collect,
    fetch_json,
    main,
    normalize,
    validate_window,
)

CITY = City("islamabad", "Islamabad", 33.6844, 73.0479, "Asia/Karachi")
VALUES = {
    "temperature_2m": ("°C", 24.5),
    "relative_humidity_2m": ("%", 62),
    "precipitation": ("mm", 0.2),
    "precipitation_probability": ("%", 35),
    "wind_speed_10m": ("m/s", 3.5),
    "wind_direction_10m": ("°", 359),
    "us_aqi": ("USAQI", 91),
    "uv_index": ("", 2.1),
    "pm2_5": ("μg/m³", 31.2),
    "pm10": ("μg/m³", 65),
    "carbon_monoxide": ("μg/m³", 300),
    "nitrogen_dioxide": ("μg/m³", 20),
    "sulphur_dioxide": ("μg/m³", 5),
    "ozone": ("μg/m³", 45),
}


def payload(fields: list[str], times: list[int]) -> dict:
    return {
        "latitude": 33.7,
        "longitude": 73.0,
        "utc_offset_seconds": 0,
        "hourly": {"time": times} | {field: [VALUES[field][1]] * len(times) for field in fields},
        "hourly_units": {"time": "unixtime"} | {field: VALUES[field][0] for field in fields},
    }


def response_payload(request: httpx.Request) -> dict:
    params = request.url.params
    assert params["timezone"] == "GMT"
    assert params["timeformat"] == "unixtime"
    assert float(params["latitude"]) == CITY.latitude
    if "start_date" in params:
        first = datetime.fromisoformat(params["start_date"]).replace(tzinfo=UTC)
        days = (date.fromisoformat(params["end_date"]) - first.date()).days + 1
        hours = days * 24
    else:
        first = datetime.fromisoformat(params["start_hour"]).replace(tzinfo=UTC)
        last = datetime.fromisoformat(params["end_hour"]).replace(tzinfo=UTC)
        hours = int((last - first).total_seconds() / 3600) + 1
    times = [int(first.timestamp()) + hour * 3600 for hour in range(hours)]
    if request.url.host == "air-quality-api.open-meteo.com":
        assert params["domains"] == "cams_global"
    else:
        assert params["wind_speed_unit"] == "ms"
    return payload(params["hourly"].split(","), times)


def test_recent_snapshot_preserves_units_provenance_and_probability(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=response_payload(request))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        run = collect(CITY, tmp_path, client)
    frame = pd.read_csv(run / "hourly.csv")
    manifest = json.loads((run / "manifest.json").read_text())
    assert len(requests) == 2
    assert len(frame) == 48 * 14
    assert manifest["status"] == "complete"
    assert manifest["is_ground_truth"] is False
    assert set(frame["data_kind"]) == {"provider_forecast"}
    assert frame.loc[frame.metric == "rain_probability", "value"].eq(0.35).all()
    assert frame.loc[frame.metric == "aqi_value", "aqi_scale"].eq("US").all()
    assert frame.loc[frame.metric == "wind_speed_mps", "value"].eq(3.5).all()
    assert pd.to_datetime(frame.valid_at_utc, utc=True).dt.tz is UTC
    assert not frame.duplicated(["city_id", "source", "product", "valid_at_utc", "metric"]).any()
    raw = json.loads((run / "air_quality.json").read_text(encoding="utf-8"))
    assert raw["response"]["hourly"]["us_aqi"] == [91] * 48
    assert raw["parameters"]["latitude"] == CITY.latitude
    assert set(frame["grid_latitude"]) == {33.7}


def test_historical_snapshot_keeps_products_distinct_and_marks_unavailable_probability(tmp_path):
    def handler(request):
        if request.url.host == "archive-api.open-meteo.com":
            assert request.url.params["models"] == "era5"
            assert "precipitation_probability" not in request.url.params["hourly"]
        return httpx.Response(200, json=response_payload(request))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        run = collect(CITY, tmp_path, client, date(2025, 1, 1), date(2025, 1, 2))
    frame = pd.read_csv(run / "hourly.csv")
    manifest = json.loads((run / "manifest.json").read_text())
    assert len(frame) == 48 * 13
    assert set(frame["data_kind"]) == {"reanalysis", "archived_provider_forecast"}
    assert "rain_probability" not in set(frame.metric)
    assert manifest["unavailable_metrics"] == ["rain_probability"]
    assert manifest["status"] == "partial"
    assert manifest["valid_from_utc"] == "2025-01-01T00:00:00+00:00"
    assert manifest["valid_until_utc"] == "2025-01-02T23:00:00+00:00"


def test_nulls_are_retained_and_reported_without_imputation(tmp_path):
    def handler(request):
        data = response_payload(request)
        if "temperature_2m" in data["hourly"]:
            data["hourly"]["temperature_2m"][0] = None
        return httpx.Response(200, json=data)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        run = collect(CITY, tmp_path, client)
    manifest = json.loads((run / "manifest.json").read_text())
    frame = pd.read_csv(run / "hourly.csv")
    assert manifest["status"] == "partial"
    assert manifest["missing_values"]["temperature_c"] == 1
    missing = frame.loc[frame.quality_status == "missing"]
    assert len(missing) == 1
    assert missing.value.isna().all()


@pytest.mark.parametrize(
    "failure", ["unavailable", "misaligned", "truncated", "bad_unit", "nonfinite"]
)
def test_failed_sources_keep_raw_evidence_but_publish_no_csv(tmp_path, failure):
    def handler(request):
        data = response_payload(request)
        if request.url.host == "air-quality-api.open-meteo.com":
            if failure == "unavailable":
                return httpx.Response(400, json={"reason": "invalid request"})
            if failure == "misaligned":
                data["hourly"]["time"] = [stamp + 3600 for stamp in data["hourly"]["time"]]
            if failure == "truncated":
                data["hourly"] = {field: values[:-1] for field, values in data["hourly"].items()}
            if failure == "bad_unit":
                data["hourly_units"]["us_aqi"] = "European AQI"
            if failure == "nonfinite":
                data["hourly"]["us_aqi"][0] = float("nan")
                return httpx.Response(200, content=json.dumps(data))
        return httpx.Response(200, json=data)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(CollectionError):
            collect(CITY, tmp_path, client)
    (run,) = tmp_path.iterdir()
    assert (run / "weather.json").exists()
    assert not (run / "hourly.csv").exists()
    assert json.loads((run / "manifest.json").read_text())["status"] == "failed"


@pytest.mark.parametrize(
    "mutation",
    [
        "empty",
        "duplicate",
        "gap",
        "off_hour",
        "offset",
        "missing_field",
        "unequal",
        "unit",
        "infinity",
        "boolean",
        "humidity",
        "negative_rain",
        "grid",
    ],
)
def test_normalize_rejects_malformed_or_invalid_data(mutation):
    data = payload(list(WEATHER), [1735689600, 1735693200])
    hourly = data["hourly"]
    if mutation == "empty":
        hourly["time"] = []
    elif mutation == "duplicate":
        hourly["time"][1] = hourly["time"][0]
    elif mutation == "gap":
        hourly["time"][1] += 3600
    elif mutation == "off_hour":
        hourly["time"][0] += 1
    elif mutation == "offset":
        data["utc_offset_seconds"] = 18000
    elif mutation == "missing_field":
        del hourly["precipitation"]
    elif mutation == "unequal":
        hourly["precipitation"].pop()
    elif mutation == "unit":
        data["hourly_units"]["wind_speed_10m"] = "km/h"
    elif mutation == "infinity":
        hourly["temperature_2m"][0] = float("inf")
    elif mutation == "boolean":
        hourly["temperature_2m"][0] = True
    elif mutation == "humidity":
        hourly["relative_humidity_2m"][0] = 101
    elif mutation == "negative_rain":
        hourly["precipitation"][0] = -0.1
    elif mutation == "grid":
        data["latitude"] = 91
    with pytest.raises(CollectionError):
        normalize(data, WEATHER, CITY, "weather_forecast", "provider_forecast", "2025-01-01T00:00Z")


@pytest.mark.parametrize("value, expected", [(0, 0), (100, 1), (None, None)])
def test_probability_boundaries_and_missing_value(value, expected):
    data = payload(["precipitation_probability"], [1735689600])
    data["hourly"]["precipitation_probability"] = [value]
    (row,) = normalize(data, PROBABILITY, CITY, "weather_forecast", "provider_forecast", "now")
    assert row["value"] == expected


@pytest.mark.parametrize("transient", [429, 503, "timeout"])
def test_transient_failures_retry_then_succeed(monkeypatch, transient):
    calls = []
    delays = []
    monkeypatch.setattr("mosamiat.collect.time.sleep", delays.append)

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            if transient == "timeout":
                raise httpx.ReadTimeout("temporary", request=request)
            return httpx.Response(transient, headers={"Retry-After": "2"})
        return httpx.Response(200, json={"ok": True})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert fetch_json(client, "https://example.test", {}) == {"ok": True}
    assert len(calls) == 3
    assert len(delays) == 2
    if transient == 429:
        assert delays == [2, 2]


@pytest.mark.parametrize("status, expected_calls", [(400, 1), (401, 1), (429, 3), (503, 3)])
def test_permanent_or_exhausted_failures_stop(monkeypatch, status, expected_calls):
    calls = []
    monkeypatch.setattr("mosamiat.collect.time.sleep", lambda delay: None)

    def handler(request):
        calls.append(request)
        return httpx.Response(status)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(CollectionError):
            fetch_json(client, "https://example.test", {})
    assert len(calls) == expected_calls


@pytest.mark.parametrize("content", [b"not json", b"[]", b'{"error":true,"reason":"unavailable"}'])
def test_invalid_json_or_error_payloads_fail(content):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=content,
            )
        )
    ) as client:
        with pytest.raises(CollectionError):
            fetch_json(client, "https://example.test", {})


@pytest.mark.parametrize(
    "retry_after",
    [
        "120",
        format_datetime(datetime.now(UTC) + timedelta(minutes=5), usegmt=True),
    ],
)
def test_long_retry_after_fails_without_sending_early_retry(monkeypatch, retry_after):
    def unexpected_sleep(delay):
        pytest.fail("Should return control instead of waiting or retrying early")

    monkeypatch.setattr("mosamiat.collect.time.sleep", unexpected_sleep)
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                429,
                headers={"Retry-After": retry_after},
            )
        )
    ) as client:
        with pytest.raises(CollectionError, match="cooldown"):
            fetch_json(client, "https://example.test", {})


@pytest.mark.parametrize(
    "start, end",
    [
        (date(2025, 1, 1), None),
        (None, date(2025, 1, 1)),
        (date(2025, 1, 2), date(2025, 1, 1)),
        (date(2025, 1, 1), date(2025, 2, 1)),
        (datetime.now(UTC).date(), datetime.now(UTC).date()),
    ],
)
def test_invalid_windows_are_rejected(start, end):
    with pytest.raises(CollectionError):
        validate_window(start, end)


def test_single_day_and_31_day_windows_are_accepted():
    validate_window(date(2025, 1, 1), date(2025, 1, 1))
    validate_window(date(2025, 1, 1), date(2025, 1, 31))
    last_available = datetime.now(UTC).date() - timedelta(days=6)
    validate_window(last_available, last_available)


def test_config_load_and_bad_coordinates(tmp_path):
    assert City.load(Path("configs/city.toml")) == CITY
    bad = tmp_path / "bad.toml"
    bad.write_text(
        '[city]\ncity_id="test"\nname="Test"\nlatitude=91\nlongitude=73\ntimezone="Asia/Karachi"\n'
    )
    with pytest.raises(CollectionError, match="coordinates"):
        City.load(bad)


def test_cli_invalid_window_exits_cleanly_without_creating_data(tmp_path, capsys):
    assert main(["--output", str(tmp_path), "--start", "2025-01-01"]) == 1
    assert "Provide --start and --end together" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_repeated_collection_creates_distinct_snapshots(tmp_path):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=response_payload(request),
            )
        )
    ) as client:
        first = collect(CITY, tmp_path, client)
        original = (first / "hourly.csv").read_bytes()
        second = collect(CITY, tmp_path, client)
    assert first != second
    assert (first / "hourly.csv").read_bytes() == original
