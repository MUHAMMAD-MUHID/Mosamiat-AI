import json
from dataclasses import replace
from datetime import date

import httpx
import pandas as pd
import pytest
from test_collect import CITY, response_payload

from mosamiat.backfill import backfill, hourly_frame, windows
from mosamiat.collect import CollectionError


def test_month_windows_cover_leap_day_without_overlap():
    assert list(windows(date(2024, 1, 30), date(2024, 3, 1))) == [
        (date(2024, 1, 30), date(2024, 1, 31)),
        (date(2024, 2, 1), date(2024, 2, 29)),
        (date(2024, 3, 1), date(2024, 3, 1)),
    ]


def test_resume_integrity_missing_streaks_and_unique_hours(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        data = response_payload(request)
        if "temperature_2m" in data["hourly"]:
            values = data["hourly"]["temperature_2m"]
            if request.url.params["start_date"] == "2025-01-31":
                values[-3:] = [None] * 3
            else:
                values[:2] = [None] * 2
        return httpx.Response(200, json=data)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        args = (CITY, tmp_path, client, date(2025, 1, 31), date(2025, 2, 1))
        run = backfill(*args)
        original = (run / "hourly.csv").read_bytes()
        frame = pd.read_csv(run / "hourly.csv")
        assert len(frame) == 48
        assert not frame.duplicated(["city_id", "valid_at_utc"]).any()
        assert frame.rain_probability.isna().all()
        assert frame.rain_probability_missing.all()
        report = json.loads((run / "quality.json").read_text())
        assert report["metrics"]["temperature_c"] == {
            "missing": 5,
            "longest_gap_hours": 5,
            "missing_pct": 10.416667,
        }
        assert report["metrics"]["rain_probability"]["longest_gap_hours"] == 48
        assert backfill(*args) == run
        assert len(calls) == 4
        assert (run / "hourly.csv").read_bytes() == original
        # An altered snapshot cannot silently become training data.
        raw = next((run.parent / "raw").glob("*/hourly.csv"))
        raw.write_text("corrupted")
        backfill(*args)
        assert len(calls) == 6
        assert len(pd.read_csv(run / "hourly.csv")) == 48


def test_failure_resumes_completed_window_and_keeps_diagnostics(tmp_path):
    calls = []
    fail = True

    def handler(request):
        calls.append(request)
        if fail and request.url.params["start_date"] == "2025-02-01":
            return httpx.Response(400)
        return httpx.Response(200, json=response_payload(request))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        args = (CITY, tmp_path, client, date(2025, 1, 31), date(2025, 2, 1))
        with pytest.raises(CollectionError):
            backfill(*args)
        destination = next(tmp_path.glob("*/2025-01-31_2025-02-01"))
        assert not (destination / "hourly.csv").exists()
        assert json.loads((destination / "manifest.json").read_text())["status"] == "failed"
        fail = False
        backfill(*args)
        assert len(calls) == 5  # 2 successful, 1 failed, 2 resumed requests.
        assert len(pd.read_csv(destination / "hourly.csv")) == 48


def test_bad_ranges_do_not_create_output(tmp_path):
    with httpx.Client() as client:
        with pytest.raises(CollectionError):
            backfill(CITY, tmp_path / "output", client, date(2025, 2, 1), date(2025, 1, 1))
    assert not list(tmp_path.iterdir())


def test_city_settings_isolate_cache_and_lock_prevents_concurrent_write(tmp_path):
    def handler(request):
        return httpx.Response(200, json=response_payload(request))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        start = end = date(2025, 1, 1)
        run = backfill(CITY, tmp_path, client, start, end)
        other = backfill(replace(CITY, longitude=74), tmp_path, client, start, end)
        assert run.parent != other.parent
        lock = run.parent / ".lock"
        lock.touch()
        with pytest.raises(CollectionError, match="locked"):
            backfill(CITY, tmp_path, client, start, end)
        assert lock.exists()


@pytest.mark.parametrize("mutation", ["duplicate", "gap", "off_hour"])
def test_alignment_rejects_ambiguous_or_missing_hours(tmp_path, mutation):
    def handler(request):
        return httpx.Response(200, json=response_payload(request))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        run = backfill(CITY, tmp_path, client, date(2025, 1, 1), date(2025, 1, 1))
    raw = next((run.parent / "raw").iterdir())
    frame = pd.read_csv(raw / "hourly.csv")
    if mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif mutation == "gap":
        frame = frame[frame.valid_at_utc != frame.valid_at_utc.iloc[0]]
    else:
        frame.loc[0, "valid_at_utc"] = "2025-01-01T00:30:00+00:00"
    frame.to_csv(raw / "hourly.csv", index=False)
    with pytest.raises(CollectionError):
        hourly_frame(raw, date(2025, 1, 1), date(2025, 1, 1))
