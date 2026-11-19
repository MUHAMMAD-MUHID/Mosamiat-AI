import json
from datetime import date
from unittest.mock import MagicMock

import httpx
import pandas as pd
import pytest
from test_collect import CITY, response_payload

from mosamiat.backfill import backfill
from mosamiat.collect import CollectionError, collect
from mosamiat.storage import Repository, load_snapshot, main, snapshots


@pytest.fixture
def run(tmp_path):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=response_payload(request))
        )
    ) as client:
        return collect(CITY, tmp_path, client, date(2025, 1, 1), date(2025, 1, 1))


def test_stable_keys_nulls_and_revisions(run):
    original = load_snapshot(run)
    pd.testing.assert_frame_equal(original, load_snapshot(run))
    assert len(original) == 24 * 13
    assert original.record_id.is_unique
    assert not original.is_ground_truth.any()
    raw = json.loads((run / "weather.json").read_text(encoding="utf-8"))
    raw["response"]["hourly"]["temperature_2m"][0] = None
    (run / "weather.json").write_text(json.dumps(raw), encoding="utf-8")
    missing = load_snapshot(run)
    assert missing.value.isna().sum() == 1
    assert list(missing.record_id) == list(original.record_id)
    raw["retrieved_at_utc"] = "2025-03-01T00:00:00+00:00"
    (run / "weather.json").write_text(json.dumps(raw), encoding="utf-8")
    revised = load_snapshot(run)
    assert set(revised[revised["product"] == "era5"].record_id).isdisjoint(original.record_id)


@pytest.mark.parametrize("mutation", ["failed", "wrong_unit", "coordinates", "coverage"])
def test_invalid_snapshots_fail_before_connect(run, mutation, monkeypatch):
    def unexpected():
        pytest.fail("Invalid source must not connect")

    monkeypatch.setattr("mosamiat.storage.connect", unexpected)
    manifest = json.loads((run / "manifest.json").read_text())
    raw = json.loads((run / "weather.json").read_text(encoding="utf-8"))
    if mutation == "failed":
        manifest["status"] = "failed"
    elif mutation == "wrong_unit":
        raw["response"]["hourly_units"]["temperature_2m"] = "F"
    elif mutation == "coordinates":
        raw["parameters"]["latitude"] = 0
    else:
        manifest["hours"] = 25
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "weather.json").write_text(json.dumps(raw), encoding="utf-8")
    assert main([str(run)]) == 1


def test_backfill_upload_rejects_changed_artifacts(tmp_path):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=response_payload(request))
        )
    ) as client:
        dataset = backfill(CITY, tmp_path, client, date(2025, 1, 1), date(2025, 1, 2))
    assert len(list(snapshots(dataset))) == 1
    assert main([str(dataset), "--dry-run"]) == 0
    (dataset / "hourly.csv").write_text("changed")
    with pytest.raises(CollectionError, match="integrity"):
        list(snapshots(dataset))


def test_upsert_reuses_group_and_waits_for_job(run):
    store = MagicMock()
    fg = store.get_or_create_feature_group.return_value
    fg.primary_key = ["record_id"]
    fg.event_time = "valid_at_utc"
    repo = Repository(store)
    frame = load_snapshot(run)
    repo.upsert(frame)
    repo.upsert(frame)
    assert store.get_or_create_feature_group.call_count == 1
    assert fg.insert.call_count == 2
    assert fg.insert.call_args.kwargs == {
        "operation": "upsert",
        "wait": True,
        "write_options": {"wait_for_job": True},
    }
    pd.testing.assert_frame_equal(
        fg.insert.call_args_list[0].args[0], fg.insert.call_args_list[1].args[0]
    )
    fg.primary_key = ["wrong"]
    with pytest.raises(CollectionError, match="incompatible"):
        Repository(store).upsert(frame)


def test_readback_detects_changed_values_and_duplicate_records(run):
    frame = load_snapshot(run)
    fg = MagicMock()
    # Behavioral mock: SDK/server compatibility still requires the live check.
    for name in ("valid_at_utc", "retrieved_at_utc"):
        getattr(fg, name).__ge__.return_value = MagicMock()
        getattr(fg, name).__le__.return_value = MagicMock()
    repo = Repository(MagicMock())
    repo.group = fg
    read = fg.select_all.return_value.filter.return_value.read
    read.return_value = frame.iloc[::-1].copy()
    repo.verify(frame)
    read.return_value.loc[0, "value"] = -999
    with pytest.raises(CollectionError, match="readback"):
        repo.verify(frame)
    read.return_value = pd.concat([frame, frame.iloc[:1]])
    with pytest.raises(CollectionError, match="readback"):
        repo.verify(frame)


def test_sdk_failure_does_not_print_credentials(run, monkeypatch, capsys):
    def fail():
        raise RuntimeError("secret-key-example")

    monkeypatch.setattr("mosamiat.storage.connect", fail)
    assert main([str(run)]) == 1
    assert "secret-key-example" not in capsys.readouterr().err
