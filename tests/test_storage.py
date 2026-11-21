import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pandas as pd
import pytest
from test_collect import CITY, response_payload

from mosamiat.backfill import backfill
from mosamiat.collect import CollectionError, collect
from mosamiat.storage import Repository, connect, hostname, load_snapshot, main, snapshots


@pytest.mark.parametrize("host", ["instance.example.com", "https://instance.example.com/"])
def test_host_accepts_hostname_or_https_root(host):
    assert hostname(host) == "instance.example.com"


@pytest.mark.parametrize(
    "host",
    ["", "http://example.com", "https://example.com/project", "https://user:secret@example.com"],
)
def test_host_rejects_invalid_settings_without_echoing_them(host):
    with pytest.raises(CollectionError, match="instance hostname") as error:
        hostname(host)
    assert "secret" not in str(error.value)


def test_connect_normalizes_url_and_uses_ignored_windows_certificate_directory(monkeypatch):
    login = MagicMock()
    monkeypatch.setitem(sys.modules, "hopsworks", SimpleNamespace(login=login))
    monkeypatch.setitem(sys.modules, "dotenv", SimpleNamespace(load_dotenv=MagicMock()))
    monkeypatch.setattr("mosamiat.storage.configure_project_certificates", MagicMock())
    for name, value in {
        "API_KEY": "test-only-key",
        "PROJECT": "test-project",
        "HOST": "https://instance.example.com/",
    }.items():
        monkeypatch.setenv(f"HOPSWORKS_{name}", value)
    assert connect() is login.return_value.get_feature_store.return_value
    settings = login.call_args.kwargs
    assert settings["host"] == "instance.example.com"
    assert Path(settings["cert_folder"]).parent == (Path.cwd() / ".cache" / "hopsworks").resolve()


def test_project_pem_reuse_is_limited_to_authenticated_store_paths(tmp_path, monkeypatch):
    from mosamiat.hopsworks_compat import configure_project_certificates

    instance = MagicMock()
    instance._get_jks_key_store_path.return_value = str(tmp_path / "keyStore.jks")
    instance._get_jks_trust_store_path.return_value = str(tmp_path / "trustStore.jks")
    paths = [tmp_path / name for name in ("ca.pem", "cert.pem", "key.pem")]
    for path in paths:
        path.write_text("test-only-placeholder")
    instance._get_ca_chain_path.return_value = str(paths[0])
    instance._get_client_cert_path.return_value = str(paths[1])
    instance._get_client_key_path.return_value = str(paths[2])
    original = instance._write_pem
    monkeypatch.setitem(
        sys.modules,
        "hopsworks_common",
        SimpleNamespace(client=SimpleNamespace(get_instance=lambda: instance)),
    )
    configure_project_certificates()
    assert instance._write_pem(
        tmp_path / "keyStore.jks", "unused", tmp_path / "trustStore.jks", "unused", "kafka"
    ) == tuple(map(str, paths))
    original.assert_not_called()
    instance._write_pem("other.jks", "other", "other-trust.jks", "other", "external")
    original.assert_called_once()
    paths[2].unlink()
    with pytest.raises(FileNotFoundError, match="PEM files are missing"):
        instance._write_pem(
            tmp_path / "keyStore.jks", "unused", tmp_path / "trustStore.jks", "unused", "kafka"
        )


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
    frame["retrieved_at_utc"] = pd.Timestamp("2025-03-01T00:00:00.050684Z")
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
    fg.retrieved_at_utc.__ge__.assert_called_with(pd.Timestamp("2025-03-01T00:00:00Z"))
    fg.retrieved_at_utc.__le__.assert_called_with(pd.Timestamp("2025-03-01T00:00:01Z"))
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
