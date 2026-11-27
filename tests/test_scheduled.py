import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from test_collect import CITY

scheduled = pytest.importorskip("mosamiat.scheduled")


def test_materialization_polls_original_job_and_bounds_wait(monkeypatch):
    from mosamiat.agents import ledger

    monkeypatch.setattr(ledger.time, "sleep", lambda _: None)
    running = SimpleNamespace(id=12, success=None)
    finished = SimpleNamespace(id=12, success=True)
    unrelated = SimpleNamespace(id=13, success=False)
    job = Mock()
    job.get_executions.side_effect = [[running], [unrelated, finished]]
    ledger._wait_for_materialization(job, "features")
    job.get_executions.side_effect = [[running]]
    with pytest.raises(scheduled.CollectionError, match="exceeded"):
        ledger._wait_for_materialization(job, "features", timeout=0)
    job.get_executions.side_effect = [[unrelated]]
    with pytest.raises(scheduled.CollectionError, match="failed"):
        ledger._wait_for_materialization(job, "features")


class DatasetFiles:
    """Exercise the SDK's directory upload and full destination move semantics."""

    def __init__(self, root):
        self.root = root
        (root / "Resources").mkdir(parents=True)

    def exists(self, remote):
        return (self.root / remote).exists()

    def mkdir(self, remote):
        (self.root / remote).mkdir()

    def list(self, remote, offset, limit):
        return [f"/{remote}/{p.name}" for p in sorted((self.root / remote).iterdir())][
            offset : offset + limit
        ]

    def upload(self, local, remote, **options):
        shutil.copy(local, self.root / remote / Path(local).name)

    def download(self, remote, local_path):
        shutil.copyfile(self.root / remote, local_path)

    def move(self, source, dest):
        assert not self.exists(dest)
        (self.root / source).rename(self.root / dest)

    def remove(self, remote):
        (self.root / remote).unlink()


def test_disposable_runner_recovers_ledger_snapshot_and_rejects_corruption(tmp_path, monkeypatch):
    api = DatasetFiles(tmp_path / "remote")
    first = tmp_path / "first"
    first.mkdir()
    monkeypatch.chdir(first)
    ledger = scheduled.ScheduledLedger(api, CITY, first)
    source = first / "data/collection/sample"
    source.mkdir(parents=True)
    for name in scheduled.SNAPSHOT:
        (source / name).write_text(name)
    ledger.log("run1", "started", "scheduled forecast")
    ledger.prepare(dict(run_id="run1", snapshot_id="sample", snapshot_path=str(source)))
    second = tmp_path / "second"
    second.mkdir()
    monkeypatch.chdir(second)
    resumed = scheduled.ScheduledLedger(api, CITY, second)
    assert resumed.interrupted() == ["run1"]
    resumed.restore_source("run1")
    assert Path(resumed.get("run1")["snapshot_path"], "weather.json").read_text() == "weather.json"
    resumed.log("run1", "completed", "all four agents")
    resumed.save()
    resumed.save()
    assert len(list(resumed.checkpoints.names())) == 2
    assert resumed.interrupted() == []
    (api.root / resumed.checkpoints.folder / resumed.versions[-1]).write_bytes(b"broken")
    third = tmp_path / "third"
    third.mkdir()
    with pytest.raises(scheduled.CollectionError, match="checksum"):
        scheduled.ScheduledLedger(api, CITY, third)


def test_hourly_skips_a_previously_completed_forecast(tmp_path, monkeypatch, capsys):
    project = Mock()
    api = DatasetFiles(tmp_path / "remote")
    project.get_dataset_api.return_value = api
    first = tmp_path / "first"
    first.mkdir()
    ledger = scheduled.ScheduledLedger(api, CITY, first)
    now = pd.Timestamp("2026-11-27T07:12:00Z")
    run_id = f"{CITY.city_id}_20261127T07Z"
    ledger.log(run_id, "completed", "all agents verified")
    ledger.log(run_id, "failed", "a later duplicate write failed")
    ledger.save()
    monkeypatch.setattr(
        scheduled, "pd", SimpleNamespace(Timestamp=Mock(now=Mock(return_value=now)))
    )
    graph = Mock()
    monkeypatch.setattr(scheduled, "build_graph", Mock(return_value=graph))

    second = tmp_path / "second"
    second.mkdir()
    scheduled.hourly(project, CITY, second, retry=None)

    graph.invoke.assert_not_called()
    assert list(ledger.checkpoints.names()) == ledger.versions
    assert f"{run_id}: forecast already verified" in capsys.readouterr().out


def test_daily_restores_same_shared_training_checkpoint(tmp_path, monkeypatch):
    project = Mock()
    project.get_dataset_api.return_value = DatasetFiles(tmp_path / "remote")
    day = pd.Timestamp("2026-11-24", tz="UTC")

    def train(store, city, cutoff, work, output):
        assert cutoff == day
        for name in scheduled.ARTIFACTS:
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "sha256": {
                        name: "checksum" for name in scheduled.ARTIFACTS if name != "manifest.json"
                    }
                }
            )
        )
        return "sha"

    fit = Mock(side_effect=train)
    monkeypatch.setattr(scheduled, "train", fit)
    verify = Mock(return_value="sha")
    monkeypatch.setattr(scheduled, "verify_training", verify)
    register = Mock(side_effect=[ConnectionError(), {}])
    monkeypatch.setattr(scheduled, "register_and_promote", register)
    for attempt in (1, 2):
        work = tmp_path / str(attempt)
        work.mkdir()
        if attempt == 1:
            with pytest.raises(ConnectionError):
                scheduled.daily(project, CITY, work, day)
        else:
            scheduled.daily(project, CITY, work, day)
            assert (work / "evaluation/lstm/model.pt").read_text() == "lstm/model.pt"
    fit.assert_called_once()
    verify.assert_called_once()
    assert [call.args[-1] for call in register.call_args_list] == [1020261124, 1020261124]


def test_drift_worker_registers_and_reuses_shared_artifacts(tmp_path, monkeypatch):
    from mosamiat.backfill import digest, write_json
    from mosamiat.modeling.training import TRAINERS

    project = Mock()
    project.get_dataset_api.return_value = DatasetFiles(tmp_path / "remote")
    request = "islamabad_20261127T070134Z"

    def fit(city, request, root, store):
        output = root / "candidates" / request / "evaluation"
        checksums = {}
        for name in scheduled.ARTIFACTS[1:]:
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
            checksums[name] = digest(path)
        write_json(
            output / "manifest.json",
            {
                "status": "complete",
                "families": {name: {"status": "complete"} for name in TRAINERS},
                "sha256": checksums,
            },
        )
        return scheduled.verify_training(output)

    trainer = Mock(side_effect=fit)
    register = Mock()
    monkeypatch.setattr(scheduled, "fit_candidate", trainer)
    monkeypatch.setattr(scheduled, "register_and_promote", register)
    first = scheduled.train_on_drift(project, CITY, request, tmp_path / "first")
    assert scheduled.train_on_drift(project, CITY, request, tmp_path / "retry") == first
    trainer.assert_called_once()
    assert register.call_count == 2
    assert register.call_args.args[-1] == int(pd.Timestamp("2026-11-27T07:01:34Z").timestamp())


def test_promotion_updates_serving_only_when_gate_changes_selection(tmp_path, monkeypatch):
    from mosamiat.backfill import write_json
    from mosamiat.modeling.inference import read_policy

    project = Mock()
    api = DatasetFiles(tmp_path / "remote")
    project.get_dataset_api.return_value = api
    path = tmp_path / "evaluation"
    path.mkdir()
    write_json(
        path / "manifest.json",
        {
            "status": "complete",
            "families": {"lstm": {"status": "complete"}},
        },
    )
    candidate = {"name": "lstm", "version": 7, "family": "lstm", "manifest_sha256": "a" * 64}
    policy = {
        "schema_version": 1,
        "city_id": CITY.city_id,
        "models": {"temperature_c:1": candidate},
        "error_limits": {},
    }
    monkeypatch.setattr(
        scheduled, "register_candidates", Mock(return_value=({"lstm": candidate}, {}))
    )
    monkeypatch.setattr(scheduled, "promotion", Mock(side_effect=[(policy, {}), (policy, {})]))
    scheduled.register_and_promote(project, CITY, path, 7)
    saved, checksum = read_policy(api, CITY)
    assert saved == policy
    scheduled.register_and_promote(project, CITY, path, 8)
    assert read_policy(api, CITY) == (policy, checksum)
