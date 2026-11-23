import shutil
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from test_collect import CITY

scheduled = pytest.importorskip("mosamiat.scheduled")


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

    def upload(self, local, remote):
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


def test_daily_restores_same_candidate_after_registry_failure(tmp_path, monkeypatch):
    project = Mock()
    project.get_dataset_api.return_value = DatasetFiles(tmp_path / "remote")
    day = pd.Timestamp("2026-11-24", tz="UTC")
    archive = Mock(return_value=(tmp_path, tmp_path / "labels"))
    monkeypatch.setattr(scheduled, "archive", archive)

    def build(dataset, output, *args):
        output.mkdir()
        for name in ("features.csv", "manifest.json"):
            (output / name).write_text(name)

    def train(dataset, output, *args):
        output.mkdir()
        for name in scheduled.EVALUATION:
            (output / name).write_text(name)

    fit = Mock(side_effect=train)
    monkeypatch.setattr(scheduled, "build", build)
    monkeypatch.setattr(scheduled, "evaluate", fit)
    register = Mock(side_effect=[ConnectionError(), ("model", "sha")])
    monkeypatch.setattr(scheduled, "register", register)
    for attempt in (1, 2):
        work = tmp_path / str(attempt)
        work.mkdir()
        if attempt == 1:
            with pytest.raises(ConnectionError):
                scheduled.daily(project, CITY, work, day)
        else:
            scheduled.daily(project, CITY, work, day)
            assert (work / "evaluation/model.joblib").read_text() == "model.joblib"
    fit.assert_called_once()
    _, start, end, _ = archive.call_args.args
    assert (end - start).days == 60 and (day - end).days == 6
    assert [call.args[2] for call in register.call_args_list] == [20261124, 20261124]
