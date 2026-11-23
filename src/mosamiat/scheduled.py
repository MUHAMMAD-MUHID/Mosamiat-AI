"""Run the hourly and daily pipelines on disposable GitHub Actions workers."""

import argparse
import re
import shutil
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from mosamiat.agents.drift import archive
from mosamiat.agents.graph import build_graph
from mosamiat.agents.ledger import Ledger, _write_verified
from mosamiat.agents.retraining import fit_candidate
from mosamiat.backfill import digest
from mosamiat.collect import City, CollectionError
from mosamiat.features.pipeline import build
from mosamiat.modeling.classical import evaluate
from mosamiat.modeling.registry import register
from mosamiat.storage import connect_project

SNAPSHOT = ("weather.json", "air_quality.json", "manifest.json", "hourly.csv")
EVALUATION = (
    "manifest.json",
    "model.joblib",
    "metrics.csv",
    "cases.csv",
    "rain_reliability.csv",
    "rain_climatology.csv",
)
DAILY = tuple("evaluation/" + name for name in EVALUATION) + (
    "features/features.csv",
    "features/manifest.json",
)
LEDGER_NAME = re.compile(r"ledger-(\d{10})-([0-9a-f]{64})\.zip")
CONFIG = Path("configs/features.toml")


def pack(directory, names, path):
    with ZipFile(path, "w", ZIP_DEFLATED) as bundle:
        for name in names:
            bundle.write(directory / name, name)


def unpack(path, names, directory):
    with ZipFile(path) as bundle:
        if sorted(bundle.namelist()) != sorted(names):
            raise CollectionError("Unexpected checkpoint members")
        for name in names:
            dest = directory / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(name) as source, dest.open("wb") as out:
                shutil.copyfileobj(source, out)


class Files:
    def __init__(self, api, city, folder):
        self.api = api
        base = f"Resources/mosamiat_{city.city_id}"
        self.folder = f"{base}/{folder}"
        for directory in (base, self.folder):
            if not api.exists(directory):
                api.mkdir(directory)

    def names(self):
        offset = 0
        while True:
            page = self.api.list(self.folder, offset=offset, limit=100)
            yield from (p.rsplit("/", 1)[-1] for p in page)
            if len(page) < 100:
                break
            offset += len(page)

    def download(self, name, path):
        self.api.download(f"{self.folder}/{name}", local_path=str(path))

    def publish(self, path, name):
        # Publish only complete bytes. Failed workers can leave an unreferenced stage file.
        with TemporaryDirectory() as tmp:
            stage = Path(tmp) / f"stage-{uuid4().hex}"
            shutil.copyfile(path, stage)
            self.api.upload(str(stage), self.folder)
            check = Path(tmp) / "readback"
            self.download(stage.name, check)
            if digest(check) != digest(path):
                raise CollectionError("Cloud checkpoint readback differs from upload")
            self.api.move(f"{self.folder}/{stage.name}", f"{self.folder}/{name}")


class ScheduledLedger(Ledger):
    def __init__(self, api, city, directory):
        self.checkpoints = Files(api, city, "ledgers")
        self.snapshots = Files(api, city, "snapshots")
        self.versions = sorted(n for n in self.checkpoints.names() if LEDGER_NAME.fullmatch(n))
        if self.versions:
            name = self.versions[-1]
            bundle = directory / "restore.zip"
            self.checkpoints.download(name, bundle)
            if digest(bundle) != LEDGER_NAME.fullmatch(name)[2]:
                raise CollectionError("Cloud ledger checksum mismatch")
            unpack(bundle, ("ledger.sqlite3",), directory)
        super().__init__(directory / "ledger.sqlite3")

    def save(self):
        with TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "ledger.zip"
            # Ledger methods close their SQLite connections before returning.
            pack(self.path.parent, (self.path.name,), bundle)
            seq = int(LEDGER_NAME.fullmatch(self.versions[-1])[1]) + 1 if self.versions else 1
            name = f"ledger-{seq:010d}-{digest(bundle)}.zip"
            self.checkpoints.publish(bundle, name)
            self.versions.append(name)
        # Keep the last two verified generations in this scheduler's checkpoint folder.
        for old in self.versions[:-2]:
            self.checkpoints.api.remove(f"{self.checkpoints.folder}/{old}")
        self.versions = self.versions[-2:]

    def prepare(self, payload):
        path = Path(payload["snapshot_path"])
        with TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "source.zip"
            pack(path, SNAPSHOT, bundle)
            self.snapshots.publish(bundle, f"{path.name}.zip")
        # Store a portable path that also works on the next runner.
        payload["snapshot_path"] = str(Path("data/collection") / path.name)
        result = super().prepare(payload)
        self.save()
        return result

    def restore_source(self, run_id):
        payload = self.get(run_id)
        if payload:
            name = payload["snapshot_id"]
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
                raise CollectionError("Invalid checkpoint snapshot ID")
            with TemporaryDirectory() as tmp:
                bundle = Path(tmp) / "source.zip"
                self.snapshots.download(f"{name}.zip", bundle)
                unpack(bundle, SNAPSHOT, Path("data/collection") / name)

    def interrupted(self):
        with closing(sqlite3.connect(self.path)) as db:
            return [
                row[0]
                for row in db.execute("""
                    SELECT runs.run_id FROM runs WHERE
                    (SELECT status FROM events WHERE events.run_id = runs.run_id
                     ORDER BY rowid DESC LIMIT 1) NOT IN ('completed','failed')
                    ORDER BY runs.rowid
                """)
            ]


def publish_inputs(store, payload):
    frame = pd.DataFrame([payload["input_features"]], dtype=float)
    frame = frame.assign(
        run_id=payload["run_id"],
        city_id=payload["city_id"],
        reference_at_utc=pd.Timestamp(payload["reference_at_utc"]),
        retrieved_at_utc=pd.Timestamp(payload["retrieved_at_utc"]),
        source_sha256=payload["source_sha256"],
        evaluation_sha256=payload["evaluation_sha256"],
    )
    _write_verified(
        store,
        frame,
        "mosamiat_hourly_features",
        "run_id",
        "reference_at_utc",
        ["reference_at_utc", "retrieved_at_utc"],
        "Past-only feature row used by the forecast model.",
    )


def hourly(project, city, work, retry):
    api = project.get_dataset_api()
    ledger = ScheduledLedger(api, city, work)
    candidates = Files(api, city, "drift_candidates")

    def trainer(city, request, root):
        name = request + ".zip"
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.zip"
            if name in candidates.names():
                candidates.download(name, path)
                unpack(path, EVALUATION, root / "candidates" / request / "evaluation")
                # The existing worker verifies all saved model and metric hashes.
                return fit_candidate(city, request, root)
            checksum = fit_candidate(city, request, root)
            pack(root / "candidates" / request / "evaluation", EVALUATION, path)
            candidates.publish(path, name)
            return checksum

    app = build_graph(city, ledger, lambda: project, publish_inputs=publish_inputs, trainer=trainer)

    def issue(run_id, reference, replay):
        ledger.restore_source(run_id)
        ledger.log(run_id, "started", "scheduled retry" if replay else "scheduled forecast")
        try:
            state = app.invoke(
                dict(run_id=run_id, reference_at_utc=reference.isoformat(), retry=replay)
            )
            ledger.log(run_id, "completed", "All four agents finished")
        except Exception as error:
            ledger.log(run_id, "failed", type(error).__name__)
            raise
        finally:
            ledger.save()
        print(f"{run_id}: verified {len(state['payload']['forecasts'])} forecast rows")

    resumed = [retry] if retry else ledger.interrupted()
    for run_id in resumed:
        payload = ledger.get(run_id)
        if payload is None:
            raise CollectionError("Requested forecast is missing from the scheduled ledger")
        issue(run_id, pd.Timestamp(payload["reference_at_utc"]), True)
    if not retry:
        reference = pd.Timestamp.now(tz="UTC").floor("h")
        run_id = f"{city.city_id}_{reference:%Y%m%dT%HZ}"
        if run_id not in resumed:
            issue(run_id, reference, False)


def daily(project, city, work, day):
    files = Files(project.get_dataset_api(), city, "daily_candidates")
    name = f"{day:%Y%m%d}.zip"
    bundle = work / name
    if name in files.names():
        files.download(name, bundle)
        unpack(bundle, DAILY, work)
    else:
        end = day - pd.Timedelta(days=6)
        dataset, labels = archive(city, end - pd.Timedelta(days=60), end, work)
        build(dataset, work / "features", city, CONFIG)
        evaluate(dataset, work / "evaluation", city, labels, CONFIG)
        pack(work, DAILY, bundle)
        files.publish(bundle, name)
    version = int(day.strftime("%Y%m%d"))
    model, sha = register(
        work / "evaluation", work / "features", version, project.get_model_registry
    )
    print(f"Verified daily candidate {model} v{version}, manifest {sha}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("hourly").add_argument("--retry", metavar="RUN_ID")
    modes.add_parser("daily")
    args = parser.parse_args(argv)
    city = City.load(Path("configs/city.toml"))
    try:
        project = connect_project()
        with TemporaryDirectory() as tmp:
            if args.mode == "daily":
                daily(project, city, Path(tmp), pd.Timestamp.now(tz="UTC").normalize())
            else:
                hourly(project, city, Path(tmp), args.retry)
    except Exception as error:
        # SDK exception text can contain authentication information.
        print(f"{args.mode} pipeline failed ({type(error).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
