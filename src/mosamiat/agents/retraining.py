"""Checkpoint drift decisions and train at most one candidate per city per 24 hours."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import pandas as pd

from mosamiat.agents.drift import archive, evaluate, publish_drift
from mosamiat.backfill import digest
from mosamiat.collect import CollectionError
from mosamiat.modeling.classical import evaluate as train_forest
from mosamiat.storage import read_json


def checkpoint(ledger, payload, assess):
    run_id = payload["run_id"]
    with closing(sqlite3.connect(ledger.path)) as db, db:
        db.execute("""CREATE TABLE IF NOT EXISTS drift_decisions (
            run_id TEXT PRIMARY KEY, content TEXT NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS retrain_requests (
            id TEXT PRIMARY KEY, city TEXT, issued TEXT)""")
        row = db.execute("SELECT content FROM drift_decisions WHERE run_id=?", (run_id,)).fetchone()
        if row:
            return json.loads(row[0])
        decision = assess()
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT content FROM drift_decisions WHERE run_id=?", (run_id,)).fetchone()
        if row:
            return json.loads(row[0])
        request = ""
        action = "monitor"
        if decision["drift_count"]:
            stamp = pd.Timestamp(payload["issued_at_utc"])
            city = payload["city_id"]
            day = pd.Timedelta(days=1)
            existing = db.execute(
                "SELECT id FROM retrain_requests WHERE city=? AND issued>? AND issued<? ORDER BY issued LIMIT 1",
                (city, (stamp - day).isoformat(), (stamp + day).isoformat()),
            ).fetchone()
            if existing:
                request, action = existing[0], "cooldown"
            else:
                request, action = f"{city}_{stamp:%Y%m%dT%H%M%SZ}", "retrain"
                db.execute(
                    "INSERT INTO retrain_requests VALUES(?,?,?)", (request, city, stamp.isoformat())
                )
        decision.update(request_id=request, action=action)
        db.execute(
            "INSERT INTO drift_decisions VALUES(?,?)",
            (run_id, json.dumps(decision, allow_nan=False)),
        )
        return decision


def fit_candidate(city, request, root):
    work = root / "candidates" / request
    work.mkdir(parents=True, exist_ok=True)
    output = work / "evaluation"
    # The process lock is released by SQLite if the worker exits, also during a fit.
    with closing(sqlite3.connect(work / "worker.sqlite3", timeout=10)) as worker, worker:
        worker.execute("BEGIN IMMEDIATE")
        path = output / "manifest.json"
        if output.exists():
            result = read_json(path)
            if result.get("status") == "complete" and all(
                digest(output / f) == h for f, h in result["sha256"].items()
            ):
                return digest(path)
            raise CollectionError("Existing candidate evaluation failed integrity verification")
        stamp = pd.to_datetime(request.rsplit("_", 1)[1], format="%Y%m%dT%H%M%SZ", utc=True)
        end = stamp.normalize() - pd.Timedelta(days=6)
        # Isolate source and fitting locks from earlier interrupted attempts.
        attempt = work / f"fit-{uuid4().hex}"
        attempt.mkdir()
        dataset, labels = archive(city, end - pd.Timedelta(days=60), end, attempt)
        train_forest(dataset, attempt / "evaluation", city, labels, Path("configs/features.toml"))
        (attempt / "evaluation").rename(output)
        return digest(path)


def run_drift(city, payload, ledger, store, root=Path("data/drift"), trainer=None):
    decision = checkpoint(ledger, payload, lambda: evaluate(city, payload, ledger, root))
    result = dict(decision, candidate_sha256="", training_status="not_requested")
    if decision["request_id"]:
        try:
            result["candidate_sha256"] = (trainer or fit_candidate)(
                city, decision["request_id"], root
            )
            result["training_status"] = "candidate_evaluated"
        except Exception as error:
            result["training_status"] = "failed"
            ledger.log(payload["run_id"], "retraining_failed", type(error).__name__)
            publish_drift(store, result)
            raise
    publish_drift(store, result)
    ledger.log(
        payload["run_id"],
        "drift_verified",
        f"{result['drift_count']} crossings, {result['action']}, {result['training_status']}",
    )
    return result
