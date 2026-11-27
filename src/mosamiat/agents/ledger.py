"""Checkpoint the first forecast for a city/hour and verify its remote ledger rows."""

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import pandas as pd

from mosamiat.agents.alerts import RULE_VERSION, forecast_hours, new_episodes
from mosamiat.agents.weather_report import render
from mosamiat.collect import CollectionError


class Ledger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with closing(sqlite3.connect(path)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS events (run_id TEXT, at_utc TEXT, status TEXT, detail TEXT)"
            )
            db.execute("""CREATE TABLE IF NOT EXISTS bulletins (
                run_id TEXT PRIMARY KEY, city_id TEXT, issued_at_utc TEXT, local_day TEXT,
                has_report INTEGER, content TEXT NOT NULL)""")

    def get(self, run_id):
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute("SELECT payload FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def prepare(self, payload):
        run_id = payload["run_id"]
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "INSERT OR IGNORE INTO runs VALUES (?, ?)",
                (run_id, json.dumps(payload, allow_nan=False)),
            )
        return self.get(run_id)

    def log(self, run_id, status, detail):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?)",
                (run_id, pd.Timestamp.now(tz="UTC").isoformat(), status, detail),
            )

    @staticmethod
    def history(path: Path, run_id: str):
        if not path.is_file():
            return []
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in db.execute(
                    "SELECT at_utc, status, detail FROM events WHERE run_id=? ORDER BY rowid",
                    (run_id,),
                )
            ]

    def bulletin(self, city, payload, alerts):
        run_id = payload["run_id"]
        issued = pd.Timestamp(payload["issued_at_utc"])
        day = issued.tz_convert(city.timezone).date().isoformat()
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT content FROM bulletins WHERE run_id=?", (run_id,)).fetchone()
            if row:
                return json.loads(row[0])
            previous = db.execute(
                "SELECT content FROM bulletins WHERE city_id=? AND issued_at_utc<? "
                "ORDER BY issued_at_utc DESC LIMIT 1",
                (city.city_id, issued.isoformat()),
            ).fetchone()
            daily = not db.execute(
                "SELECT 1 FROM bulletins WHERE city_id=? AND local_day=? AND has_report=1 LIMIT 1",
                (city.city_id, day),
            ).fetchone()
            new = new_episodes(alerts, json.loads(previous[0])["alerts"] if previous else [])
            reasons = [
                label
                for label, triggered in (("daily", daily), ("new_alert", bool(new)))
                if triggered
            ]
            report = render(city, payload, forecast_hours(payload), alerts) if reasons else ""
            result = dict(
                run_id=run_id,
                city_id=city.city_id,
                issued_at_utc=issued.isoformat(),
                rule_version=RULE_VERSION,
                alerts=alerts,
                new_alert_count=len(new),
                report_trigger="+".join(reasons) or "skipped",
                report=report,
            )
            db.execute(
                "INSERT INTO bulletins VALUES (?,?,?,?,?,?)",
                (
                    run_id,
                    city.city_id,
                    issued.isoformat(),
                    day,
                    int(bool(report)),
                    json.dumps(result, allow_nan=False),
                ),
            )
            return result


def publish(store, payload):
    metadata = {
        k: v
        for k, v in payload.items()
        if k not in {"forecasts", "snapshot_path", "input_features"}
    }
    frame = pd.DataFrame(payload["forecasts"]).assign(**metadata)
    frame["forecast_id"] = frame.run_id + "_" + frame.horizon_hours.astype(str) + "_" + frame.metric
    timestamps = ["target_at_utc", "reference_at_utc", "issued_at_utc", "retrieved_at_utc"]
    for column in timestamps:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    _write_verified(
        store,
        frame,
        "mosamiat_forecast_ledger",
        "forecast_id",
        "target_at_utc",
        timestamps,
        "Hourly Islamabad model forecasts with source and model versions.",
    )


def publish_bulletin(store, result):
    record = dict(result, alerts=json.dumps(result["alerts"], ensure_ascii=False, allow_nan=False))
    frame = pd.DataFrame([record])
    frame["issued_at_utc"] = pd.to_datetime(frame.issued_at_utc, utc=True)
    _write_verified(
        store,
        frame,
        "mosamiat_weather_bulletins",
        "run_id",
        "issued_at_utc",
        ["issued_at_utc"],
        "Forecast alerts and weather report decisions per hourly run.",
    )


def _wait_for_materialization(job, name, timeout=360):
    deadline = time.monotonic() + timeout
    execution_id = None
    while True:
        # The SDK returns executions newest first. Follow the same ID on later polls.
        executions = job.get_executions()
        if executions and execution_id is None:
            execution_id = executions[0].id
        current = next((item for item in executions if item.id == execution_id), None)
        if current is not None:
            if current.success is True:
                return
            if current.success is False:
                raise CollectionError(f"{name} materialization {execution_id} failed")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CollectionError(f"{name} materialization {execution_id} exceeded {timeout}s")
        time.sleep(min(5, remaining))


def _write_verified(store, frame, name, key, event_time, timestamps, description):
    group = store.get_or_create_feature_group(
        name=name,
        version=1,
        primary_key=[key],
        event_time=event_time,
        online_enabled=False,
        time_travel_format="HUDI",
        statistics_config=False,
        description=description,
    )
    if list(group.primary_key) != [key] or group.event_time != event_time:
        raise CollectionError(f"Incompatible {name} keys")
    print(f"Publishing {len(frame)} rows to {name}", flush=True)
    job, _ = group.insert(frame, operation="upsert", wait=False)
    if job is not None:
        _wait_for_materialization(job, name)
    print(f"Reading back {name}", flush=True)
    actual = (
        group.select_all()
        .filter(group.run_id == frame.run_id.iloc[0])
        .read(read_options={"arrow_flight_config": {"timeout": 90}})
    )
    for column in timestamps:
        actual[column] = pd.to_datetime(actual[column], utc=True)
    try:
        pd.testing.assert_frame_equal(
            frame.set_index(key).sort_index().sort_index(axis=1),
            actual.set_index(key).sort_index().sort_index(axis=1),
            check_dtype=False,
            rtol=1e-9,
            atol=1e-9,
        )
    except AssertionError as error:
        raise CollectionError(f"{name} readback differs from the saved run") from error
    return actual
