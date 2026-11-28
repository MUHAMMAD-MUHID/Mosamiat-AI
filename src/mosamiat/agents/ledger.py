"""Checkpoint the first forecast for a city/hour and verify its remote ledger rows."""

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import httpx
import pandas as pd
import requests

from mosamiat.agents.alerts import RULE_VERSION, forecast_hours, new_episodes
from mosamiat.agents.weather_report import render
from mosamiat.checkpoints import VERIFIED_BULLETIN
from mosamiat.collect import CollectionError
from mosamiat.hopsworks_compat import insert_with_execution


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
            db.execute("""CREATE TABLE IF NOT EXISTS materializations (
                run_id TEXT, feature_group TEXT, execution_id INTEGER,
                PRIMARY KEY (run_id, feature_group))""")

    def checkpoint(self):
        """The local ledger is durable once its SQLite transaction commits."""

    def materialization(self, run_id, feature_group):
        with closing(sqlite3.connect(self.path)) as db:
            return db.execute(
                "SELECT execution_id FROM materializations WHERE run_id=? AND feature_group=?",
                (run_id, feature_group),
            ).fetchone()

    def record_materialization(self, run_id, feature_group, execution_id, *, previous=None):
        with closing(sqlite3.connect(self.path)) as db, db:
            if previous is None:
                change = db.execute(
                    "INSERT OR IGNORE INTO materializations VALUES (?, ?, ?)",
                    (run_id, feature_group, execution_id),
                )
            else:
                change = db.execute(
                    "UPDATE materializations SET execution_id=? "
                    "WHERE run_id=? AND feature_group=? AND execution_id IS ?",
                    (execution_id, run_id, feature_group, previous[0]),
                )
            if change.rowcount != 1:
                raise CollectionError("Materialization record changed while preparing the write")
        self.checkpoint()

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
                "SELECT content FROM bulletins b WHERE city_id=? AND issued_at_utc<? "
                f"AND {VERIFIED_BULLETIN} "
                "ORDER BY issued_at_utc DESC LIMIT 1",
                (city.city_id, issued.isoformat()),
            ).fetchone()
            daily = not db.execute(
                f"SELECT 1 FROM bulletins b WHERE city_id=? AND local_day=? AND has_report=1 "
                f"AND {VERIFIED_BULLETIN} LIMIT 1",
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


def publish(store, payload, *, reconcile=False, ledger=None):
    forecast_hours(payload)
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
        reconcile=reconcile,
        ledger=ledger,
    )


def publish_bulletin(store, result, *, reconcile=False, ledger=None):
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
        reconcile=reconcile,
        ledger=ledger,
    )


class MaterializationError(CollectionError):
    """Carry only diagnostics safe to emit from a Hopsworks operation."""

    def __init__(self, operation, name, execution_id, detail, *, error=None, retryable=False):
        status = getattr(getattr(error, "response", None), "status_code", None)
        category = ""
        if isinstance(status, int) and 100 <= status <= 599:
            category = {
                401: "authentication", 403: "authorization", 429: "rate_limit",
            }.get(status, f"{status // 100}xx")
            category = f" http={status} category={category}"
        kind = f" error={type(error).__name__}" if error is not None else ""
        self.retryable = retryable
        super().__init__(
            f"operation={operation} feature_group={name} execution_id={execution_id or 'unknown'}"
            f"{category}{kind}: {detail}"
        )


def safe_error_detail(error):
    return str(error) if isinstance(error, MaterializationError) else type(error).__name__


def _read_retryable(error):
    if isinstance(error, (requests.exceptions.SSLError, requests.exceptions.ProxyError)):
        return False
    return isinstance(error, (
        ConnectionError, TimeoutError, httpx.NetworkError, httpx.TimeoutException,
        requests.exceptions.ConnectionError, requests.exceptions.Timeout,
    ))


def _wait_for_materialization(job, name, execution_id, timeout=360):
    deadline = time.monotonic() + timeout
    while True:
        current = next((item for item in job.get_executions() if item.id == execution_id), None)
        if current is not None and current.success is not None:
            return current.success
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MaterializationError(
                "poll_materialization", name, execution_id, f"wait exceeded {timeout}s"
            )
        time.sleep(min(5, remaining))


def _write_verified(
    store, frame, name, key, event_time, timestamps, description, *, reconcile=False, ledger=None
):
    run_id = frame.run_id.iloc[0]
    execution_id = None
    operation = "load_submission"

    def read():
        print(f"Reading back {name}", flush=True)
        actual = (
            group.select_all()
            .filter(group.run_id == run_id)
            .read(read_options={"arrow_flight_config": {"timeout": 90}})
        )
        if actual.empty:
            return actual
        return actual.assign(**{
            column: pd.to_datetime(actual[column], utc=True)
            for column in timestamps if column in actual
        })

    def missing(actual):
        if actual.empty:
            return frame
        expected = frame.loc[frame[key].isin(actual[key])] if key in actual else frame
        try:
            pd.testing.assert_frame_equal(
                expected.set_index(key).sort_index().sort_index(axis=1),
                actual.set_index(key).sort_index().sort_index(axis=1),
                check_dtype=False,
                rtol=1e-9,
                atol=1e-9,
            )
        except (AssertionError, KeyError):
            raise MaterializationError(
                "readback", name, execution_id, "readback differs from the saved run"
            ) from None
        return frame.loc[~frame[key].isin(actual[key])]

    def submit(start, *args, **kwargs):
        nonlocal operation, execution_id
        operation = "submit_materialization"
        execution = start(*args, **kwargs)
        if execution is not None and type(execution.id) is int and execution.id > 0:
            execution_id = execution.id
            if ledger is not None:
                operation = "checkpoint_execution"
                ledger.record_materialization(run_id, name, execution_id, previous=(None,))
        return execution

    try:
        record = ledger.materialization(run_id, name) if ledger is not None else None
        execution_id = record[0] if record is not None else None
        operation = "feature_group"
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
            raise MaterializationError(operation, name, execution_id, "incompatible keys")

        pending = frame
        if reconcile or record is not None:
            operation = "readback"
            actual = read()
            pending = missing(actual)
            if pending.empty:
                return actual
            if record is not None and execution_id is None:
                raise MaterializationError(
                    operation, name, None, "submission outcome unknown. Refusing another write."
                )

        operation = "lookup_materialization"
        # The SDK creates a new group's job inside insert(). Only existing groups
        # can expose the cached Job instance before insert() starts.
        job = group.materialization_job if type(group.id) is int and group.id > 0 else None
        if execution_id is not None:
            operation = "poll_materialization"
            _wait_for_materialization(job, name, execution_id)
            operation = "readback"
            actual = read()
            pending = missing(actual)
            if pending.empty:
                return actual
        if job is not None:
            operation = "check_active_materialization"
            if any(item.success is None for item in job.get_executions()):
                raise MaterializationError(operation, name, execution_id, "another execution is active")

        if ledger is not None:
            operation = "checkpoint_submission"
            ledger.record_materialization(run_id, name, None, previous=record)
        execution_id = None
        operation = "stage_rows"
        print(f"Publishing {len(pending)} rows to {name}", flush=True)
        try:
            returned_job, _ = insert_with_execution(group, pending, job, submit)
        except Exception as error:
            if execution_id is not None:
                raise
            # A lost response can still leave a fully committed write. Verify
            # content and never retry the mutation when its outcome is unknown.
            try:
                actual = read()
                if missing(actual).empty:
                    detail = MaterializationError(
                        operation, name, None, "submission error recovered by full readback",
                        error=error,
                    )
                    print(detail, flush=True)
                    return actual
            except MaterializationError:
                raise MaterializationError(
                    "reconcile_submission", name, None, "readback differs from the saved run",
                    error=error,
                ) from None
            except Exception:
                pass
            raise MaterializationError(
                operation, name, None, "submission outcome unknown. Refusing another write.",
                error=error,
            ) from None

        if execution_id is not None:
            operation = "poll_materialization"
            if not _wait_for_materialization(job, name, execution_id):
                raise MaterializationError(operation, name, execution_id, "materialization failed")
        operation = "readback"
        deadline = time.monotonic() + 360
        while True:
            actual = read()
            pending = missing(actual)
            if pending.empty:
                return actual
            remaining = deadline - time.monotonic()
            if execution_id is not None or returned_job is None or remaining <= 0:
                raise MaterializationError(operation, name, execution_id, "readback is incomplete")
            time.sleep(min(5, remaining))
    except MaterializationError:
        raise
    except Exception as error:
        raise MaterializationError(
            operation, name, execution_id, "Hopsworks operation failed", error=error,
            retryable=operation in {"readback", "poll_materialization", "lookup_materialization"}
            and _read_retryable(error),
        ) from None
