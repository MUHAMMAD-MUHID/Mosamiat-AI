"""Monitor deployed inputs and matured t+24 errors against versioned reference limits."""

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import httpx
import pandas as pd

from mosamiat.agents.ledger import _write_verified
from mosamiat.backfill import backfill, digest
from mosamiat.collect import CollectionError
from mosamiat.features.time_features import encode_wind
from mosamiat.modeling import rain_labels
from mosamiat.modeling.inference import MANIFEST, VERSION, recent_history
from mosamiat.modeling.metrics import metrics
from mosamiat.reference import load_reference
from mosamiat.storage import read_json

PROFILE = Path("configs/drift_islamabad_v1.json")


def archive(city, start, end, root):
    with httpx.Client(timeout=30) as client:
        dataset = backfill(city, root / "backfill", client, start.date(), end.date())
    labels = dataset / "liquid_rain.json"
    rain_labels.collect(dataset, labels, city)
    return dataset, labels


def input_checks(history, limits):
    values = encode_wind(history)
    checks = []
    for field, limit in limits.items():
        s = values[field].dropna()
        score = float(s.mean()) if len(s) == 24 else None
        outside = score is not None and (
            score < limit["lower"] - 1e-9 or score > limit["upper"] + 1e-9
        )
        checks.append(
            dict(
                kind="input",
                metric=field,
                measure="mean_24h",
                count=len(s),
                first=history.index[0].isoformat(),
                last=history.index[-1].isoformat(),
                score=score,
                lower=limit["lower"],
                upper=limit["upper"],
                status="insufficient" if score is None else "drift" if outside else "within_limits",
            )
        )
    return checks


def error_checks(runs, actuals, start, stop, limits, *, active=False):
    """Score at most the most recent 24 consecutive hourly target slots per metric."""
    forecasts = {field: {} for field in limits}
    for run in runs:
        if not active and run["evaluation_sha256"] != MANIFEST:
            continue
        issued = pd.Timestamp(run["issued_at_utc"])
        for row in run["forecasts"]:
            field = row["metric"]
            target = pd.Timestamp(row["target_at_utc"])
            if (
                row["horizon_hours"] == 24
                and start <= target < stop
                and issued < target
                and row["model_name"] == limits[field]["model"]
                and row["model_version"] == limits[field].get("model_version", VERSION)
            ):
                forecasts[field][target] = row["value"]
    checks = []
    for field, values in forecasts.items():
        spec = limits[field]
        last = max(values) if values else stop - pd.Timedelta(hours=1)
        first = last - pd.Timedelta(hours=23)
        pairs = pd.DataFrame({"forecast": pd.Series(values, dtype=float), "actual": actuals[field]})
        pairs = pairs.reindex(pd.date_range(first, last, freq="h")).dropna()
        score = None
        if len(pairs) == 24:
            score = float(
                metrics(field, pairs.actual.to_numpy(), pairs.forecast.to_numpy())[spec["measure"]]
            )
        checks.append(
            dict(
                kind="model_error",
                metric=field,
                measure=spec["measure"],
                count=len(pairs),
                first=first.isoformat(),
                last=last.isoformat(),
                score=score,
                upper=spec["upper"],
                status="insufficient"
                if score is None or spec["upper"] is None
                else "drift"
                if score > spec["upper"] + 1e-9
                else "within_limits",
            )
        )
    return checks


def evaluate(city, payload, ledger, root, policy=None):
    profile = read_json(PROFILE)
    if (
        (policy is None and profile["evaluation_sha256"] != payload["evaluation_sha256"])
        or profile["city_id"] != payload["city_id"]
        or profile["evaluation_sha256"] != MANIFEST
    ):
        raise CollectionError("Drift profile differs from forecast evaluation")
    path = Path(payload.get("snapshot_path", Path("data/collection") / payload["snapshot_id"]))
    source_hash = hashlib.sha256(
        (digest(path / "weather.json") + digest(path / "air_quality.json")).encode()
    ).hexdigest()
    if source_hash != payload["source_sha256"]:
        raise CollectionError("Drift input snapshot differs from forecast input")
    issued = pd.Timestamp(payload["issued_at_utc"])
    history, _ = recent_history(path, city, issued)
    checks = input_checks(history, profile["inputs"])
    stop = issued.normalize() - pd.Timedelta(days=5)
    start = stop - pd.Timedelta(days=7)
    # A t+24 target belongs to the reference hour exactly one day earlier.
    with closing(sqlite3.connect(ledger.path)) as db:
        rows = db.execute(
            """SELECT payload FROM runs WHERE json_extract(payload,'$.city_id')=?
            AND EXISTS (SELECT 1 FROM events WHERE events.run_id=runs.run_id AND status='verified')
            AND json_extract(payload,'$.reference_at_utc') >= ?
            AND json_extract(payload,'$.reference_at_utc') < ?""",
            (
                city.city_id,
                (start - pd.Timedelta(days=1)).isoformat(),
                (stop - pd.Timedelta(days=1)).isoformat(),
            ),
        )
        runs = [json.loads(row[0]) for row in rows]
    actuals = pd.DataFrame(
        columns=profile["errors"], index=pd.DatetimeIndex([], tz="UTC"), dtype=float
    )
    actual_hash = ""
    if runs:
        dataset, labels = archive(city, start, stop - pd.Timedelta(days=1), root)
        ref = load_reference(dataset, city)
        actuals = ref.values.assign(
            rain_probability=rain_labels.events(read_json(labels), ref, city)
        )
        actuals["wind_direction_deg"] = actuals.wind_direction_deg.where(actuals.wind_speed_mps > 0)
        actual_hash = digest(dataset / "manifest.json") + ":" + digest(labels)
    limits = profile["errors"] | (policy["error_limits"] if policy else {})
    checks.extend(error_checks(runs, actuals, start, stop, limits, active=policy is not None))
    return dict(
        run_id=payload["run_id"],
        city_id=city.city_id,
        issued_at_utc=issued.isoformat(),
        profile_sha256=digest(PROFILE),
        evaluation_sha256=payload["evaluation_sha256"],
        source_sha256=source_hash,
        archive_sha256=actual_hash,
        checks=checks,
        drift_count=sum(c["status"] == "drift" for c in checks),
    )


def publish_drift(store, decision):
    row = dict(decision, checks=json.dumps(decision["checks"], allow_nan=False))
    frame = pd.DataFrame([row])
    frame["issued_at_utc"] = pd.to_datetime(frame.issued_at_utc, utc=True)
    _write_verified(
        store,
        frame,
        "mosamiat_drift_log",
        "run_id",
        "issued_at_utc",
        ["issued_at_utc"],
        "Hourly input and matured forecast-error checks and candidate retraining decisions.",
    )
