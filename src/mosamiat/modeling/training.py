"""Train all forecast families on one verified Feature Store snapshot."""

import hashlib
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from mosamiat.agents.drift import archive
from mosamiat.agents.ledger import _write_verified
from mosamiat.backfill import digest, write_json
from mosamiat.collect import CollectionError
from mosamiat.features.pipeline import load_definitions, transform, validate_hourly
from mosamiat.modeling import classical, deep_learning, statistical
from mosamiat.modeling.compare import compare
from mosamiat.modeling.evaluate import TrainingData, comparison_cases
from mosamiat.modeling.rain_labels import events
from mosamiat.reference import load_reference
from mosamiat.storage import read_json

CONFIG = Path("configs/features.toml")
TRAINERS = {
    "random_forest": classical.evaluate,
    "sarimax": statistical.evaluate,
    "lstm": deep_learning.evaluate,
}
ARTIFACTS = (
    "manifest.json",
    "inference_history.csv",
    "random_forest/manifest.json",
    "random_forest/model.joblib",
    "random_forest/metrics.csv",
    "random_forest/cases.csv",
    "random_forest/rain_reliability.csv",
    "random_forest/rain_climatology.csv",
    "sarimax/manifest.json",
    "sarimax/metrics.csv",
    "sarimax/cases.csv",
    "lstm/manifest.json",
    "lstm/model.pt",
    "lstm/metrics.csv",
    "lstm/cases.csv",
    "lstm/history.csv",
    "lstm/forecast_example.csv",
    "comparison/manifest.json",
    "comparison/metrics.csv",
)


def checkpoint_members(statuses):
    complete = all(status == "complete" for status in statuses.values())
    return {
        name
        for name in ARTIFACTS
        if name == "inference_history.csv"
        or statuses.get(name.split("/")[0]) == "complete"
        or (complete and name.startswith("comparison/"))
    }


def verify_training(output):
    path = output / "manifest.json"
    manifest = read_json(path)
    statuses = {family: result["status"] for family, result in manifest["families"].items()}
    if set(statuses) != set(TRAINERS) or any(
        status not in {"complete", "failed"} for status in statuses.values()
    ):
        raise CollectionError("Unexpected training family statuses")
    complete = all(status == "complete" for status in statuses.values())
    expected = checkpoint_members(statuses)
    if (
        manifest["status"] != ("complete" if complete else "incomplete")
        or set(manifest["sha256"]) != expected
        or any(digest(output / name) != hash_ for name, hash_ in manifest["sha256"].items())
    ):
        raise CollectionError("Training checkpoint failed integrity verification")
    return digest(path)


def historical_inputs(store, city, day, work):
    end = day.normalize() - pd.Timedelta(days=6)
    dataset, rain_path = archive(city, end - pd.Timedelta(days=60), end, work)
    reference = load_reference(dataset, city)
    values = reference.values
    validate_hourly(values)
    if not values.index.equals(reference.expected_index) or values.index[-1] >= day:
        raise CollectionError("Training archive must contain only complete, matured hours")
    definitions = load_definitions(CONFIG)
    features = transform(values, definitions, city.timezone)
    rain = events(read_json(rain_path), reference, city)
    identity = hashlib.sha256(
        (
            city.city_id + digest(dataset / "manifest.json") + digest(rain_path) + digest(CONFIG)
        ).encode()
    ).hexdigest()
    # Store each matured target label at its valid hour. Horizon joins and split
    # purging use these timestamps after readback, independently of upload time.
    frame = pd.concat([values.add_prefix("label_"), features.add_prefix("feature_")], axis=1)
    frame = frame.assign(label_rain_event=rain, city_id=city.city_id, run_id=identity)
    frame.index.name = "valid_at_utc"
    frame = frame.reset_index()
    frame["training_row_id"] = identity + "_" + frame.valid_at_utc.dt.strftime("%Y%m%d%H")
    actual = (
        _write_verified(
            store,
            frame,
            "mosamiat_training_history",
            "training_row_id",
            "valid_at_utc",
            ["valid_at_utc"],
            "Historical point-ordered features and matured provider-reference labels.",
        )
        .set_index("valid_at_utc")
        .sort_index()
    )
    actual.index = actual.index.as_unit(values.index.unit)
    values = actual[["label_" + col for col in values]].set_axis(values.columns, axis=1)
    features = actual[["feature_" + col for col in features]].set_axis(features.columns, axis=1)
    rain = actual["label_rain_event"]
    reference = replace(reference, values=values)
    cases, manifest = comparison_cases(dataset, reference, city, None, None, rain)
    manifest["provenance"]["feature_store_snapshot"] = identity
    return (
        dataset,
        rain_path,
        TrainingData(reference, features, rain, cases, manifest, digest(rain_path)),
    )


def train(store, city, day, work, output):
    dataset, labels, data = historical_inputs(store, city, day, work)
    output.mkdir(parents=True)
    results = {}
    for family, evaluator in TRAINERS.items():
        start = time.monotonic()
        extra = {
            "random_forest": (labels, CONFIG),
            "sarimax": (),
            "lstm": (labels,),
        }[family]
        try:
            manifest = evaluator(dataset, output / family, city, *extra, prepared=data)
        except CollectionError as error:
            results[family] = {"status": "failed", "reason": str(error)}
        else:
            manifest["training_duration_seconds"] = round(time.monotonic() - start, 3)
            write_json(output / family / "manifest.json", manifest)
            results[family] = {
                "manifest_sha256": digest(output / family / "manifest.json"),
                "status": manifest["status"],
            }
        results[family]["duration_seconds"] = round(time.monotonic() - start, 3)
        print(
            f"{family}: {results[family]['status']} in {results[family]['duration_seconds']:.3f}s"
        )
    complete = all(record["status"] == "complete" for record in results.values())
    if complete:
        compare(*(output / family for family in TRAINERS), output / "comparison")
    # Retain the readback snapshot for reproducible validation and inference checks.
    issue = pd.Timestamp(data.manifest["validation_start_utc"])
    data.reference.values.to_csv(output / "inference_history.csv")
    statuses = {family: result["status"] for family, result in results.items()}
    members = {name: digest(output / name) for name in sorted(checkpoint_members(statuses))}
    write_json(
        output / "manifest.json",
        {
            "status": "complete" if complete else "incomplete",
            "families": results,
            "sha256": members,
            "timezone": city.timezone,
            "probe_issue_utc": issue.isoformat(),
        },
    )
    return verify_training(output)
