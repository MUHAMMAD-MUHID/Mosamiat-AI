"""Evaluate the seasonal naive baseline on retrospective hourly reference data."""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from mosamiat.backfill import METRICS, digest, write_json
from mosamiat.collect import City, CollectionError
from mosamiat.features.pipeline import validate_hourly
from mosamiat.features.target_builder import build_targets
from mosamiat.modeling.baseline import FORECAST_METRICS, baseline_cases
from mosamiat.modeling.metrics import score_predictions
from mosamiat.modeling.splits import split_targets
from mosamiat.reference import load_reference
from mosamiat.storage import read_json


def boundary(value: str | None, index: pd.DatetimeIndex, percent: int) -> pd.Timestamp:
    if value is None:
        return index[len(index) * percent // 100]
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None or stamp not in index:
        raise CollectionError("Split boundary must be a timezone-aware hour in the dataset")
    return stamp.tz_convert("UTC")


def evaluate(
    dataset: Path,
    output: Path,
    city: City,
    validation_start: str | None = None,
    test_start: str | None = None,
) -> dict:
    reference = load_reference(dataset, city)
    reference.protect_output(dataset, output)
    values = reference.values
    validate_hourly(values)
    if not values.index.equals(reference.expected_index):
        raise CollectionError("Dataset coverage differs from the declared range")
    validation = boundary(validation_start, values.index, 60)
    test = boundary(test_start, values.index, 80)
    end = values.index[-1] + pd.Timedelta(hours=1)
    targets = build_targets(values[list(FORECAST_METRICS)], list(range(1, 25)))
    folds = split_targets(targets, validation, test, end)
    cases = baseline_cases(values, folds)
    scores = score_predictions(cases, cases.seasonal_naive)
    manifest = {
        "kind": "seasonal_naive_evaluation",
        "schema_version": 1,
        "status": "building",
        "city_id": city.city_id,
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "train_start_utc": values.index[0].isoformat(),
        "validation_start_utc": validation.isoformat(),
        "test_start_utc": test.isoformat(),
        "end_exclusive_utc": end.isoformat(),
        "fold_pairs": {name: len(frame) for name, frame in folds.items()},
        "scored_metric_cases": int(scores.scored.sum()),
        "units": {metric: METRICS[metric] for metric in FORECAST_METRICS},
        "provenance": reference.provenance(dataset),
        "is_ground_truth": False,
        "operational_point_in_time_safe": False,
        "attribution": "Open-Meteo, ECMWF ERA5, Copernicus CAMS",
    }
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".lock"
    try:
        stream = lock.open("x")
    except FileExistsError as error:
        raise CollectionError("Evaluation output is locked by another writer") from error
    try:
        with stream:
            manifest_path = output / "manifest.json"
            if any(path != lock for path in output.iterdir()) and (
                not manifest_path.exists()
                or read_json(manifest_path).get("kind") != manifest["kind"]
            ):
                raise CollectionError(
                    "Choose an empty output directory or a prior baseline evaluation"
                )
            write_json(manifest_path, manifest)
            for name, table in (("cases.csv", cases), ("metrics.csv", scores)):
                table.to_csv(output / name)
            manifest.update(
                status="complete",
                sha256={name: digest(output / name) for name in ("cases.csv", "metrics.csv")},
            )
            write_json(manifest_path, manifest)
    finally:
        lock.unlink()
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--city-config", type=Path, default=Path("configs/city.toml"))
    parser.add_argument(
        "--validation-start", help="first validation issue hour, including UTC offset"
    )
    parser.add_argument("--test-start", help="first test issue hour, including UTC offset")
    args = parser.parse_args(argv)
    try:
        result = evaluate(
            args.dataset,
            args.output,
            City.load(args.city_config),
            args.validation_start,
            args.test_start,
        )
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
    print(
        f"Scored {result['scored_metric_cases']} retrospective metric cases. "
        f"See {args.output / 'metrics.csv'} for coverage by split and horizon."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
