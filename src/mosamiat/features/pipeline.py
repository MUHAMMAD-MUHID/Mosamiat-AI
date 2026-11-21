"""Build versioned local research features from integrity-checked historical snapshots."""

import argparse
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from mosamiat.backfill import METRICS, digest, write_json
from mosamiat.collect import City, CollectionError
from mosamiat.features.lag_features import lag_features
from mosamiat.features.target_builder import build_targets
from mosamiat.features.time_features import calendar_features, encode_wind
from mosamiat.reference import load_reference


def load_definitions(path: Path) -> dict:
    with path.open("rb") as stream:
        definitions = tomllib.load(stream)
    if definitions.get("schema_version") != 1:
        raise CollectionError("Unsupported feature definitions")
    for key in ("lags_hours", "rolling_hours", "horizons_hours"):
        values = definitions.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(type(value) is not int or value < 1 for value in values)
            or values != sorted(set(values))
        ):
            raise CollectionError(f"{key} must contain sorted unique positive integers")
    if definitions["horizons_hours"] != list(range(1, 25)):
        raise CollectionError("Targets must cover every horizon from 1 to 24")
    return definitions


def validate_hourly(values: pd.DataFrame) -> None:
    index = values.index
    if (
        values.empty
        or not isinstance(index, pd.DatetimeIndex)
        or index.tz is None
        or str(index.tz) != "UTC"
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
        or not index.equals(pd.date_range(index[0], index[-1], freq="h"))
        or not (index == index.floor("h")).all()
        or set(values.columns) != set(METRICS)
        or np.isinf(values.to_numpy(dtype=float)).any()
    ):
        raise CollectionError("Expected unique continuous hourly values for all source metrics")


def transform(values: pd.DataFrame, definitions: dict, timezone: str) -> pd.DataFrame:
    """Calendar at t plus values strictly before t; never drop inference rows."""
    validate_hourly(values)
    return pd.concat(
        [
            calendar_features(values.index, timezone),
            lag_features(
                encode_wind(values), definitions["lags_hours"], definitions["rolling_hours"]
            ),
        ],
        axis=1,
    )


def build(dataset: Path, output: Path, city: City, config: Path) -> Path:
    definitions = load_definitions(config)
    reference = load_reference(dataset, city)
    reference.protect_output(dataset, output)
    values = reference.values
    if not values.index.equals(reference.expected_index):
        raise CollectionError("Dataset coverage differs from the declared range")
    features = transform(values, definitions, city.timezone)
    features.index.name = "reference_at_utc"
    targets = build_targets(values, definitions["horizons_hours"])
    features.insert(0, "city_id", city.city_id)
    targets.insert(0, "city_id", city.city_id)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "status": "building",
        "is_ground_truth": False,
        "operational_point_in_time_safe": False,
        "purpose": "retrospective_provider_reference_research",
        "city_id": city.city_id,
        "timezone": city.timezone,
        "definitions": definitions,
        "source_snapshots": [path.name for path in reference.paths],
        "source_manifest_sha256": digest(dataset / "manifest.json"),
        "units": METRICS,
        "attribution": "Open-Meteo, ECMWF ERA5, Copernicus CAMS",
    }
    write_json(output / "manifest.json", manifest)
    for name, table, index in (
        ("features.csv", features, True),
        ("targets.csv", targets, False),
    ):
        temporary = output / f"{name}.tmp"
        table.to_csv(temporary, index=index)
        temporary.replace(output / name)
    manifest.update(
        status="complete",
        feature_rows=len(features),
        target_rows=len(targets),
        source_missing={metric: int(values[metric].isna().sum()) for metric in METRICS},
        target_missing={
            metric: int(targets[f"target_{metric}_missing"].sum()) for metric in METRICS
        },
        sha256={name: digest(output / name) for name in ("features.csv", "targets.csv")},
    )
    write_json(output / "manifest.json", manifest)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/features"))
    parser.add_argument("--city-config", type=Path, default=Path("configs/city.toml"))
    parser.add_argument("--config", type=Path, default=Path("configs/features.toml"))
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() == args.dataset.resolve():
            raise CollectionError("Feature output must differ from its source dataset")
        destination = build(args.dataset, args.output, City.load(args.city_config), args.config)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Feature build failed: {error}", file=sys.stderr)
        return 1
    report = json.loads((destination / "manifest.json").read_text())
    print(f"Saved {report['feature_rows']} feature rows and {report['target_rows']} target rows")
    print("REFERENCE DATA ONLY: retrospective retrieval; not operational training evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
