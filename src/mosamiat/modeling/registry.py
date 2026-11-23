"""Register a versioned forecast candidate and verify its predictions after download."""

import argparse
import shutil
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from mosamiat.backfill import digest
from mosamiat.collect import CollectionError
from mosamiat.modeling.baseline import FORECAST_METRICS
from mosamiat.modeling.forest import HORIZONS, ForecastForest
from mosamiat.storage import connect_project, read_json


def verified_forest(directory: Path, manifest_hash: str) -> ForecastForest:
    """Load only the model bytes identified by the caller's trusted evaluation."""
    if digest(directory / "manifest.json") != manifest_hash:
        raise CollectionError("Registry version contains a different evaluation manifest")
    manifest = read_json(directory / "manifest.json")
    if manifest["kind"] != "random_forest_evaluation" or manifest["status"] != "complete":
        raise CollectionError("Expected a completed Random Forest evaluation")
    if manifest["sklearn_version"] != version("scikit-learn"):
        raise CollectionError("Install the scikit-learn version recorded in the evaluation")
    for name in ("model.joblib", "metrics.csv"):
        if digest(directory / name) != manifest["sha256"][name]:
            raise CollectionError(f"Evaluation checksum mismatch for {name}")
    forest = joblib.load(directory / "model.joblib")
    if (
        not isinstance(forest, ForecastForest)
        or forest.columns != manifest["features"]
        or set(forest.estimators) != set(FORECAST_METRICS)
    ):
        raise CollectionError("Model input or output fields differ from the evaluation schema")
    return forest


def load_forecast_bundle(registry, name: str, model_version: int, manifest_hash: str):
    entry = registry.get_model(name, version=model_version)
    if entry is None:
        raise CollectionError("Requested model version is missing from the registry")
    with TemporaryDirectory() as local:
        downloaded = Path(entry.download(local_path=local))
        forest = verified_forest(downloaded, manifest_hash)
        return (
            forest,
            read_json(downloaded / "manifest.json"),
            pd.read_csv(downloaded / "metrics.csv"),
        )


def load_version(registry, name: str, model_version: int, manifest_hash: str) -> ForecastForest:
    forest, _, _ = load_forecast_bundle(registry, name, model_version, manifest_hash)
    return forest


def register(
    evaluation: Path, features: Path, model_version: int, registry_factory
) -> tuple[str, str]:
    if model_version < 1:
        raise CollectionError("Model version must be a positive integer")
    manifest_hash = digest(evaluation / "manifest.json")
    with TemporaryDirectory() as local:
        bundle = Path(local)
        for filename in ("manifest.json", "model.joblib", "metrics.csv"):
            shutil.copyfile(evaluation / filename, bundle / filename)
        forest = verified_forest(bundle, manifest_hash)
        manifest = read_json(bundle / "manifest.json")
        feature_manifest = read_json(features / "manifest.json")
        if (
            feature_manifest["status"] != "complete"
            or feature_manifest["city_id"] != manifest["city_id"]
            or feature_manifest["timezone"] != manifest["timezone"]
            or feature_manifest["definitions"] != manifest["feature_definitions"]
            or feature_manifest["source_manifest_sha256"]
            != manifest["provenance"]["source_manifest_sha256"]
            or digest(features / "features.csv") != feature_manifest["sha256"]["features.csv"]
        ):
            raise CollectionError("Verification features differ from the evaluated dataset")
        inputs = pd.read_csv(
            features / "features.csv", index_col="reference_at_utc", parse_dates=True
        )
        issue = pd.Timestamp(manifest["validation_start_utc"])
        sample = inputs.loc[[issue], forest.columns]
        expected = forest.predict(sample)
        targets = pd.date_range(issue + pd.Timedelta(hours=1), periods=HORIZONS, freq="h")
        if (
            not targets.equals(pd.DatetimeIndex(expected.target_at_utc))
            or not np.isfinite(expected[list(FORECAST_METRICS)].to_numpy()).all()
        ):
            raise CollectionError("Expected a complete, finite 24-hour forecast")

        scores = pd.read_csv(bundle / "metrics.csv")
        aqi = scores.query(
            "split == 'validation' and metric == 'aqi_value' and horizon_hours == 24"
        )
        metrics = {
            "aqi_rmse_24h": float(aqi.loc[aqi.model.eq("random_forest"), "rmse"].item()),
            "baseline_aqi_rmse_24h": float(aqi.loc[aqi.model.eq("seasonal_naive"), "rmse"].item()),
        }
        if not np.isfinite(list(metrics.values())).all():
            raise CollectionError("Missing 24-hour AQI validation scores")
        name = f"mosamiat_{manifest['city_id']}_random_forest"
        registry = registry_factory()
        if not any(entry.version == model_version for entry in registry.get_models(name)):
            entry = registry.python.create_model(
                name,
                version=model_version,
                metrics=metrics,
                description="Candidate for hourly weather and AQI forecasts at horizons 1–24.",
            )
            entry.save(str(bundle), keep_original_files=True)
    restored = load_version(registry, name, model_version, manifest_hash)
    if not expected.equals(restored.predict(sample)):
        raise CollectionError("Reloaded forecast differs from the local model")
    return name, manifest_hash


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument(
        "--features", type=Path, required=True, help="Feature pipeline output directory"
    )
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        name, checksum = register(
            args.evaluation,
            args.features,
            args.version,
            lambda: connect_project().get_model_registry(),
        )
    except Exception as error:
        # SDK exceptions can include credentials in request details.
        message = str(error) if isinstance(error, CollectionError) else type(error).__name__
        print(f"Registry verification failed: {message}", file=sys.stderr)
        return 1
    print(f"Verified {name} v{args.version}: 24 hours, 8 fields, manifest SHA-256 {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
