"""Register a versioned forecast candidate and verify its predictions after download."""

import argparse
import shutil
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from mosamiat.backfill import digest
from mosamiat.collect import CollectionError
from mosamiat.features.pipeline import transform
from mosamiat.hopsworks_compat import save_registry_model
from mosamiat.modeling import statistical
from mosamiat.modeling.baseline import FORECAST_METRICS
from mosamiat.modeling.forest import HORIZONS, ForecastForest
from mosamiat.storage import connect_project, read_json

ARTIFACT = {"random_forest": "model.joblib", "sarimax": None, "lstm": "model.pt"}


def verified_candidate(directory, family, manifest_hash):
    if digest(directory / "manifest.json") != manifest_hash:
        raise CollectionError("Registry version contains a different evaluation manifest")
    manifest = read_json(directory / "manifest.json")
    if manifest["kind"] != f"{family}_evaluation" or manifest["status"] != "complete":
        raise CollectionError(f"Expected a completed {family} evaluation")
    files = ["metrics.csv"]
    if ARTIFACT[family]:
        files.append(ARTIFACT[family])
    if any(digest(directory / name) != manifest["sha256"][name] for name in files):
        raise CollectionError(f"{family} evaluation checksum mismatch")
    if family == "random_forest":
        fitted = verified_forest(directory, manifest_hash)
    elif family == "sarimax":
        if manifest["statsmodels_version"] != version("statsmodels"):
            raise CollectionError("Install the statsmodels version recorded in the evaluation")
        fitted = {metric: manifest["models"][metric]["params"] for metric in statistical.METRICS}
    else:
        from mosamiat.modeling.lstm import Forecaster

        if manifest["torch_version"] != version("torch"):
            raise CollectionError("Install the PyTorch version recorded in the evaluation")
        fitted = Forecaster.load(directory / "model.pt")
        if (
            asdict(fitted.prep) != manifest["preprocessing"]
            or asdict(fitted.settings) != manifest["settings"]
        ):
            raise CollectionError(
                "LSTM preprocessing or settings differ from the evaluation manifest"
            )
    return fitted, manifest, pd.read_csv(directory / "metrics.csv")


def load_candidate(registry, selection):
    entry = registry.get_model(selection["name"], version=selection["version"])
    if entry is None:
        raise CollectionError("Selected model version is missing from the registry")
    with TemporaryDirectory() as local:
        directory = Path(entry.download(local_path=local))
        return verified_candidate(directory, selection["family"], selection["manifest_sha256"])


def candidate_predictions(family, bundle, history, issue, city):
    return candidate_batch(family, bundle, history, pd.DatetimeIndex([issue]), city).xs(
        issue, level="reference_at_utc"
    )


def candidate_batch(family, bundle, history, issues, city):
    fitted, manifest, _ = bundle
    if (
        manifest["city_id"] != city.city_id
        or manifest.get("timezone", city.timezone) != city.timezone
    ):
        raise CollectionError("Registered model is for a different city")
    if family == "random_forest":
        values = history.reindex(history.index.union(issues))
        features = transform(values, manifest["feature_definitions"], city.timezone)
        return fitted.predict(features.loc[issues])
    if family == "lstm":
        return fitted.predict(history, issues)
    index = pd.MultiIndex.from_product(
        [issues, range(1, HORIZONS + 1)], names=["reference_at_utc", "horizon_hours"]
    )
    targets = index.get_level_values(0) + pd.to_timedelta(index.get_level_values(1), unit="h")
    keys = pd.MultiIndex.from_arrays(
        [index.get_level_values(0), targets], names=["reference_at_utc", "target_at_utc"]
    )
    return pd.DataFrame(
        {
            "target_at_utc": targets,
            **{
                metric: statistical.predict(history[metric], keys, params).to_numpy()
                for metric, params in fitted.items()
            },
        },
        index=index,
    )


def register_candidates(registry, city, evaluation, model_version):
    run = read_json(evaluation / "manifest.json")
    issue = pd.Timestamp(run["probe_issue_utc"])
    history = pd.read_csv(
        evaluation / "inference_history.csv",
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )
    history = history.loc[history.index < issue].tail(24)
    selections, failures = {}, {}
    for family, artifact in ARTIFACT.items():
        if run["families"][family]["status"] != "complete":
            print(f"Skipped registry candidate {family}: training failed")
            continue
        try:
            selections[family] = register_candidate(
                registry,
                city,
                evaluation / family,
                issue,
                history,
                family,
                artifact,
                model_version,
                run["families"][family]["manifest_sha256"],
            )
        except Exception as error:
            failures[family] = type(error).__name__
            print(f"Registry candidate {family} v{model_version} failed ({type(error).__name__})")
    return selections, failures


def register_candidate(
    registry, city, path, issue, history, family, artifact, model_version, manifest_hash
):
    selection = {
        "family": family,
        "name": f"mosamiat_{city.city_id}_{family}",
        "version": model_version,
        "manifest_sha256": manifest_hash,
    }
    local = verified_candidate(path, family, selection["manifest_sha256"])
    expected = candidate_predictions(family, local, history, issue, city)
    targets = pd.date_range(issue + pd.Timedelta(hours=1), periods=HORIZONS, freq="h")
    fields = statistical.METRICS if family == "sarimax" else FORECAST_METRICS
    if (
        not targets.equals(pd.DatetimeIndex(expected.target_at_utc))
        or set(expected.columns) != {"target_at_utc", *fields}
        or not np.isfinite(expected[list(fields)].to_numpy()).all()
    ):
        raise CollectionError(f"Expected 24 hours of finite {family} predictions")
    if not any(m.version == model_version for m in registry.get_models(selection["name"])):
        scores = local[2]
        scores = scores[scores.split.eq("validation") & scores.model.eq(family)]
        summary = {}
        for row in scores[scores.horizon_hours.eq(24)].itertuples():
            measure = {
                "rain_probability": "brier",
                "wind_direction_deg": "circular_mae_deg",
            }.get(row.metric, "rmse")
            value = getattr(row, measure)
            if np.isfinite(value):
                summary[f"{row.metric}_{measure}_24h"] = float(value)
        with TemporaryDirectory() as folder:
            for filename in filter(None, ("manifest.json", "metrics.csv", artifact)):
                shutil.copyfile(path / filename, Path(folder) / filename)
            entry = registry.python.create_model(
                selection["name"],
                version=model_version,
                metrics=summary,
                description=f"Validated {family} candidate for {city.name} hourly forecasts.",
            )
            save_registry_model(entry, folder)
    loaded = load_candidate(registry, selection)
    actual = candidate_predictions(family, loaded, history, issue, city)
    if not expected.equals(actual):
        raise CollectionError(f"Reloaded {family} forecast differs from the fitted candidate")
    print(f"Verified registry candidate {selection['name']} v{model_version}")
    return selection


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
