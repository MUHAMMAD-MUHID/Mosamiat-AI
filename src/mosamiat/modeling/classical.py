"""Train direct Random Forest forecasts on the historical reference benchmark."""

import argparse
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from mosamiat.backfill import digest
from mosamiat.collect import City, CollectionError
from mosamiat.features.pipeline import load_definitions, transform
from mosamiat.modeling.evaluate import (
    TrainingData,
    aligned_predictions,
    comparison_cases,
    publish_evaluation,
)
from mosamiat.modeling.metrics import score_predictions
from mosamiat.modeling.rain_labels import events
from mosamiat.reference import load_reference
from mosamiat.storage import read_json


def evaluate(
    dataset: Path,
    output: Path,
    city: City,
    rain_path: Path,
    config: Path,
    validation_start: str | None = None,
    test_start: str | None = None,
    *,
    prepared: TrainingData | None = None,
) -> dict:
    try:
        import joblib

        from mosamiat.modeling.forest import PARAMETERS, fit
    except ImportError as error:
        raise CollectionError("Install the modeling extra to fit Random Forests") from error

    definitions = load_definitions(config)
    if prepared is None:
        reference = load_reference(dataset, city)
        rain = events(read_json(rain_path), reference, city)
        cases, manifest = comparison_cases(
            dataset, reference, city, validation_start, test_start, rain
        )
        features = transform(reference.values, definitions, city.timezone)
        rain_sha256 = digest(rain_path)
    else:
        reference, rain, features = prepared.reference, prepared.rain, prepared.features
        cases, manifest = prepared.cases.copy(), prepared.manifest.copy()
        rain_sha256 = prepared.rain_sha256
    reference.protect_output(dataset, output)
    validation = pd.Timestamp(manifest["validation_start_utc"])
    warmup = max(2, max(definitions["lags_hours"]), max(definitions["rolling_hours"]))
    forest = fit(features, reference.values, rain, validation, warmup)

    # Predict once per distinct issue, then align all models on the benchmark keys.
    issues = cases.index.get_level_values("reference_at_utc").unique()
    wide = forest.predict(features.loc[issues])
    cases["random_forest"] = aligned_predictions(cases, wide)
    scores = pd.concat(
        {
            name: score_predictions(cases, cases[name])
            for name in ("seasonal_naive", "random_forest")
        },
        names=["model"],
    )
    rain_cases = cases.xs("rain_probability", level="metric").loc[lambda rows: rows.eligible].copy()
    rain_cases["probability_bin"] = pd.cut(
        rain_cases.random_forest, np.linspace(0, 1, 11), include_lowest=True
    )
    reliability = rain_cases.groupby(
        ["split", "horizon_hours", "probability_bin"], observed=True
    ).agg(
        cases=("actual", "size"),
        mean_probability=("random_forest", "mean"),
        event_rate=("actual", "mean"),
    )
    prevalence = float(rain.loc[rain.index < validation].mean())
    rain_cases["climatology_squared_error"] = (rain_cases.actual - prevalence) ** 2
    climatology = (
        rain_cases.groupby(["split", "horizon_hours"])
        .climatology_squared_error.mean()
        .to_frame("brier")
    )
    manifest.update(
        kind="random_forest_evaluation",
        feature_definitions=definitions,
        timezone=city.timezone,
        features=forest.columns,
        parameters=PARAMETERS,
        sklearn_version=version("scikit-learn"),
        training_issue_counts=forest.training_counts,
        warmup_hours=warmup,
        rain_label_sha256=rain_sha256,
        rain_event="ERA5 hourly liquid rain > 0 mm",
        rain_training_event_rate=prevalence,
    )
    return publish_evaluation(
        output,
        manifest,
        {
            "metrics.csv": scores,
            "cases.csv": cases,
            "rain_reliability.csv": reliability,
            "rain_climatology.csv": climatology,
        },
        save_model=lambda path: joblib.dump(forest, path, compress=3),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--rain-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/classical"))
    parser.add_argument("--city-config", type=Path, default=Path("configs/city.toml"))
    parser.add_argument("--config", type=Path, default=Path("configs/features.toml"))
    parser.add_argument("--validation-start", help="first validation hour with UTC offset")
    parser.add_argument("--test-start", help="first test hour with UTC offset")
    args = parser.parse_args(argv)
    try:
        evaluate(
            args.dataset,
            args.output,
            City.load(args.city_config),
            args.rain_labels,
            args.config,
            args.validation_start,
            args.test_start,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Classical evaluation failed: {error}", file=sys.stderr)
        return 1
    print(f"Saved fitted Random Forests and per-horizon comparisons to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
