"""Train and score a CPU LSTM on the project's verified hourly forecast cases."""

import argparse
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from mosamiat.backfill import digest
from mosamiat.collect import City, CollectionError
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
    validation_start: str | None = None,
    test_start: str | None = None,
    *,
    prepared: TrainingData | None = None,
) -> dict:
    try:
        from mosamiat.modeling.lstm import fit
    except ImportError as error:
        raise CollectionError("Install the deep-learning extra to train the LSTM") from error

    if prepared is None:
        reference = load_reference(dataset, city)
        rain = events(read_json(rain_path), reference, city)
        cases, manifest = comparison_cases(
            dataset, reference, city, validation_start, test_start, rain
        )
        rain_sha256 = digest(rain_path)
    else:
        reference, rain = prepared.reference, prepared.rain
        cases, manifest = prepared.cases.copy(), prepared.manifest.copy()
        rain_sha256 = prepared.rain_sha256
    reference.protect_output(dataset, output)
    validation = pd.Timestamp(manifest["validation_start_utc"])
    test = pd.Timestamp(manifest["test_start_utc"])
    # Reserve the public validation fold for model selection in automated training.
    fit_validation = validation
    fit_test = test
    if prepared is not None:
        fit_test = validation
        fit_validation = reference.values.index[reference.values.index.get_loc(validation) * 4 // 5]
    model, history, best_epoch = fit(
        reference.values,
        rain,
        fit_validation,
        fit_test,
        city.timezone,
    )
    issues = cases.index.get_level_values("reference_at_utc").unique()
    wide = model.predict(reference.values, issues)
    cases["lstm"] = aligned_predictions(cases, wide)
    scores = pd.concat(
        {name: score_predictions(cases, cases[name]) for name in ("seasonal_naive", "lstm")},
        names=["model"],
    )
    next_issue = pd.DatetimeIndex([reference.values.index[-1] + pd.Timedelta(hours=1)])
    example = model.predict(reference.values, next_issue)
    manifest.update(
        kind="lstm_evaluation",
        torch_version=version("torch"),
        settings=asdict(model.settings),
        preprocessing=asdict(model.prep),
        best_epoch=best_epoch,
        epochs_run=len(history),
        fitted_until_exclusive_utc=fit_test.isoformat(),
        early_stopping_start_utc=fit_validation.isoformat(),
        lookback_hours=24,
        horizons_hours=list(range(1, 25)),
        rain_label_sha256=rain_sha256,
        rain_event="ERA5 hourly liquid rain > 0 mm",
    )
    return publish_evaluation(
        output,
        manifest,
        {
            "cases.csv": cases,
            "metrics.csv": scores,
            "history.csv": history,
            "forecast_example.csv": example,
        },
        model.save,
        "model.pt",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--rain-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/lstm"))
    parser.add_argument("--city-config", type=Path, default=Path("configs/city.toml"))
    parser.add_argument("--validation-start", help="first validation issue hour with UTC offset")
    parser.add_argument("--test-start", help="first test issue hour with UTC offset")
    args = parser.parse_args(argv)
    try:
        manifest = evaluate(
            args.dataset,
            args.output,
            City.load(args.city_config),
            args.rain_labels,
            args.validation_start,
            args.test_start,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"LSTM evaluation failed: {error}", file=sys.stderr)
        return 1
    print(
        f"Saved {args.output / 'model.pt'} from epoch {manifest['best_epoch']} "
        f"of {manifest['epochs_run']} and scores for 24 horizons."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
