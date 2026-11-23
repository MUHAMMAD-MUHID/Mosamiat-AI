"""Compare saved forecasts on identical issue times, targets and verification labels."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from mosamiat.backfill import digest
from mosamiat.collect import CollectionError
from mosamiat.modeling.baseline import CASE_KEY, FORECAST_METRICS
from mosamiat.modeling.evaluate import publish_evaluation
from mosamiat.modeling.metrics import score_predictions
from mosamiat.storage import read_json

RUN_FIELDS = (
    "city_id",
    "train_start_utc",
    "validation_start_utc",
    "test_start_utc",
    "end_exclusive_utc",
    "provenance",
)
SHARED_COLUMNS = ["source_at_utc", "actual", "seasonal_naive", "eligible"]


def compare(classical: Path, statistical: Path, lstm: Path, output: Path) -> dict:
    scores = {}
    inputs = {}
    for model, directory in (
        ("random_forest", classical),
        ("sarimax", statistical),
        ("lstm", lstm),
    ):
        manifest = read_json(directory / "manifest.json")
        if manifest["status"] != "complete" or manifest["kind"] != f"{model}_evaluation":
            raise CollectionError(f"Expected a completed {model} evaluation in {directory}")
        path = directory / "cases.csv"
        inputs[model] = digest(path)
        if inputs[model] != manifest["sha256"]["cases.csv"]:
            raise CollectionError(f"Evaluation cases have changed in {directory}")
        cases = pd.read_csv(path, index_col=CASE_KEY).sort_index()
        run = {key: manifest[key] for key in RUN_FIELDS}
        supported = ("humidity_pct", "temperature_c") if model == "sarimax" else FORECAST_METRICS
        if set(cases.index.get_level_values("metric")) != set(supported):
            raise CollectionError(f"Unexpected forecast fields for {model}")
        if model == "random_forest":
            reference_run = run
            shared = cases[SHARED_COLUMNS]
            scores["seasonal_naive"] = score_predictions(cases, cases.seasonal_naive)
        else:
            expected = shared[shared.index.get_level_values("metric").isin(supported)]
            if run != reference_run or not expected.equals(cases[SHARED_COLUMNS]):
                raise CollectionError(
                    f"{model} uses different source data, splits or forecast cases"
                )
        table = score_predictions(cases, cases[model])
        scores[model] = table[table.index.get_level_values("metric").isin(supported)]
    return publish_evaluation(
        output,
        dict(reference_run, kind="model_comparison", status="building", input_cases_sha256=inputs),
        {"metrics.csv": pd.concat(scores, names=["model"])},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("classical", "statistical", "lstm", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        compare(args.classical, args.statistical, args.lstm, args.output)
    except (OSError, KeyError, ValueError) as error:
        print(f"Model comparison failed: {error}", file=sys.stderr)
        return 1
    print(f"Saved validation and test scores to {args.output / 'metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
