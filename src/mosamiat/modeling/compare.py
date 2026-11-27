"""Compare saved forecasts on identical issue times, targets and verification labels."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mosamiat.backfill import digest
from mosamiat.collect import CollectionError
from mosamiat.modeling.baseline import CASE_KEY, FORECAST_METRICS
from mosamiat.modeling.evaluate import aligned_predictions, publish_evaluation
from mosamiat.modeling.inference import MANIFEST, MODEL, VERSION
from mosamiat.modeling.metrics import metrics, score_predictions
from mosamiat.modeling.registry import candidate_batch, load_candidate
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
MIN_PAIRED = 48
MIN_COVERAGE = 0.95
MIN_IMPROVEMENT = 0.01
LOSS = {"rain_probability": "brier", "wind_direction_deg": "circular_mae_deg"}


def validation_winner(metric, cases, families):
    """Score every challenger and the incumbent on the same validation rows."""
    fields = ["actual", "seasonal_naive", "incumbent", *families]
    eligible = cases.loc[cases.eligible, fields]
    paired = eligible.replace([np.inf, -np.inf], np.nan).dropna()
    result = {"eligible": len(eligible), "paired": len(paired), "winner": None}
    if len(paired) < MIN_PAIRED or len(paired) < MIN_COVERAGE * len(eligible):
        return result
    if metric == "rain_probability" and min((paired.actual == n).sum() for n in (0, 1)) < 5:
        return result
    measure = LOSS.get(metric, "rmse")
    errors = {
        name: float(metrics(metric, paired.actual.to_numpy(), paired[name].to_numpy())[measure])
        for name in fields[1:]
    }
    best = min(families, key=errors.__getitem__)
    limit = min(errors["incumbent"], errors["seasonal_naive"])
    if errors[best] < limit and errors[best] <= limit * (1 - MIN_IMPROVEMENT):
        result["winner"] = best
    return dict(result, measure=measure, errors=errors)


def promotion(registry, city, evaluation, candidates, active):
    cases = {}
    for family in candidates:
        path = evaluation / family / "cases.csv"
        data = pd.read_csv(path, parse_dates=["reference_at_utc", "target_at_utc"])
        cases[family] = data.loc[data.split.eq("validation")].set_index(CASE_KEY).sort_index()

    main = cases["random_forest"]
    history = pd.read_csv(
        evaluation / "inference_history.csv",
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )
    issues = main.index.get_level_values("reference_at_utc").unique()
    incumbent = {
        "family": "random_forest",
        "name": MODEL,
        "version": VERSION,
        "manifest_sha256": MANIFEST,
    }
    models = dict(active["models"]) if active else {}
    profile = dict(active["error_limits"]) if active else {}
    references = {item["manifest_sha256"]: item for item in [incumbent, *models.values()]}
    forecasts, manifests, legacy_errors = {}, {}, {}
    for checksum, item in references.items():
        bundle = load_candidate(registry, item)
        manifests[checksum] = bundle[1]
        forecasts[checksum] = aligned_predictions(
            main, candidate_batch(item["family"], bundle, history, issues, city)
        )
        if checksum == MANIFEST:
            legacy_errors = (
                bundle[2]
                .query("split == 'validation'")
                .pivot(index=["metric", "horizon_hours"], columns="model")
            )
    report = {}
    for (metric, horizon), group in main.groupby(level=["metric", "horizon_hours"]):
        key = f"{metric}:{horizon}"
        panel = group[["actual", "eligible", "seasonal_naive"]].copy()
        choices = []
        for family, data in cases.items():
            if metric in data.index.get_level_values("metric"):
                panel[family] = data[family].reindex(panel.index)
                choices.append(family)
        selected = models.get(key, incumbent)
        checksum = selected["manifest_sha256"]
        manifest = manifests[checksum]
        # The incumbent and challenger must both be evaluated after their fitter
        # last saw labels. The challenger cases already start at its outer holdout.
        cutoff = pd.Timestamp(
            manifest.get("fitted_until_exclusive_utc", manifest["validation_start_utc"])
        )
        panel = panel.loc[panel.index.get_level_values("reference_at_utc") >= cutoff].copy()
        if key in models:
            panel["incumbent"] = forecasts[checksum].reindex(panel.index)
        elif metric == "rain_probability":
            panel["incumbent"] = manifest["rain_training_event_rate"]
        else:
            measure = LOSS.get(metric, "rmse")
            scores = legacy_errors.loc[(metric, horizon), measure]
            panel["incumbent"] = (
                forecasts[MANIFEST].reindex(panel.index)
                if scores.random_forest < scores.seasonal_naive
                else panel.seasonal_naive
            )
        decision = validation_winner(metric, panel, choices)
        report[key] = decision
        winner = decision["winner"]
        if winner is None:
            continue
        models[key] = candidates[winner]
        if horizon == 24:
            errors = (panel[winner] - panel.actual).where(panel.eligible)
            errors.index = errors.index.get_level_values("target_at_utc")
            errors = errors.sort_index().asfreq("h")
            if metric == "wind_direction_deg":
                rolling = ((errors + 180) % 360 - 180).abs().rolling(24).mean()
            else:
                rolling = errors.pow(2).rolling(24).mean()
                if metric != "rain_probability":
                    rolling = np.sqrt(rolling)
            upper = rolling.max()
            profile[metric] = {
                "model": models[key]["name"],
                "model_version": models[key]["version"],
                "measure": LOSS.get(metric, "rmse"),
                "upper": float(upper) if np.isfinite(upper) else None,
            }
    return {
        "schema_version": 1,
        "city_id": city.city_id,
        "models": models,
        "error_limits": profile,
    }, report


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
