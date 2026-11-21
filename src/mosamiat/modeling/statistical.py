"""Fit and evaluate daily harmonic SARIMAX models for temperature and humidity."""

import argparse
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from mosamiat.collect import City, CollectionError
from mosamiat.modeling.evaluate import comparison_cases, publish_evaluation
from mosamiat.modeling.metrics import score_predictions
from mosamiat.reference import load_reference

METRICS = ("temperature_c", "humidity_pct")
MIN_OBSERVATIONS = 72


def daily_terms(times: pd.DatetimeIndex) -> pd.DataFrame:
    angle = 2 * np.pi * times.hour / 24
    return pd.DataFrame(
        {
            "const": 1.0,
            "sin24": np.sin(angle),
            "cos24": np.cos(angle),
            "sin12": np.sin(2 * angle),
            "cos12": np.cos(2 * angle),
        },
        index=times,
    )


def fit(training: pd.Series) -> dict[str, float]:
    observed = training.dropna()
    if len(observed) < MIN_OBSERVATIONS or observed.index.hour.nunique() < 24:
        raise CollectionError(
            f"{training.name}: need at least {MIN_OBSERVATIONS} observed training "
            "hours with all 24 UTC hours represented"
        )
    if observed.nunique() == 1:
        raise CollectionError(f"{training.name}: training values are constant")
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as error:
        raise CollectionError("Install the modeling extra to fit SARIMAX") from error
    training = training.asfreq("h")
    model = SARIMAX(training, order=(1, 0, 0), trend="n", exog=daily_terms(training.index))
    result = model.fit(disp=False, maxiter=200)
    if not result.mle_retvals["converged"] or not np.isfinite(result.params).all():
        raise CollectionError(f"{training.name}: SARIMAX parameter estimation failed to converge")
    return result.params.to_dict()


def predict(history: pd.Series, keys: pd.MultiIndex, params: dict[str, float]) -> pd.Series:
    """Forecast each key from the last observation strictly before its issue hour."""
    if not history.index.is_unique or not history.index.is_monotonic_increasing:
        raise CollectionError("Prediction history must contain unique chronological hours")
    issues = keys.get_level_values("reference_at_utc")
    targets = keys.get_level_values("target_at_utc")
    design = daily_terms(targets)
    beta = np.array([params[col] for col in design])
    forecasts = design.to_numpy() @ beta
    observed = history.dropna()
    # Regression with AR(1) errors. Between observations, each elapsed hour
    # multiplies the last observed residual by phi, including hours in gaps.
    positions = observed.index.searchsorted(issues, side="left") - 1
    known = positions >= 0
    last = observed.iloc[positions[known]]
    residual = last.to_numpy() - daily_terms(last.index).to_numpy() @ beta
    steps = (targets[known] - last.index) // pd.Timedelta(hours=1)
    forecasts[known] += residual * np.power(params["ar.L1"], steps)
    if history.name == "humidity_pct":
        forecasts = np.clip(forecasts, 0, 100)
    return pd.Series(forecasts, index=keys, name="sarimax")


def evaluate(
    dataset: Path,
    output: Path,
    city: City,
    validation_start: str | None = None,
    test_start: str | None = None,
) -> dict:
    reference = load_reference(dataset, city)
    reference.protect_output(dataset, output)
    cases, manifest = comparison_cases(dataset, reference, city, validation_start, test_start)
    validation = pd.Timestamp(manifest["validation_start_utc"])
    cases = cases[cases.index.get_level_values("metric").isin(METRICS)].copy()
    models, predictions = {}, []
    for metric in METRICS:
        series = reference.values[metric]
        training = series[series.index < validation]
        params = fit(training)
        models[metric] = {"params": params, "observed_training_hours": int(training.count())}
        keys = cases.xs(metric, level="metric", drop_level=False).index
        predictions.append(predict(series, keys, params))
    cases["sarimax"] = pd.concat(predictions)
    tables = {}
    for name in ("seasonal_naive", "sarimax"):
        scores = score_predictions(cases, cases[name])
        tables[name] = scores[scores.index.get_level_values("metric").isin(METRICS)]
    comparison = pd.concat(tables, names=["model"])
    manifest.update(
        kind="sarimax_evaluation",
        models=models,
        statsmodels_version=version("statsmodels"),
        order=[1, 0, 0],
        seasonal_terms="UTC daily Fourier harmonics 1 and 2",
        humidity_bounds_pct=[0, 100],
        fitted_until_exclusive_utc=validation.isoformat(),
    )
    return publish_evaluation(output, manifest, {"cases.csv": cases, "metrics.csv": comparison})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/statistical"))
    parser.add_argument("--city-config", type=Path, default=Path("configs/city.toml"))
    parser.add_argument("--validation-start", help="first validation issue hour with UTC offset")
    parser.add_argument("--test-start", help="first test issue hour with UTC offset")
    args = parser.parse_args(argv)
    try:
        evaluate(
            args.dataset,
            args.output,
            City.load(args.city_config),
            args.validation_start,
            args.test_start,
        )
    except (ValueError, OSError, KeyError, TypeError, np.linalg.LinAlgError) as error:
        print(f"Statistical evaluation failed: {error}", file=sys.stderr)
        return 1
    print(f"Saved fitted parameters and held-out SARIMAX/baseline scores to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
