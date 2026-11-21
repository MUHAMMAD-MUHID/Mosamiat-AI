"""Score aligned model predictions on the benchmark's fixed comparison cases."""

import numpy as np
import pandas as pd

from mosamiat.collect import CollectionError
from mosamiat.modeling.baseline import FORECAST_METRICS


def metrics(metric: str, actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape:
        raise CollectionError("Expected paired one-dimensional samples")
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise CollectionError("Scored samples must be finite")
    if not actual.size:
        return {}
    error = predicted - actual
    if metric == "rain_probability":
        if not np.isin(actual, [0, 1]).all() or not ((predicted >= 0) & (predicted <= 1)).all():
            raise CollectionError("Brier score requires binary events and probabilities in [0, 1]")
        return {"brier": np.mean(error**2)}
    if metric == "wind_direction_deg":
        return {"circular_mae_deg": np.mean(np.abs((error + 180) % 360 - 180))}
    mse = np.mean(error**2)
    variance = np.mean((actual - actual.mean()) ** 2)
    return {
        "rmse": np.sqrt(mse),
        "mae": np.mean(np.abs(error)),
        "r2": 1 - mse / variance if np.ptp(actual) > 0 and variance > 0 else np.nan,
    }


def score_predictions(cases: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
    """Require one aligned prediction per eligible key, including for future candidates."""
    if not cases.index.is_unique or not predictions.index.is_unique:
        raise CollectionError("Duplicate evaluation keys")
    if not predictions.index.difference(cases.index).empty:
        raise CollectionError("Unexpected evaluation keys")
    frame = cases.assign(prediction=predictions.reindex(cases.index))
    if frame.loc[frame.eligible, "prediction"].isna().any():
        raise CollectionError("Missing predictions for eligible comparison cases")
    rows = []
    for (split, metric, horizon), group in frame.groupby(
        level=["split", "metric", "horizon_hours"]
    ):
        scored = group[group.eligible]
        rows.append(
            {
                "split": split,
                "metric": metric,
                "horizon_hours": horizon,
                "pairs": len(group),
                "labels": group.actual.count(),
                "scored": len(scored),
                **metrics(metric, scored.actual.to_numpy(), scored.prediction.to_numpy()),
            }
        )
    keys = ["split", "metric", "horizon_hours"]
    grid = pd.MultiIndex.from_product(
        [["validation", "test"], FORECAST_METRICS, range(1, 25)], names=keys
    )
    summary = pd.DataFrame(rows).set_index(keys).reindex(grid)
    counts = ["pairs", "labels", "scored"]
    summary[counts] = summary[counts].fillna(0).astype(int)
    return summary.reindex(columns=[*counts, "rmse", "mae", "r2", "brier", "circular_mae_deg"])
