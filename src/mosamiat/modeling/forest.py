"""Direct multi-output hourly forecasts using one forest per forecast variable."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer

from mosamiat.collect import CollectionError
from mosamiat.modeling.baseline import FORECAST_METRICS

HORIZONS = 24
PARAMETERS = dict(
    n_estimators=64, max_depth=10, min_samples_leaf=3, max_features=0.5, n_jobs=1, random_state=42
)


@dataclass
class ForecastForest:
    columns: list[str]
    imputer: SimpleImputer
    estimators: dict
    training_counts: dict[str, int]
    wind_fallback_deg: float

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """One row per issue and horizon, with the eight forecast fields in native units."""
        if not features.index.is_unique:
            raise CollectionError("Duplicate forecast issue hours")
        matrix = self.imputer.transform(features[self.columns]).astype(np.float32, copy=False)
        outputs = {}
        for metric, estimator in self.estimators.items():
            if metric == "rain_probability":
                # Multiplication by the class value also handles a single class.
                predictions = np.column_stack(
                    [
                        proba @ classes
                        for proba, classes in zip(
                            estimator.predict_proba(matrix), estimator.classes_, strict=True
                        )
                    ]
                )
            elif metric == "wind_direction_deg":
                sine, cosine = np.split(estimator.predict(matrix), 2, axis=1)
                predictions = np.rad2deg(np.arctan2(sine, cosine)) % 360
                predictions = np.where(
                    np.hypot(sine, cosine) > 1e-8, predictions, self.wind_fallback_deg
                )
                predictions %= 360
            else:
                predictions = estimator.predict(matrix)
            outputs[metric] = predictions.ravel()
        index = pd.MultiIndex.from_product(
            [features.index, range(1, HORIZONS + 1)], names=["reference_at_utc", "horizon_hours"]
        )
        result = pd.DataFrame(outputs, index=index)
        result.insert(
            0,
            "target_at_utc",
            index.get_level_values(0) + pd.to_timedelta(index.get_level_values(1), unit="h"),
        )
        return result


def fit(
    features: pd.DataFrame,
    values: pd.DataFrame,
    rain_events: pd.Series,
    validation_start: pd.Timestamp,
    warmup: int,
) -> ForecastForest:
    if not features.index.equals(values.index) or not values.index.equals(rain_events.index):
        raise CollectionError("Features, reference values and rain events must align")
    hours = values.index
    if (
        not isinstance(hours, pd.DatetimeIndex)
        or hours.empty
        or str(hours.tz) != "UTC"
        or not hours.equals(pd.date_range(hours[0], periods=len(hours), freq="h"))
        or not (hours == hours.floor("h")).all()
    ):
        raise CollectionError("Direct target windows require continuous UTC hours")
    if not rain_events.dropna().isin([0, 1]).all():
        raise CollectionError("Expected binary liquid-rain events")
    training = values.loc[values.index < validation_start]
    # The last valid issue is validation_start - 25h. A label at the boundary
    # belongs to validation. Never materialize future holdout labels for fitting.
    inputs = features.loc[training.index].iloc[warmup:-HORIZONS]
    if len(inputs) < 2 * PARAMETERS["min_samples_leaf"]:
        raise CollectionError("Insufficient training history after warm-up and horizon purging")
    useful = inputs.nunique(dropna=True).gt(1)
    columns = useful.index[useful].tolist()
    if not columns:
        raise CollectionError("Training features have no varying values")
    imputer = SimpleImputer(strategy="median")
    matrix = imputer.fit_transform(inputs[columns]).astype(np.float32)
    estimators, counts = {}, {}
    for metric in FORECAST_METRICS:
        series = training[metric]
        if metric == "rain_probability":
            series = rain_events.loc[training.index]
        elif metric == "wind_direction_deg":
            series = series.where(training.wind_speed_mps > 0)
        hourly = series.iloc[warmup + 1 :].to_numpy(dtype=float)
        labels = np.lib.stride_tricks.sliding_window_view(hourly, HORIZONS)
        valid = np.isfinite(labels).all(axis=1)
        counts[metric] = int(valid.sum())
        if counts[metric] < 2 * PARAMETERS["min_samples_leaf"]:
            raise CollectionError(f"Insufficient complete training label paths for {metric}")
        target = labels[valid]
        if metric == "wind_direction_deg":
            radians = np.deg2rad(target)
            target = np.column_stack([np.sin(radians), np.cos(radians)])
        if metric == "rain_probability":
            estimator = RandomForestClassifier(**PARAMETERS)
            target = target.astype(np.uint8)
        else:
            estimator = RandomForestRegressor(**PARAMETERS)
        estimators[metric] = estimator.fit(matrix[valid], target)
    last_direction = (
        training.wind_direction_deg.where(training.wind_speed_mps > 0).dropna().iloc[-1]
    )
    return ForecastForest(columns, imputer, estimators, counts, float(last_direction) % 360)
