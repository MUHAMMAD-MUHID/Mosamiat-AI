"""Build shared held-out cases for the direct 1–24 hour forecast contract."""

import numpy as np
import pandas as pd

FORECAST_METRICS = (
    "aqi_value",
    "temperature_c",
    "precipitation_mm",
    "rain_probability",
    "humidity_pct",
    "uv_index",
    "wind_speed_mps",
    "wind_direction_deg",
)
CASE_KEY = ["split", "reference_at_utc", "target_at_utc", "horizon_hours", "metric"]


def baseline_cases(
    values: pd.DataFrame, folds: dict[str, pd.DataFrame], rain_events: pd.Series | None = None
) -> pd.DataFrame:
    targets = pd.concat({name: folds[name] for name in ("validation", "test")}, names=["split"])
    targets = targets.reset_index(level="split").reset_index(drop=True)
    # Repeat the last 24 completed hours. At h=24 yesterday's same-hour value
    # falls at issue time and is unavailable, so repeat the value at issue-24h.
    source = targets.target_at_utc - pd.Timedelta(hours=24)
    source = source.where(source < targets.reference_at_utc, source - pd.Timedelta(hours=24))
    history = values.reindex(source).reset_index(drop=True)
    tables = []
    for metric in FORECAST_METRICS:
        actual = targets[f"target_{metric}"]
        seasonal = history[metric]
        if metric == "wind_direction_deg":
            actual = actual.where(targets.target_wind_speed_mps > 0)
            seasonal = seasonal.where(history.wind_speed_mps > 0)
        if metric == "rain_probability":
            # Brier verification requires a binary observed rain event.
            # The source schema supplies only provider probabilities and total precipitation.
            actual = pd.Series(np.nan, index=targets.index)
            if rain_events is not None:
                actual = rain_events.reindex(targets.target_at_utc).reset_index(drop=True)
                seasonal = rain_events.reindex(source).reset_index(drop=True)
        cases = targets[CASE_KEY[:4]].assign(
            metric=metric,
            source_at_utc=source,
            actual=actual,
            seasonal_naive=seasonal,
            eligible=actual.notna() & seasonal.notna(),
        )
        tables.append(cases)
    return pd.concat(tables, ignore_index=True).set_index(CASE_KEY)
