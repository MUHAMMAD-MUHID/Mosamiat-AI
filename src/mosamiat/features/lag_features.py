"""Use strictly earlier hourly values; leave gaps and warm-up values missing."""

import pandas as pd


def lag_features(values: pd.DataFrame, lags: list[int], windows: list[int]) -> pd.DataFrame:
    columns = {}
    clock = pd.Series(values.index, index=values.index)
    for metric, series in values.items():
        past = series.shift(1)
        columns[f"{metric}_missing"] = past.isna()
        last_valid = clock.where(series.notna()).ffill().shift(1)
        columns[f"{metric}_age_hours"] = (clock - last_valid).dt.total_seconds() / 3600
        for lag in lags:
            columns[f"{metric}_lag_{lag}h"] = series.shift(lag)
        for window in windows:
            rolling = past.rolling(window, min_periods=window)
            columns[f"{metric}_mean_{window}h"] = rolling.mean()
            columns[f"{metric}_variance_{window}h"] = rolling.var(ddof=0)
        columns[f"{metric}_change_1h"] = past - series.shift(2)
    return pd.DataFrame(columns, index=values.index)
