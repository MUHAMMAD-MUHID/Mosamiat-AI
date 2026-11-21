"""Direct 1–24-hour reference targets, kept separate from input features."""

import pandas as pd


def build_targets(values: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    tables = []
    for horizon in horizons:
        labels = values.shift(-horizon).copy()
        labels.columns = [f"target_{name}" for name in labels]
        labels.insert(0, "horizon_hours", horizon)
        labels.insert(0, "target_at_utc", values.index + pd.Timedelta(hours=horizon))
        labels.index.name = "reference_at_utc"
        for metric in values:
            labels[f"target_{metric}_missing"] = labels[f"target_{metric}"].isna()
        tables.append(labels.reset_index())
    return pd.concat(tables, ignore_index=True).sort_values(
        ["reference_at_utc", "horizon_hours"], ignore_index=True
    )
