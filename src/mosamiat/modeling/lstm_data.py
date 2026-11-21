"""Training-only normalization and chronological hourly sequence windows."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from mosamiat.collect import CollectionError
from mosamiat.features.pipeline import validate_hourly
from mosamiat.features.time_features import calendar_features, encode_wind

HOURS = 24
SCALARS = (
    "aqi_value",
    "temperature_c",
    "precipitation_mm",
    "humidity_pct",
    "uv_index",
    "wind_speed_mps",
)
DIRECTION = ("wind_direction_sin", "wind_direction_cos")
REGRESSION = (*SCALARS, *DIRECTION)


def channels(values: pd.DataFrame) -> pd.DataFrame:
    validate_hourly(values)
    frame = encode_wind(values.drop(columns="rain_probability"))
    frame.loc[~(values.wind_speed_mps > 0), list(DIRECTION)] = np.nan
    return frame.loc[:, [*REGRESSION, *sorted(set(frame) - set(REGRESSION))]]


@dataclass
class Preprocessing:
    columns: list[str]
    means: list[float]
    scales: list[float]
    timezone: str
    rain_rate: float
    wind_fallback_deg: float

    @classmethod
    def fit(cls, training: pd.DataFrame, rain: pd.Series, timezone: str):
        if not training.index.equals(rain.index) or not rain.dropna().isin([0, 1]).all():
            raise CollectionError("Expected aligned binary rain events")
        if training[list(REGRESSION)].count().min() == 0 or not rain.notna().any():
            raise CollectionError("Training requires reference labels for every forecast field")
        means = training.mean().fillna(0.0)
        scales = training.std(ddof=0).fillna(1.0)
        scales = scales.where(scales > 1e-8, 1.0)
        means.loc[list(DIRECTION)] = 0.0
        scales.loc[list(DIRECTION)] = 1.0
        last = training[list(DIRECTION)].dropna().iloc[-1]
        bearing = float(np.rad2deg(np.arctan2(*last)) % 360)
        return cls(
            list(training), means.tolist(), scales.tolist(), timezone, float(rain.mean()), bearing
        )

    def normalized(self, frame: pd.DataFrame) -> np.ndarray:
        return ((frame[self.columns].to_numpy(dtype=float) - self.means) / self.scales).astype(
            "float32"
        )

    def inputs(self, frame: pd.DataFrame) -> torch.Tensor:
        values = self.normalized(frame)
        clock = calendar_features(frame.index, self.timezone)[["hour_sin", "hour_cos"]]
        matrix = np.column_stack([np.nan_to_num(values, nan=0.0), np.isnan(values), clock])
        return torch.from_numpy(matrix.astype("float32"))


class HourlyWindows(Dataset):
    """An issue at i uses rows [i-24, i) and targets [i+1, i+25)."""

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor, start: int, stop: int):
        self.inputs, self.targets = inputs, targets
        self.start = max(start, HOURS)
        self.stop = stop - HOURS
        if self.stop <= self.start or stop > len(targets) or len(inputs) != len(targets):
            raise CollectionError(
                "Insufficient history for complete 24-hour windows inside the split"
            )

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        issue = self.start + index
        return self.inputs[issue - HOURS : issue], self.targets[issue + 1 : issue + 1 + HOURS]
