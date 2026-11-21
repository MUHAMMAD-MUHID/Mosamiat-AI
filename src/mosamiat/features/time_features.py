"""Calendar and circular wind features shared by research and inference."""

import numpy as np
import pandas as pd


def calendar_features(index: pd.DatetimeIndex, timezone: str) -> pd.DataFrame:
    local = index.tz_convert(timezone)
    return pd.DataFrame(
        {
            "hour": local.hour,
            "day_of_week": local.dayofweek,
            "month": local.month,
            "hour_sin": np.sin(2 * np.pi * local.hour / 24),
            "hour_cos": np.cos(2 * np.pi * local.hour / 24),
        },
        index=index,
    )


def encode_wind(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.drop(columns="wind_direction_deg").copy()
    angle = np.deg2rad(frame.wind_direction_deg)
    result["wind_direction_sin"] = np.sin(angle)
    result["wind_direction_cos"] = np.cos(angle)
    return result
