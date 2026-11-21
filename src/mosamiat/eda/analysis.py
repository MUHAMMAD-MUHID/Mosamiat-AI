"""Numerical audits with explicit missingness, sample support and circular wind."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mosamiat.backfill import METRICS
from mosamiat.features.time_features import encode_wind
from mosamiat.reference import ReferenceDataset

WEATHER_METRICS = [
    "temperature_c",
    "humidity_pct",
    "precipitation_mm",
    "rain_probability",
    "wind_speed_mps",
    "wind_direction_deg",
]
AIR_METRICS = [name for name in METRICS if name not in WEATHER_METRICS]
IQR_MULTIPLIER = 1.5
MIN_CORRELATION_PAIRS = 3


@dataclass(frozen=True)
class Analysis:
    values: pd.DataFrame
    metrics: pd.DataFrame
    anomalies: pd.DataFrame
    sources: pd.DataFrame
    correlations: pd.DataFrame
    correlation_counts: pd.DataFrame
    hourly_profile: pd.DataFrame
    daily_profile: pd.DataFrame
    daily_coverage: pd.DataFrame
    daily_precipitation: pd.DataFrame
    summary: dict


def longest_missing(series: pd.Series) -> int:
    absent = series.isna()
    return int(absent.groupby((~absent).cumsum()).sum().max()) if len(series) else 0


def circular_summary(direction: pd.Series) -> dict:
    angle = np.deg2rad(direction.dropna().to_numpy() % 360)
    if not len(angle):
        return {"count": 0, "mean_direction_deg": None, "resultant_length": None}
    sine, cosine = float(np.sin(angle).mean()), float(np.cos(angle).mean())
    strength = float(np.hypot(sine, cosine))
    mean = float(np.rad2deg(np.arctan2(sine, cosine)) % 360) if strength > 1e-12 else None
    return {"count": len(angle), "mean_direction_deg": mean, "resultant_length": strength}


def linear_values(values: pd.DataFrame) -> pd.DataFrame:
    """Exclude undefined calm/unknown-speed directions from circular statistics."""
    direction = values.wind_direction_deg.where(values.wind_speed_mps.gt(0))
    return encode_wind(values.assign(wind_direction_deg=direction))


def profile(values: pd.DataFrame, groups, group_name: str) -> pd.DataFrame:
    """Wind components replace degrees in linear aggregates and correlations."""
    encoded = linear_values(values)
    means = encoded.groupby(groups).mean()
    counts = encoded.groupby(groups).count().add_suffix("_count")
    result = means.join(counts)
    result.index.name = group_name
    return result


def analyze(reference: ReferenceDataset, timezone: str) -> Analysis:
    source_values = reference.values
    grid = reference.expected_index
    if grid is None:
        grid = pd.date_range(source_values.index.min(), source_values.index.max(), freq="h")
    values = source_values.reindex(grid)
    values.index.name = "valid_at_utc"
    rows, flags = [], []
    for metric, unit in METRICS.items():
        series = values[metric]
        valid = series.dropna()
        circular = metric == "wind_direction_deg"
        record = {
            "metric": metric,
            "unit": unit,
            "expected_hours": len(grid),
            "present": len(valid),
            "missing": int(series.isna().sum()),
            "missing_pct": float(series.isna().mean() * 100),
            "longest_gap_hours": longest_missing(series),
            "minimum": valid.min(),
            "median": np.nan if circular else valid.median(),
            "mean": np.nan if circular else valid.mean(),
            "std": np.nan if circular else valid.std(ddof=0),
            "maximum": valid.max(),
            "q25": np.nan if circular else valid.quantile(0.25),
            "q75": np.nan if circular else valid.quantile(0.75),
            "iqr_lower": np.nan,
            "iqr_upper": np.nan,
            "iqr_review_count": 0,
            "iqr_available": False,
        }
        if len(valid) >= 4 and not circular:
            iqr = record["q75"] - record["q25"]
            if iqr > 0:
                lower = record["q25"] - IQR_MULTIPLIER * iqr
                upper = record["q75"] + IQR_MULTIPLIER * iqr
                unusual = valid[(valid < lower) | (valid > upper)]
                record.update(
                    iqr_lower=lower,
                    iqr_upper=upper,
                    iqr_review_count=len(unusual),
                    iqr_available=True,
                )
                flags.extend(
                    {
                        "valid_at_utc": timestamp,
                        "metric": metric,
                        "value": value,
                        "iqr_lower": lower,
                        "iqr_upper": upper,
                    }
                    for timestamp, value in unusual.items()
                )
        rows.append(record)
    metrics = pd.DataFrame(rows).set_index("metric")
    anomalies = pd.DataFrame(
        flags, columns=["valid_at_utc", "metric", "value", "iqr_lower", "iqr_upper"]
    )
    encoded = linear_values(values)
    present = encoded.notna().astype("int64")
    counts = present.T @ present
    correlations = encoded.corr(min_periods=MIN_CORRELATION_PAIRS)
    local = grid.tz_convert(timezone)
    hourly = profile(values, local.hour, "local_hour").reindex(range(24))
    count_columns = hourly.columns[hourly.columns.str.endswith("_count")]
    hourly[count_columns] = hourly[count_columns].fillna(0).astype("int64")
    daily = profile(values, local.date, "local_date")
    coverage = values.notna().groupby(grid.date).mean() * 100
    coverage.index.name = "utc_date"
    rain = values.precipitation_mm
    daily_rain = rain.groupby(local.date).agg(["count", "sum"])
    daily_rain.columns = ["present_hours", "observed_sum_mm"]
    daily_rain["observed_sum_mm"] = daily_rain.observed_sum_mm.where(daily_rain.present_hours > 0)
    # DST-aware expected length for each local calendar day, including clipped edges.
    daily_rain["expected_day_hours"] = [
        len(
            pd.date_range(
                pd.Timestamp(day, tz=timezone),
                pd.Timestamp(day, tz=timezone) + pd.DateOffset(days=1),
                freq="h",
                inclusive="left",
            )
        )
        for day in daily_rain.index
    ]
    daily_rain["complete_local_day"] = daily_rain.present_hours == daily_rain.expected_day_hours
    daily_rain.index.name = "local_date"
    source = reference.rows
    sources = (
        source.groupby(
            ["source", "product", "data_kind", "metric", "unit", "aqi_scale"], dropna=False
        )
        .agg(
            source_rows=("value", "size"),
            non_null=("value", "count"),
            first_valid_at_utc=("valid_at_utc", "min"),
            last_valid_at_utc=("valid_at_utc", "max"),
            first_retrieved_at_utc=("retrieved_at_utc", "min"),
            last_retrieved_at_utc=("retrieved_at_utc", "max"),
        )
        .reset_index()
    )
    moving = values.wind_speed_mps.gt(0) & values.wind_direction_deg.notna()
    circular = circular_summary(values.loc[moving, "wind_direction_deg"])
    warnings = [
        "ERA5/CAMS provider reference values are not verified station observations.",
        "Retrospective retrieval does not establish historical forecast-time availability.",
        "IQR flags identify values for review; legitimate weather extremes are not errors.",
        "Correlations are descriptive, autocorrelated and do not establish causation.",
        "Source/model changes, seasons and unequal historical access can mimic concept drift.",
    ]
    if len(grid) < 24 * 30:
        warnings.append(
            "Less than 30 days: hourly/day profiles are descriptive; seasonal inference is unsupported."
        )
    if values.rain_probability.isna().all():
        warnings.append(
            "Historical rain probability is unavailable; no event definition or probability label is inferred."
        )
    if len(grid) < 49:
        warnings.append(
            "Insufficient history for 24-hour feature warm-up plus a complete t+24 label."
        )
    summary = {
        "schema_version": 1,
        "expected_hours": len(grid),
        "source_hours": len(source_values),
        "missing_timestamps": len(grid.difference(source_values.index)),
        "duplicate_metric_hours": int(source.duplicated(["valid_at_utc", "metric"]).sum()),
        "valid_from_utc": grid.min().isoformat(),
        "valid_until_utc": grid.max().isoformat(),
        "local_dates": len(set(local.date)),
        "calendar_months": sorted(set(local.strftime("%Y-%m"))),
        "timezone": timezone,
        "is_ground_truth": False,
        "operational_point_in_time_safe": False,
        "rain": {
            "present_hours": int(rain.count()),
            "zero_hours": int(rain.eq(0).sum()),
            "positive_hours": int(rain.gt(0).sum()),
            "zero_fraction_of_present": float(rain.eq(0).sum() / rain.count())
            if rain.count()
            else None,
            "observed_sum_mm": float(rain.sum()) if rain.count() else None,
        },
        "wind": circular
        | {
            "calm_hours": int(values.wind_speed_mps.eq(0).sum()),
            "unknown_speed_direction_hours": int(
                (values.wind_speed_mps.isna() & values.wind_direction_deg.notna()).sum()
            ),
        },
        "aqi_scales": sorted(
            source.loc[source.metric.eq("aqi_value"), "aqi_scale"].dropna().unique()
        ),
        "iqr_multiplier": IQR_MULTIPLIER,
        "minimum_correlation_pairs": MIN_CORRELATION_PAIRS,
        "warnings": warnings,
    }
    return Analysis(
        values,
        metrics,
        anomalies,
        sources,
        correlations,
        counts,
        hourly,
        daily,
        coverage,
        daily_rain,
        summary,
    )
