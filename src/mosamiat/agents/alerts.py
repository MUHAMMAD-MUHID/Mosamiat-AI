"""Version 1 of the project's forecast screening rules. Units follow the forecast ledger."""

from dataclasses import dataclass
from itertools import groupby

import numpy as np
import pandas as pd

from mosamiat.backfill import METRICS
from mosamiat.collect import CollectionError
from mosamiat.modeling.baseline import FORECAST_METRICS

RULE_VERSION = 1


@dataclass(frozen=True)
class Rule:
    key: str
    label: str
    field: str
    threshold: float
    higher: bool
    elevated: float
    public: str
    sensitive: str
    workers: str


RULES = (
    Rule(
        "air_quality",
        "Poor air quality",
        "aqi_value",
        101,
        True,
        151,
        "Check local air-quality updates before strenuous outdoor activity.",
        "Consider a shorter or less strenuous outdoor activity.",
        "Plan strenuous tasks for lower-AQI hours where possible.",
    ),
    Rule(
        "heat",
        "High temperature",
        "temperature_c",
        35,
        True,
        40,
        "Plan demanding activities for cooler hours.",
        "Choose a cool place for breaks.",
        "Plan water and shaded rest breaks.",
    ),
    Rule(
        "cold",
        "Near-freezing temperature",
        "temperature_c",
        2,
        False,
        0,
        "Take warm layers outdoors.",
        "Plan access to a warm indoor space.",
        "Arrange warm breaks during prolonged outdoor work.",
    ),
    Rule(
        "rain",
        "Heavy hourly precipitation",
        "precipitation_mm",
        8,
        True,
        16,
        "Allow extra travel time and check local updates.",
        "Use a sheltered route where possible.",
        "Review drainage and footing before outdoor work.",
    ),
    Rule(
        "rain_chance",
        "Elevated rain chance",
        "rain_probability",
        0.6,
        True,
        0.8,
        "Carry rain protection.",
        "Plan a sheltered waiting place.",
        "Protect equipment and prepare a wet-weather work plan.",
    ),
    Rule(
        "humidity",
        "High humidity",
        "humidity_pct",
        85,
        True,
        95,
        "Check the temperature before planning strenuous activity.",
        "Choose a ventilated place for breaks.",
        "Review workload alongside the temperature forecast.",
    ),
    Rule(
        "uv",
        "Sun protection advised",
        "uv_index",
        3,
        True,
        8,
        "Use shade, protective clothing and sun protection outdoors.",
        "Plan shaded activities during the flagged hours.",
        "Schedule shaded breaks and cover exposed skin.",
    ),
    Rule(
        "wind",
        "Strong wind",
        "wind_speed_mps",
        14,
        True,
        21,
        "Secure loose outdoor items.",
        "Choose sheltered routes.",
        "Review site wind limits before exposed work.",
    ),
)


def forecast_hours(payload):
    rows = pd.DataFrame(payload["forecasts"])
    if rows.empty or not {"target_at_utc", "metric", "unit", "value"} <= set(rows):
        raise CollectionError("Forecast rows lack required fields")
    if not rows.unit.eq(rows.metric.map(METRICS)).all():
        raise CollectionError("Unexpected forecast units")
    rows["target_at_utc"] = pd.to_datetime(rows.target_at_utc, utc=True)
    hours = rows.pivot(index="target_at_utc", columns="metric", values="value").sort_index()
    start = pd.Timestamp(payload["reference_at_utc"]) + pd.Timedelta(hours=1)
    if (
        not hours.index.equals(pd.date_range(start, periods=24, freq="h"))
        or set(hours.columns) != set(FORECAST_METRICS)
        or not np.isfinite(hours.to_numpy()).all()
    ):
        raise CollectionError("Alerts require a complete 24-hour forecast for all eight fields")
    return hours


def evaluate(hours):
    alerts = []
    for rule in RULES:
        series = hours[rule.field]
        crossed = series >= rule.threshold if rule.higher else series <= rule.threshold
        for active, items in groupby(
            zip(crossed, series.items(), strict=True), key=lambda item: item[0]
        ):
            if not active:
                continue
            episode = [item for _, item in items]
            peak_at, peak = (max if rule.higher else min)(episode, key=lambda item: item[1])
            elevated = peak >= rule.elevated if rule.higher else peak <= rule.elevated
            alerts.append(
                dict(
                    rule=rule.key,
                    label=rule.label,
                    metric=rule.field,
                    unit=METRICS[rule.field],
                    severity="elevated" if elevated else "advisory",
                    threshold=rule.threshold,
                    comparison=">=" if rule.higher else "<=",
                    first=episode[0][0].isoformat(),
                    last=episode[-1][0].isoformat(),
                    peak=float(peak),
                    peak_at=peak_at.isoformat(),
                    public=rule.public,
                    sensitive=rule.sensitive,
                    workers=rule.workers,
                )
            )
    return sorted(alerts, key=lambda alert: (alert["first"], alert["rule"]))


def new_episodes(current, previous):
    """A continuing episode can move by one hourly target as the forecast rolls forward."""
    hour = pd.Timedelta(hours=1)
    return [
        item
        for item in current
        if not any(
            old["rule"] == item["rule"]
            and (old["severity"] == "elevated" or item["severity"] == "advisory")
            and pd.Timestamp(item["first"]) <= pd.Timestamp(old["last"]) + hour
            and pd.Timestamp(old["first"]) <= pd.Timestamp(item["last"]) + hour
            for old in previous
        )
    ]
