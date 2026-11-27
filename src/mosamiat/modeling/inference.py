"""Build an hourly forecast from the registered benchmark and one recent source revision."""

import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from mosamiat.backfill import METRICS, digest
from mosamiat.collect import AIR, PROBABILITY, WEATHER, City, CollectionError
from mosamiat.features.pipeline import transform
from mosamiat.hopsworks_compat import list_project_files
from mosamiat.modeling.baseline import FORECAST_METRICS
from mosamiat.storage import load_snapshot, read_json

MODEL = "mosamiat_islamabad_random_forest"
VERSION = 1
MANIFEST = "1487f5d407c6a4df2c6f5af7e5abd99d9a7abeebd04fb2b6d4b81cfe415ac726"
POLICY_NAME = re.compile(r"policy-(\d{20})-([0-9a-f]{64})\.json")


def read_policy(api, city, checksum=None):
    folder = f"Resources/mosamiat_{city.city_id}/serving"
    if not api.exists(folder):
        return None, MANIFEST
    matches = []
    offset = 0
    while True:
        page = list_project_files(api, folder, offset=offset, limit=100)
        for path in page:
            name = path.rsplit("/", 1)[-1]
            if (match := POLICY_NAME.fullmatch(name)) and (
                checksum is None or match[2] == checksum
            ):
                matches.append(name)
        if len(page) < 100:
            break
        offset += len(page)
    if not matches:
        return None, MANIFEST
    name = max(matches)
    expected = POLICY_NAME.fullmatch(name)[2]
    with TemporaryDirectory() as temp:
        path = Path(temp) / name
        api.download(f"{folder}/{name}", local_path=str(path))
        if digest(path) != expected:
            raise CollectionError("Serving selection checksum mismatch")
        policy = read_json(path)
    if policy["schema_version"] != 1 or policy["city_id"] != city.city_id:
        raise CollectionError("Serving selection is for a different schema or city")
    return policy, expected


def active_forecast(rows, policy, predictions):
    specs = {m.name: m for m in (WEATHER | AIR | PROBABILITY).values()}
    updated = []
    for row in rows:
        metric, horizon = row["metric"], row["horizon_hours"]
        chosen = policy["models"].get(f"{metric}:{horizon}")
        if chosen is None:
            updated.append(row)
            continue
        name, version = chosen["name"], chosen["version"]
        forecast_row = predictions[name, version].loc[horizon]
        value = float(forecast_row[metric])
        spec = specs[metric]
        if (
            forecast_row.target_at_utc != pd.Timestamp(row["target_at_utc"])
            or not np.isfinite(value)
            or (spec.minimum is not None and value < spec.minimum / spec.divisor)
            or (spec.maximum is not None and value > spec.maximum / spec.divisor)
        ):
            raise CollectionError(f"Invalid active {metric} prediction at horizon {horizon}")
        updated.append(dict(row, value=value, model_name=name, model_version=version))
    return updated


def recent_history(snapshot: Path, city: City, now: pd.Timestamp):
    frame = load_snapshot(snapshot)
    reference = now.floor("h")
    if (
        not frame.city_id.eq(city.city_id).all()
        or not frame.city_timezone.eq(city.timezone).all()
        or not frame.requested_latitude.eq(city.latitude).all()
        or not frame.requested_longitude.eq(city.longitude).all()
        or not frame.data_kind.eq("provider_forecast").all()
        or not frame.retrieved_at_utc.between(reference, now).all()
    ):
        raise CollectionError("Forecast requires a recent city snapshot retrieved in this UTC hour")
    past = frame[frame.valid_at_utc.lt(reference)]
    history = past.pivot(index="valid_at_utc", columns="metric", values="value")
    hours = pd.date_range(end=reference - pd.Timedelta(hours=1), periods=24, freq="h")
    if not hours.isin(history.index).all() or set(history.columns) != set(METRICS):
        raise CollectionError("Snapshot lacks the 24 completed hours needed for inference")
    history = history.loc[hours]
    if history[list(FORECAST_METRICS)].iloc[-1].isna().any():
        raise CollectionError("Latest completed hour is missing a forecast field")
    return history, frame.retrieved_at_utc.max()


def issue_features(history, reference, city, definitions):
    # Empty issue-hour row supplies the forecast clock. All inputs come from previous hours.
    values = history.reindex(history.index.append(pd.DatetimeIndex([reference])))
    return transform(values, definitions, city.timezone).tail(1)


def forecast(history, reference, city, bundle, *, sample=None):
    forest, manifest, scores = bundle
    if manifest["city_id"] != city.city_id or manifest["timezone"] != city.timezone:
        raise CollectionError("Registered model is for a different city")
    if sample is None:
        sample = issue_features(history, reference, city, manifest["feature_definitions"])
    predictions = forest.predict(sample).xs(reference, level="reference_at_utc")
    scores = (
        scores.loc[scores.split.eq("validation")]
        .set_index(["metric", "horizon_hours", "model"])
        .sort_index()
    )
    specs = {m.name: m for m in (WEATHER | AIR | PROBABILITY).values()}
    rows = []
    for h in range(1, 25):
        target = reference + pd.Timedelta(hours=h)
        if predictions.at[h, "target_at_utc"] != target:
            raise CollectionError("Model returned an unexpected target timestamp")
        previous = history.loc[target - pd.Timedelta(hours=48 if h == 24 else 24)]
        for metric in FORECAST_METRICS:
            if metric == "rain_probability":
                value = manifest["rain_training_event_rate"]
                model = "training_rain_frequency"
            else:
                measure = "circular_mae_deg" if metric == "wind_direction_deg" else "rmse"
                errors = scores.loc[(metric, h), measure]
                if not np.isfinite(errors[["random_forest", "seasonal_naive"]]).all():
                    raise CollectionError(f"Missing validation errors for {metric} at horizon {h}")
                if errors["random_forest"] < errors["seasonal_naive"]:
                    model, value = MODEL, predictions.at[h, metric]
                else:
                    model, value = "seasonal_naive", previous[metric]
                    if metric == "wind_direction_deg" and previous.wind_speed_mps == 0:
                        raise CollectionError(
                            "Previous-day wind direction is undefined in calm air"
                        )
            spec = specs[metric]
            if (
                not np.isfinite(value)
                or (spec.minimum is not None and value < spec.minimum / spec.divisor)
                or (spec.maximum is not None and value > spec.maximum / spec.divisor)
            ):
                raise CollectionError(f"Invalid {metric} prediction at horizon {h}")
            rows.append(
                dict(
                    target_at_utc=target.isoformat(),
                    horizon_hours=h,
                    metric=metric,
                    value=float(value),
                    unit=METRICS[metric],
                    model_name=model,
                    model_version=VERSION,
                )
            )
    return rows
