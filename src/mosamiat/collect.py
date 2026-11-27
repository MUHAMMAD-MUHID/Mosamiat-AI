"""Collect explicitly labeled provider data; never treat it as station observations."""

import argparse
import json
import logging
import math
import re
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from itertools import pairwise
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
import pandas as pd


class CollectionError(ValueError):
    """Source data or collection settings do not meet the collection contract."""


@dataclass(frozen=True)
class City:
    city_id: str
    name: str
    latitude: float
    longitude: float
    timezone: str

    @classmethod
    def load(cls, path: Path) -> "City":
        with path.open("rb") as stream:
            try:
                city = cls(**tomllib.load(stream)["city"])
            except (KeyError, TypeError, tomllib.TOMLDecodeError) as error:
                raise CollectionError(f"Invalid city configuration: {error}") from error
        if not isinstance(city.city_id, str) or not re.fullmatch(r"[a-z0-9_-]+", city.city_id):
            raise CollectionError(
                "city_id must use lowercase letters, digits, underscores or hyphens"
            )
        if not isinstance(city.name, str) or not city.name.strip():
            raise CollectionError("City name must not be empty")
        for value, limit in [(city.latitude, 90), (city.longitude, 180)]:
            if not is_number(value) or not -limit <= value <= limit:
                raise CollectionError(
                    "City coordinates must be finite and within geographic bounds"
                )
        try:
            ZoneInfo(city.timezone)
        except (ZoneInfoNotFoundError, TypeError, ValueError) as error:
            raise CollectionError("City timezone must be a valid IANA timezone") from error
        return city


@dataclass(frozen=True)
class Metric:
    name: str
    api_unit: str
    unit: str
    minimum: float | None = None
    maximum: float | None = None
    divisor: float = 1


WEATHER = {
    "temperature_2m": Metric("temperature_c", "°C", "degC"),
    "relative_humidity_2m": Metric("humidity_pct", "%", "%", 0, 100),
    "precipitation": Metric("precipitation_mm", "mm", "mm", 0),
    "wind_speed_10m": Metric("wind_speed_mps", "m/s", "m/s", 0),
    "wind_direction_10m": Metric("wind_direction_deg", "°", "degree", 0, 360),
}
PROBABILITY = {
    "precipitation_probability": Metric("rain_probability", "%", "1", 0, 100, 100),
}
AIR = {
    "us_aqi": Metric("aqi_value", "USAQI", "US_AQI", 0),
    "uv_index": Metric("uv_index", "", "1", 0),
    **{
        field: Metric(f"{field}_ug_m3", "μg/m³", "ug/m3", 0)
        for field in [
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
        ]
    },
}


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fetch_json(client: httpx.Client, url: str, params: dict, *, live: bool = False) -> dict:
    """Retry only transient failures, with a bounded delay and attempt count."""
    for attempt in range(3):
        try:
            response = client.get(url, params=params)
        except httpx.TransportError as error:
            if attempt == 2:
                raise CollectionError(f"Source unavailable after 3 attempts: {error}") from error
            time.sleep(2**attempt)
            continue
        if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
            live_rate_limit = live and response.status_code == 429
            fallback = (20, 40)[attempt] if live_rate_limit else 2**attempt
            try:
                delay = float(response.headers.get("Retry-After", fallback))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(response.headers["Retry-After"])
                    delay = (retry_at - datetime.now(UTC)).total_seconds()
                except (ValueError, TypeError, OverflowError):
                    delay = fallback
            limit = 120 if live_rate_limit else 30
            if not math.isfinite(delay) or delay > limit:
                logging.getLogger(__name__).warning(
                    "%s returned HTTP %s with a cooldown outside the %ss retry budget",
                    response.request.url.host,
                    response.status_code,
                    limit,
                )
                raise CollectionError("Provider requested a longer cooldown; retry the run later")
            if live_rate_limit:
                delay = max(fallback, delay)
                logging.getLogger(__name__).warning(
                    "%s returned HTTP 429; retrying in %.1fs (attempt %s of 3)",
                    response.request.url.host,
                    delay,
                    attempt + 2,
                )
            time.sleep(max(0, delay))
            continue
        if response.status_code == 429:
            logging.getLogger(__name__).error(
                "%s still returned HTTP 429 after 3 attempts", response.request.url.host
            )
        try:
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPStatusError, ValueError) as error:
            raise CollectionError(f"Invalid source response from {url}: {error}") from error
        if not isinstance(payload, dict) or payload.get("error"):
            raise CollectionError(f"Provider rejected request: {payload}")
        return payload
    raise CollectionError("Source request attempts exhausted")


def normalize(
    payload: dict,
    metrics: dict[str, Metric],
    city: City,
    product: str,
    data_kind: str,
    retrieved_at: str,
) -> list[dict]:
    """Validate hourly arrays and retain nulls with source provenance."""
    hourly = payload.get("hourly")
    units = payload.get("hourly_units")
    if not isinstance(hourly, dict) or not isinstance(units, dict):
        raise CollectionError(f"{product}: missing hourly data or units")
    timestamps = hourly.get("time")
    if not isinstance(timestamps, list) or not timestamps:
        raise CollectionError(f"{product}: missing hourly timestamps")
    if units.get("time") != "unixtime" or payload.get("utc_offset_seconds") != 0:
        raise CollectionError(f"{product}: expected Unix timestamps and UTC offset 0")
    if any(not is_number(stamp) or stamp % 3600 != 0 for stamp in timestamps):
        raise CollectionError(f"{product}: timestamps must fall on UTC hours")
    if any(right - left != 3600 for left, right in pairwise(timestamps)):
        raise CollectionError(f"{product}: timestamps must be consecutive, unique and ordered")
    for key, limit in [("latitude", 90), ("longitude", 180)]:
        value = payload.get(key)
        if not is_number(value) or not -limit <= value <= limit:
            raise CollectionError(f"{product}: invalid returned grid {key}")
    try:
        valid_times = [datetime.fromtimestamp(stamp, UTC).isoformat() for stamp in timestamps]
    except (ValueError, OverflowError, OSError) as error:
        raise CollectionError(f"{product}: invalid timestamp range") from error
    context = {
        "city_id": city.city_id,
        "city_timezone": city.timezone,
        "requested_latitude": city.latitude,
        "requested_longitude": city.longitude,
        "grid_latitude": payload["latitude"],
        "grid_longitude": payload["longitude"],
        "retrieved_at_utc": retrieved_at,
        "source": "open_meteo",
        "product": product,
        "data_kind": data_kind,
    }
    rows = []
    for field, metric in metrics.items():
        values = hourly.get(field)
        if not isinstance(values, list) or len(values) != len(timestamps):
            raise CollectionError(f"{product}: missing or unequal array for {field}")
        if units.get(field) != metric.api_unit:
            raise CollectionError(f"{product}: unexpected unit for {field}: {units.get(field)!r}")
        for valid_at, value in zip(valid_times, values, strict=True):
            if value is not None:
                if not is_number(value):
                    raise CollectionError(f"{product}: non-finite or nonnumeric {field}")
                if metric.minimum is not None and value < metric.minimum:
                    raise CollectionError(f"{product}: {field} below its valid range")
                if metric.maximum is not None and value > metric.maximum:
                    raise CollectionError(f"{product}: {field} above its valid range")
                value /= metric.divisor
            rows.append(
                context
                | {
                    "valid_at_utc": valid_at,
                    "metric": metric.name,
                    "value": value,
                    "unit": metric.unit,
                    "aqi_scale": "US" if field == "us_aqi" else "",
                    "quality_status": "missing" if value is None else "valid",
                }
            )
    return rows


def validate_window(start: date | None, end: date | None) -> None:
    if (start is None) != (end is None):
        raise CollectionError("Provide --start and --end together")
    if start is not None and end is not None:
        if not 0 <= (end - start).days < 31:
            raise CollectionError("History window must contain 1 to 31 inclusive days")
        if end > datetime.now(UTC).date() - timedelta(days=6):
            raise CollectionError(
                "History must end at least six UTC days ago for ERA5 availability"
            )


def write_json(path: Path, content: dict) -> None:
    path.write_text(
        json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def collect(
    city: City,
    output: Path,
    client: httpx.Client,
    start: date | None = None,
    end: date | None = None,
) -> Path:
    """Save one independent source snapshot and an auditable completion manifest."""
    validate_window(start, end)
    historical = start is not None
    common = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "timezone": "GMT",
        "timeformat": "unixtime",
    }
    if historical:
        window = {"start_date": start.isoformat(), "end_date": end.isoformat()}
        first_hour = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
        hours = ((end - start).days + 1) * 24
    else:
        first_hour = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(
            hours=24
        )
        hours = 48
        window = {
            "start_hour": first_hour.strftime("%Y-%m-%dT%H:%M"),
            "end_hour": (first_hour + timedelta(hours=hours - 1)).strftime("%Y-%m-%dT%H:%M"),
        }
    expected_times = [int(first_hour.timestamp()) + hour * 3600 for hour in range(hours)]
    weather_metrics = WEATHER if historical else WEATHER | PROBABILITY
    sources = [
        (
            "weather",
            "https://archive-api.open-meteo.com/v1/archive"
            if historical
            else "https://api.open-meteo.com/v1/forecast",
            weather_metrics,
            {"wind_speed_unit": "ms"} | ({"models": "era5"} if historical else {}),
            "era5" if historical else "weather_forecast",
            "reanalysis" if historical else "provider_forecast",
        ),
        (
            "air_quality",
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            AIR,
            {"domains": "cams_global"},
            "cams_global",
            "archived_provider_forecast" if historical else "provider_forecast",
        ),
    ]
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / f"{city.city_id}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid4().hex}"
    run_dir.mkdir()
    manifest = {
        "schema_version": 1,
        "status": "collecting",
        "city_id": city.city_id,
        "mode": "historical" if historical else "recent",
        "requested_window": window,
        "is_ground_truth": False,
        "sources": {},
        "unavailable_metrics": ["rain_probability"] if historical else [],
        "attribution": "Open-Meteo, ECMWF ERA5, Copernicus Atmosphere Monitoring Service (CAMS)",
    }
    try:
        rows = []
        for name, url, metrics, extra, product, kind in sources:
            params = common | window | extra | {"hourly": ",".join(metrics)}
            payload = fetch_json(client, url, params, live=not historical)
            retrieved_at = datetime.now(UTC).isoformat()
            write_json(
                run_dir / f"{name}.json",
                {
                    "endpoint": url,
                    "parameters": params,
                    "retrieved_at_utc": retrieved_at,
                    "response": payload,
                },
            )
            rows.extend(normalize(payload, metrics, city, product, kind, retrieved_at))
            timestamps = payload["hourly"]["time"]
            if timestamps != expected_times:
                raise CollectionError(
                    f"{product}: timestamps do not match the requested UTC window"
                )
            manifest["sources"][name] = {
                "product": product,
                "data_kind": kind,
                "hours": len(timestamps),
                "retrieved_at_utc": retrieved_at,
            }
        frame = pd.DataFrame(rows)
        missing = frame["value"].isna().groupby(frame["metric"]).sum().astype(int).to_dict()
        manifest.update(
            {
                "status": "partial" if any(missing.values()) or historical else "complete",
                "rows": len(frame),
                "hours": hours,
                "missing_values": missing,
                "valid_from_utc": frame["valid_at_utc"].min(),
                "valid_until_utc": frame["valid_at_utc"].max(),
            }
        )
        frame.to_csv(run_dir / "hourly.csv.tmp", index=False, encoding="utf-8")
        (run_dir / "hourly.csv.tmp").replace(run_dir / "hourly.csv")
    except (ValueError, OSError) as error:
        manifest.update({"status": "failed", "error": str(error)})
        raise CollectionError(f"{error}; run diagnostics: {run_dir}") from error
    finally:
        write_json(run_dir / "manifest.json", manifest)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/city.toml"))
    parser.add_argument("--output", type=Path, default=Path("data/collection"))
    parser.add_argument("--start", type=date.fromisoformat, help="First inclusive UTC history date")
    parser.add_argument("--end", type=date.fromisoformat, help="Last inclusive UTC history date")
    args = parser.parse_args(argv)
    try:
        validate_window(args.start, args.end)
        city = City.load(args.config)
        with httpx.Client(
            timeout=httpx.Timeout(20, connect=10),
            headers={
                "User-Agent": "Mosamiat/0.1 (educational data collection)",
            },
        ) as client:
            run_dir = collect(city, args.output, client, args.start, args.end)
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (CollectionError, OSError) as error:
        print(f"Collection failed: {error}", file=sys.stderr)
        return 1
    print(
        f"{manifest['status'].upper()}: {manifest['hours']} hours, {manifest['rows']} metric rows"
    )
    print(f"Missing values: {sum(manifest['missing_values'].values())}")
    if manifest["unavailable_metrics"]:
        print(f"Unavailable: {', '.join(manifest['unavailable_metrics'])}")
    print("Source data are provider model outputs/reanalysis, not station observations.")
    print(f"Saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
