"""Read a city's verified forecast and its most recent published weather report."""

import json
import shutil
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from urllib.parse import quote
from zipfile import BadZipFile, ZipFile

import httpx
import pandas as pd

from mosamiat.agents.alerts import evaluate, forecast_hours
from mosamiat.backfill import digest
from mosamiat.checkpoints import LEDGER_NAME
from mosamiat.collect import City, CollectionError
from mosamiat.storage import connection_settings

FIELDS = {
    "temperature_c": ("Temperature", "°C"),
    "aqi_value": ("Air quality", "US AQI"),
    "precipitation_mm": ("Precipitation", "mm"),
    "rain_probability": ("Rain chance", "%"),
    "humidity_pct": ("Humidity", "%"),
    "uv_index": ("UV index", "index"),
    "wind_speed_mps": ("Wind speed", "m/s"),
    "wind_direction_deg": ("Wind direction", "°"),
}
_CACHE_LOCK = Lock()


@dataclass
class Outlook:
    run_id: str
    issued: pd.Timestamp
    verified: pd.Timestamp
    hours: pd.DataFrame
    alerts: list[dict]
    report: str
    report_issued: pd.Timestamp | None


def read_outlook(path: Path, city: City) -> Outlook:
    uri = path.resolve().as_uri() + "?mode=ro"
    verified = """WITH verified AS (
        SELECT b.content, b.has_report, r.payload,
               json_extract(r.payload, '$.issued_at_utc') AS issued_at_utc,
               coalesce(min(CASE WHEN e.status IN ('forecast_completed', 'verified') THEN e.at_utc END),
                        max(e.at_utc)) AS verified_at
        FROM runs r JOIN events e USING(run_id)
        LEFT JOIN bulletins b ON b.run_id = r.run_id AND EXISTS (
            SELECT 1 FROM events v WHERE v.run_id = r.run_id
            AND v.status IN ('bulletin_verified', 'completed')
        )
        WHERE json_extract(r.payload, '$.city_id') = ?
          AND e.status IN ('forecast_completed', 'verified', 'completed')
        GROUP BY r.run_id
    ) """
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.execute("BEGIN")
        entry = db.execute(
            verified + "SELECT payload, content, verified_at FROM verified "
            "ORDER BY issued_at_utc DESC LIMIT 1",
            (city.city_id,),
        ).fetchone()
        if entry is None:
            raise CollectionError("No verified forecast is available for this city")
        payload = json.loads(entry[0])
        bulletin = json.loads(entry[1]) if entry[1] else None
        report_row = db.execute(
            verified + "SELECT content FROM verified WHERE has_report=1 AND issued_at_utc<=? "
            "ORDER BY issued_at_utc DESC LIMIT 1",
            (city.city_id, payload["issued_at_utc"]),
        ).fetchone()
    if payload["city_id"] != city.city_id or (
        bulletin is not None and bulletin["run_id"] != payload["run_id"]
    ):
        raise CollectionError("Forecast and bulletin identifiers differ")
    report = json.loads(report_row[0]) if report_row else {}
    forecast = forecast_hours(payload)
    hours = forecast[list(FIELDS)].tz_convert(city.timezone)
    hours["rain_probability"] *= 100
    return Outlook(
        payload["run_id"],
        pd.Timestamp(payload["issued_at_utc"]).tz_convert(city.timezone),
        pd.Timestamp(entry[2]).tz_convert(city.timezone),
        hours,
        bulletin["alerts"] if bulletin else evaluate(forecast),
        report.get("report", ""),
        pd.Timestamp(report["issued_at_utc"]).tz_convert(city.timezone) if report else None,
    )


def _read_bundle(bundle, city):
    if digest(bundle) != LEDGER_NAME.fullmatch(bundle.name)[2]:
        raise CollectionError("Dashboard checkpoint checksum mismatch")
    with TemporaryDirectory() as tmp, ZipFile(bundle) as archive:
        if archive.namelist() != ["ledger.sqlite3"]:
            raise CollectionError("Unexpected dashboard checkpoint contents")
        path = Path(tmp) / "ledger.sqlite3"
        with archive.open("ledger.sqlite3") as source, path.open("wb") as dest:
            shutil.copyfileobj(source, dest)
        return read_outlook(path, city)


def _cloud_outlook(city, cache):
    settings = connection_settings()
    with httpx.Client(
        base_url=f"https://{settings['HOST']}/hopsworks-api/api/project/",
        headers={"Authorization": f"ApiKey {settings['API_KEY']}"},
        timeout=httpx.Timeout(15, connect=5),
    ) as client:
        return _fetch_checkpoint(client, settings["PROJECT"], city, cache)


def _fetch_checkpoint(client, project, city, cache):
    project_id = int(
        client.get(f"getProjectInfo/{quote(project, safe='')}")
        .raise_for_status()
        .json()["projectId"]
    )
    folder = f"Resources/mosamiat_{city.city_id}/ledgers"
    dataset = f"{project_id}/dataset"
    listing = (
        client.get(
            f"{dataset}/{quote(folder, safe='')}",
            params={
                "action": "listing",
                "expand": "inodes",
                "sort_by": "ID:desc",
                "limit": 100,
            },
        )
        .raise_for_status()
        .json()
    )
    names = [item["attributes"]["name"] for item in listing.get("items", [])]
    name = max((n for n in names if LEDGER_NAME.fullmatch(n)), default=None)
    if name is None:
        raise CollectionError("No published ledger in Hopsworks")
    bundle = cache / name
    if bundle.is_file() and digest(bundle) == LEDGER_NAME.fullmatch(name)[2]:
        return _read_bundle(bundle, city)
    with TemporaryDirectory(dir=cache) as tmp:
        downloaded = Path(tmp) / name
        remote = quote(f"{folder}/{name}", safe="")
        with client.stream(
            "GET", f"{dataset}/download/with_auth/{remote}", params={"type": "DATASET"}
        ) as response:
            response.raise_for_status()
            with downloaded.open("wb") as dest:
                for chunk in response.iter_bytes():
                    dest.write(chunk)
        result = _read_bundle(downloaded, city)
        downloaded.replace(bundle)
    for prior in cache.glob("ledger-*.zip"):
        if LEDGER_NAME.fullmatch(prior.name) and prior.name < name:
            prior.unlink(missing_ok=True)
    return result


def load_outlook(city, cache=Path("data/dashboard"), local=Path("data/forecasts/ledger.sqlite3")):
    cache = cache / city.city_id
    with _CACHE_LOCK:
        try:
            cache.mkdir(parents=True, exist_ok=True)
            return _cloud_outlook(city, cache), "Hopsworks"
        except Exception:
            # Provider exceptions can contain credentials. Show only source status.
            for bundle in sorted(cache.glob("ledger-*.zip"), reverse=True):
                if LEDGER_NAME.fullmatch(bundle.name):
                    try:
                        return _read_bundle(bundle, city), "Cached Hopsworks checkpoint"
                    except (OSError, ValueError, sqlite3.Error, KeyError, BadZipFile):
                        continue
            if local.is_file():
                return read_outlook(local, city), "Local verified ledger"
            raise CollectionError("No verified forecast is available") from None
