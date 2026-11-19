"""Build a resumable hourly reference dataset from historical provider products."""

import argparse
import calendar
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

from mosamiat.collect import (
    AIR,
    PROBABILITY,
    WEATHER,
    City,
    CollectionError,
    collect,
    validate_window,
)
from mosamiat.collect import write_json as _write_json

METRICS = {metric.name: metric.unit for metric in (WEATHER | AIR | PROBABILITY).values()}
FILES = ("weather.json", "air_quality.json", "hourly.csv", "manifest.json")
SCHEMA_VERSION = 1


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    _write_json(temporary, value)
    temporary.replace(path)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def windows(start: date, end: date):
    """Partition inclusive dates at calendar-month boundaries, at most 31 days each."""
    if start > end:
        raise CollectionError("Start must not follow end")
    validate_window(end, end)
    while start <= end:
        last = min(end, start.replace(day=calendar.monthrange(start.year, start.month)[1]))
        yield start, last
        start = last + timedelta(days=1)


def snapshot(city: City, root: Path, client: httpx.Client, start: date, end: date) -> Path:
    """Reuse only a fully written, unchanged snapshot; missing values are valid data."""
    checkpoint = root / f"{start}_{end}.json"
    try:
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        run = root / "raw" / saved["run"]
        if run.parent != root / "raw":
            raise ValueError("Invalid checkpoint path")
        if all(digest(run / name) == saved["sha256"][name] for name in FILES):
            return run
    except (OSError, ValueError, KeyError, TypeError):
        pass
    run = collect(city, root / "raw", client, start, end)
    write_json(
        checkpoint, {"run": run.name, "sha256": {name: digest(run / name) for name in FILES}}
    )
    return run


def hourly_frame(run: Path, start: date, end: date) -> pd.DataFrame:
    frame = pd.read_csv(run / "hourly.csv")
    frame["valid_at_utc"] = pd.to_datetime(frame["valid_at_utc"], utc=True)
    if frame.duplicated(["valid_at_utc", "metric"]).any():
        raise CollectionError(f"Duplicate metric/hour in {run}")
    expected = pd.date_range(str(start), periods=((end - start).days + 1) * 24, freq="h", tz="UTC")
    wide = frame.pivot(index="valid_at_utc", columns="metric", values="value")
    if not wide.index.equals(expected):
        raise CollectionError(f"Unexpected hourly coverage in {run}")
    wide = wide.reindex(columns=METRICS)
    wide.index.name = "valid_at_utc"
    return wide


def backfill(city: City, output: Path, client: httpx.Client, start: date, end: date) -> Path:
    # Validate before creating output; keep memory bounded to one monthly window.
    next(windows(start, end))
    identity = {"schema_version": SCHEMA_VERSION, "city": asdict(city)}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    root = output / f"{city.city_id}_{fingerprint}"
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".lock"
    try:
        lock_stream = lock.open("x")
    except FileExistsError as error:
        raise CollectionError(f"Backfill already locked: {lock}") from error
    try:
        with lock_stream:
            destination = root / f"{start}_{end}"
            destination.mkdir(exist_ok=True)
            manifest = identity | {
                "status": "collecting",
                "start": str(start),
                "end": str(end),
                "is_ground_truth": False,
                "unavailable_metrics": ["rain_probability"],
                "units": METRICS,
                "windows": [],
                "attribution": "Open-Meteo, ECMWF ERA5, Copernicus CAMS",
            }
            write_json(destination / "manifest.json", manifest)
            stats = {name: {"missing": 0, "longest_gap_hours": 0} for name in METRICS}
            streaks = dict.fromkeys(METRICS, 0)
            hours = 0
            try:
                with (destination / "hourly.csv.tmp").open(
                    "w", encoding="utf-8", newline=""
                ) as stream:
                    for first, last in windows(start, end):
                        run = snapshot(city, root, client, first, last)
                        wide = hourly_frame(run, first, last)
                        for name, series in wide.items():
                            missing = series.isna()
                            stats[name]["missing"] += int(missing.sum())
                            for absent in missing:
                                streaks[name] = streaks[name] + 1 if absent else 0
                                stats[name]["longest_gap_hours"] = max(
                                    stats[name]["longest_gap_hours"], streaks[name]
                                )
                        result = wide.assign(city_id=city.city_id, snapshot=run.name)
                        for name in METRICS:
                            result[f"{name}_missing"] = wide[name].isna()
                        result.to_csv(stream, header=hours == 0, date_format="%Y-%m-%dT%H:%M:%SZ")
                        hours += len(wide)
                        manifest["windows"].append(
                            {
                                "start": str(first),
                                "end": str(last),
                                "snapshot": run.name,
                                "missing": {name: int(wide[name].isna().sum()) for name in METRICS},
                            }
                        )
                for item in stats.values():
                    item["missing_pct"] = round(100 * item["missing"] / hours, 6)
                write_json(destination / "quality.json", {"hours": hours, "metrics": stats})
                (destination / "hourly.csv.tmp").replace(destination / "hourly.csv")
                manifest.update(
                    status="partial",
                    hours=hours,
                    rows=hours,
                    sha256={
                        name: digest(destination / name) for name in ("hourly.csv", "quality.json")
                    },
                )
            except (ValueError, OSError) as error:
                manifest.update(status="failed", error=str(error))
                raise CollectionError(f"{error}; diagnostics: {destination}") from error
            finally:
                write_json(destination / "manifest.json", manifest)
            return destination
    finally:
        lock.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/city.toml"))
    parser.add_argument("--output", type=Path, default=Path("data/backfill"))
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    args = parser.parse_args(argv)
    try:
        city = City.load(args.config)
        with httpx.Client(
            timeout=httpx.Timeout(20, connect=10),
            headers={"User-Agent": "Mosamiat/0.1 (educational data collection)"},
        ) as client:
            destination = backfill(city, args.output, client, args.start, args.end)
    except (ValueError, OSError) as error:
        print(f"Backfill failed: {error}", file=sys.stderr)
        return 1
    print(f"Saved to: {destination}")
    print("PARTIAL: historical rain probability unavailable; provider reference data only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
