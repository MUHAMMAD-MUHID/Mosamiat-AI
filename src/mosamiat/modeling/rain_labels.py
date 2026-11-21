"""Collect separate ERA5 liquid-rain labels for validated historical snapshots."""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from mosamiat.backfill import digest, write_json
from mosamiat.collect import City, CollectionError, Metric, fetch_json, normalize
from mosamiat.reference import ReferenceDataset, load_reference
from mosamiat.storage import read_json

FIELD = {"rain": Metric("rain_mm", "mm", "mm", 0)}
ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"


def events(document: dict, reference: ReferenceDataset, city: City) -> pd.Series:
    """Verify the source revision, grid and UTC hours before deriving binary labels."""
    chunks = document.get("windows", [])
    if document.get("schema_version") != 1 or len(chunks) != len(reference.paths):
        raise CollectionError("Unexpected rain label schema or snapshot count")
    series = []
    for chunk, path in zip(chunks, reference.paths, strict=True):
        weather = read_json(path / "weather.json")
        expected_params = weather["parameters"] | {"hourly": "rain"}
        response = chunk["response"]
        if (
            chunk["weather_sha256"] != digest(path / "weather.json")
            or chunk["endpoint"] != ENDPOINT
            or chunk["parameters"] != expected_params
            or expected_params["models"] != "era5"
            or any(response[key] != weather["response"][key] for key in ("latitude", "longitude"))
            or response["hourly"]["time"] != weather["response"]["hourly"]["time"]
        ):
            raise CollectionError("Rain labels do not match the weather source, grid or hours")
        rows = normalize(response, FIELD, city, "era5", "reanalysis", chunk["retrieved_at_utc"])
        table = pd.DataFrame(rows)
        rain = pd.Series(
            table.value.to_numpy(dtype=float), index=pd.to_datetime(table.valid_at_utc, utc=True)
        )
        series.append(rain.gt(0).astype(float).where(rain.notna()))
    result = pd.concat(series).sort_index().rename("rain_probability")
    if not result.index.equals(reference.values.index):
        raise CollectionError("Rain labels must align with historical coverage")
    return result


def collect(dataset: Path, output: Path, city: City) -> None:
    reference = load_reference(dataset, city)
    if output.suffix.lower() != ".json":
        raise CollectionError("Rain label output must be a JSON file")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + ".lock")
    lock.touch(exist_ok=False)
    try:
        if output.is_file():
            events(read_json(output), reference, city)
            return
        chunks = []
        with httpx.Client(timeout=30) as client:
            for path in reference.paths:
                params = read_json(path / "weather.json")["parameters"] | {"hourly": "rain"}
                response = fetch_json(client, ENDPOINT, params)
                chunks.append(
                    {
                        "weather_sha256": digest(path / "weather.json"),
                        "endpoint": ENDPOINT,
                        "parameters": params,
                        "retrieved_at_utc": datetime.now(UTC).isoformat(),
                        "response": response,
                    }
                )
        document = {"schema_version": 1, "windows": chunks}
        events(document, reference, city)
        write_json(output, document)
    finally:
        lock.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--city-config", type=Path, default=Path("configs/city.toml"))
    args = parser.parse_args()
    try:
        collect(args.dataset, args.output, City.load(args.city_config))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Rain collection failed: {error}", file=sys.stderr)
        return 1
    print(f"Verified liquid rain label snapshots: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
