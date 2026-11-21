"""Upload validated provider snapshots to a versioned Hopsworks feature group."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from mosamiat.backfill import digest
from mosamiat.collect import AIR, PROBABILITY, WEATHER, City, CollectionError, normalize
from mosamiat.hopsworks_compat import configure_project_certificates

GROUP = "mosamiat_provider_data"
VERSION = 1


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CollectionError(f"Expected JSON object: {path}")
    return value


def snapshots(path: Path):
    """Accept a collector run or a published backfill; never scan arbitrary directories."""
    manifest = read_json(path / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("status") not in {"complete", "partial"}:
        raise CollectionError("Only finished schema-version-1 datasets can be uploaded")
    if manifest.get("is_ground_truth") is not False:
        raise CollectionError("Expected explicitly labeled provider reference data")
    if "windows" not in manifest:
        yield path
        return
    for name in ("hourly.csv", "quality.json"):
        if digest(path / name) != manifest["sha256"][name]:
            raise CollectionError(f"Backfill integrity check failed: {name}")
    for window in manifest["windows"]:
        name = window["snapshot"]
        if not isinstance(name, str) or Path(name).name != name:
            raise CollectionError("Invalid snapshot name")
        run = path.parent / "raw" / name
        checkpoint = read_json(path.parent / f"{window['start']}_{window['end']}.json")
        if checkpoint["run"] != name or any(
            digest(run / file) != checkpoint["sha256"][file]
            for file in ("weather.json", "air_quality.json", "hourly.csv", "manifest.json")
        ):
            raise CollectionError(f"Snapshot integrity check failed: {name}")
        yield run


def load_snapshot(path: Path) -> pd.DataFrame:
    """Revalidate raw source arrays instead of trusting an editable normalized CSV."""
    manifest = read_json(path / "manifest.json")
    if (
        manifest.get("status") not in {"complete", "partial"}
        or manifest.get("is_ground_truth") is not False
    ):
        raise CollectionError("Snapshot is not a finished provider dataset")
    if manifest.get("schema_version") != 1 or manifest.get("mode") not in {"recent", "historical"}:
        raise CollectionError("Unsupported snapshot schema or mode")
    historical = manifest["mode"] == "historical"
    csv = pd.read_csv(path / "hourly.csv", nrows=1)
    if csv.empty:
        raise CollectionError("Empty snapshot")
    row = csv.iloc[0]
    city = City(
        manifest["city_id"],
        manifest["city_id"],
        float(row.requested_latitude),
        float(row.requested_longitude),
        row.city_timezone,
    )
    rows = []
    for name, metrics, product, kind, endpoint in [
        (
            "weather",
            WEATHER if historical else WEATHER | PROBABILITY,
            "era5" if historical else "weather_forecast",
            "reanalysis" if historical else "provider_forecast",
            "https://archive-api.open-meteo.com/v1/archive"
            if historical
            else "https://api.open-meteo.com/v1/forecast",
        ),
        (
            "air_quality",
            AIR,
            "cams_global",
            "archived_provider_forecast" if historical else "provider_forecast",
            "https://air-quality-api.open-meteo.com/v1/air-quality",
        ),
    ]:
        raw = read_json(path / f"{name}.json")
        params = raw["parameters"]
        if (
            raw["endpoint"] != endpoint
            or params["latitude"] != city.latitude
            or params["longitude"] != city.longitude
            or any(params.get(key) != value for key, value in manifest["requested_window"].items())
            or (historical and name == "weather" and params.get("models") != "era5")
            or (name == "air_quality" and params.get("domains") != "cams_global")
        ):
            raise CollectionError("Source request does not match snapshot metadata")
        rows.extend(
            normalize(raw["response"], metrics, city, product, kind, raw["retrieved_at_utc"])
        )
    frame = pd.DataFrame(rows)
    expected = pd.date_range(manifest["valid_from_utc"], manifest["valid_until_utc"], freq="h")
    for _, group in frame.groupby("metric"):
        if not pd.DatetimeIndex(pd.to_datetime(group.valid_at_utc, utc=True)).equals(expected):
            raise CollectionError("Snapshot coverage differs from manifest")
    if len(frame) != manifest["rows"] or len(expected) != manifest["hours"]:
        raise CollectionError("Snapshot counts differ from manifest")
    for name in ("valid_at_utc", "retrieved_at_utc"):
        frame[name] = pd.to_datetime(frame[name], utc=True, format="ISO8601", errors="raise")
        if frame[name].isna().any():
            raise CollectionError(f"Missing timestamp: {name}")
    identity = [
        "city_id",
        "requested_latitude",
        "requested_longitude",
        "source",
        "product",
        "data_kind",
        "valid_at_utc",
        "retrieved_at_utc",
        "metric",
    ]
    frame["record_id"] = [
        hashlib.sha256(
            json.dumps(list(values), default=str, separators=(",", ":")).encode()
        ).hexdigest()
        for values in frame[identity].itertuples(index=False, name=None)
    ]
    for name in (
        "value",
        "requested_latitude",
        "requested_longitude",
        "grid_latitude",
        "grid_longitude",
    ):
        frame[name] = frame[name].astype(float)
    frame["is_ground_truth"] = False
    if frame.record_id.duplicated().any():
        raise CollectionError("Duplicate source records")
    return frame


def hostname(value: str) -> str:
    """Accept a bare hostname or an HTTPS instance URL without a path."""
    parsed = urlsplit(value if "://" in value else f"https://{value}")
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path.strip("/")
        or parsed.query
        or parsed.fragment
        or parsed.port not in (None, 443)
    ):
        raise CollectionError("HOPSWORKS_HOST must be an instance hostname or HTTPS root URL")
    return parsed.hostname


def connect():
    # .env is local and ignored; existing environment values take precedence.
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env", override=False)
    settings = {
        name: os.environ.get(f"HOPSWORKS_{name}", "").strip()
        for name in ("API_KEY", "PROJECT", "HOST")
    }
    if not all(settings.values()):
        raise CollectionError("Set HOPSWORKS_API_KEY, HOPSWORKS_PROJECT and HOPSWORKS_HOST in .env")
    host = hostname(settings["HOST"])
    # The SDK defaults to /tmp, which does not exist on many Windows machines.
    identity = hashlib.sha256(f"{host}/{settings['PROJECT']}".encode()).hexdigest()[:16]
    cert_folder = Path.cwd() / ".cache" / "hopsworks" / f"{identity}-{os.getpid()}"
    import hopsworks

    project = hopsworks.login(
        api_key_value=settings["API_KEY"],
        project=settings["PROJECT"],
        host=host,
        cert_folder=str(cert_folder.resolve()),
    )
    configure_project_certificates()
    return project.get_feature_store()


class Repository:
    def __init__(self, store):
        self.store = store
        self.group = None

    def upsert(self, frame: pd.DataFrame) -> None:
        if self.group is None:
            self.group = self.store.get_or_create_feature_group(
                name=GROUP,
                version=VERSION,
                primary_key=["record_id"],
                event_time="valid_at_utc",
                online_enabled=False,
                time_travel_format="HUDI",
                statistics_config=False,
                description="Open-Meteo provider forecasts/reanalysis; not station observations.",
            )
            if (
                list(self.group.primary_key) != ["record_id"]
                or self.group.event_time != "valid_at_utc"
            ):
                raise CollectionError("Existing feature group has incompatible keys")
        self.group.insert(
            frame, operation="upsert", wait=True, write_options={"wait_for_job": True}
        )

    def verify(self, expected: pd.DataFrame) -> None:
        """Read only the inserted snapshot and compare every stored column and row."""
        fg = self.group
        if fg is None:
            raise CollectionError("Upload a snapshot before verifying")
        first = expected.iloc[0]
        query = fg.select_all().filter(
            (fg.city_id == first.city_id)
            & (fg.valid_at_utc >= expected.valid_at_utc.min().to_pydatetime())
            & (fg.valid_at_utc <= expected.valid_at_utc.max().to_pydatetime())
            # Server-side timestamp filters can lose sub-millisecond precision.
            # Widen only the query bounds; record IDs and stored values stay exact.
            & (fg.retrieved_at_utc >= expected.retrieved_at_utc.min().floor("s").to_pydatetime())
            & (fg.retrieved_at_utc <= expected.retrieved_at_utc.max().ceil("s").to_pydatetime())
        )
        actual = query.read()
        actual = actual[actual.record_id.isin(expected.record_id)].copy()
        for name in ("valid_at_utc", "retrieved_at_utc"):
            actual[name] = pd.to_datetime(actual[name], utc=True)
        try:
            pd.testing.assert_frame_equal(
                expected.set_index("record_id").sort_index().sort_index(axis=1),
                actual.set_index("record_id").sort_index().sort_index(axis=1),
                check_dtype=False,
                check_exact=False,
                rtol=1e-9,
                atol=1e-9,
            )
        except AssertionError as error:
            raise CollectionError("Hopsworks readback differs from uploaded snapshot") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Collector run or completed backfill directory")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate locally without credentials"
    )
    parser.add_argument("--verify", action="store_true", help="Read back and compare each upload")
    args = parser.parse_args(argv)
    try:
        repo = None
        count = 0
        for path in snapshots(args.path):
            frame = load_snapshot(path)
            if not args.dry_run:
                if repo is None:
                    repo = Repository(connect())
                repo.upsert(frame)
                if args.verify:
                    repo.verify(frame)
            count += len(frame)
        if count == 0:
            raise CollectionError("Dataset contains no source snapshots")
    except Exception as error:
        # SDK exceptions may contain authentication details. Do not echo them.
        message = str(error) if isinstance(error, CollectionError) else type(error).__name__
        print(f"Storage failed: {message}", file=sys.stderr)
        return 1
    print(f"{'Validated' if args.dry_run else 'Uploaded'} {count} provider records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
