"""Load validated historical provider snapshots shared by features and EDA."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mosamiat.backfill import FILES, METRICS, digest
from mosamiat.collect import City, CollectionError
from mosamiat.storage import load_snapshot, read_json, snapshots


@dataclass(frozen=True)
class ReferenceDataset:
    rows: pd.DataFrame
    values: pd.DataFrame
    paths: tuple[Path, ...]
    expected_index: pd.DatetimeIndex | None = None

    def provenance(self, dataset: Path) -> dict:
        return {
            "source_manifest_sha256": digest(dataset / "manifest.json"),
            "snapshots": [
                {"name": path.name, "sha256": {name: digest(path / name) for name in FILES}}
                for path in self.paths
            ],
        }

    def protect_output(self, dataset: Path, output: Path) -> None:
        for source in (dataset, *self.paths):
            if output.resolve() == source.resolve() or output.resolve() in source.resolve().parents:
                raise CollectionError("Output must not overwrite source artifacts")


def load_reference(dataset: Path, city: City) -> ReferenceDataset:
    frames, paths = [], []
    for path in snapshots(dataset):
        metadata = read_json(path / "manifest.json")
        if metadata.get("mode") != "historical" or metadata.get("city_id") != city.city_id:
            raise CollectionError("Expected historical snapshots for the configured city")
        frame = load_snapshot(path)
        if (
            not frame.requested_latitude.eq(city.latitude).all()
            or not frame.requested_longitude.eq(city.longitude).all()
            or not frame.city_timezone.eq(city.timezone).all()
        ):
            raise CollectionError("Snapshot coordinates/timezone differ from city configuration")
        frames.append(frame)
        paths.append(path)
    if not frames:
        raise CollectionError("Dataset has no snapshots")
    rows = pd.concat(frames, ignore_index=True)
    if rows.duplicated(["valid_at_utc", "metric"]).any():
        raise CollectionError("Overlapping source revisions need an explicit selection policy")
    values = rows.pivot(index="valid_at_utc", columns="metric", values="value")
    values = values.reindex(columns=METRICS).sort_index()
    metadata = read_json(dataset / "manifest.json")
    if "windows" in metadata:
        expected = pd.date_range(
            pd.Timestamp(metadata["start"], tz="UTC"),
            pd.Timestamp(metadata["end"], tz="UTC") + pd.Timedelta(days=1),
            freq="h",
            inclusive="left",
        )
    else:
        expected = pd.date_range(metadata["valid_from_utc"], metadata["valid_until_utc"], freq="h")
    if expected.empty or not values.index.isin(expected).all():
        raise CollectionError("Source timestamps fall outside the declared dataset range")
    return ReferenceDataset(rows, values, tuple(paths), expected)
