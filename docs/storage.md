# Module 03: Hopsworks provider-data storage

This implements the first storage slice agreed for the current collector:
one versioned provider-data feature group, upload and readback verification.
The roadmap's feature, forecast, ledger, agent and drift tables remain later work
when their producing modules and contracts exist. No cloud tables have been
created or verified yet; a Hopsworks account and credentials are required.

## Setup

Install the optional storage dependencies from the project root:

Setup is currently pending: repeated PyPI DNS/TLS timeouts prevented completing
SDK installation and updating `uv.lock`. The optional requirements are declared
in `pyproject.toml`; the existing lockfile has not been manually fabricated.
Once connectivity is available, run the command below to resolve, lock and install
them before the cloud test. Until then, `uv sync --locked` will reject the changed
dependency declaration. Local collection commands and storage dry runs still work
in the existing environment.

```powershell
& ./.tools/bootstrap/bin/uv.exe sync --extra storage
```

Copy `.env.example` to `.env` if it does not already exist. Set these values in
that local file (never paste an API key into chat or commit it):

```dotenv
HOPSWORKS_API_KEY=your-key
HOPSWORKS_PROJECT=your-project-name
HOPSWORKS_HOST=your-instance-hostname
```

Use the hostname supplied by your Hopsworks instance, without a scheme or path.
The SDK is constrained to the documented 4.8 client family; confirm compatibility
with your instance before the live test. Existing environment variables override
`.env`. The file is loaded only when connecting. `.env` is excluded by Git.

## Commands

Pass an individual collector run directory or a completed backfill directory
containing its `manifest.json` (not the parent data directory):

```powershell
& ./.venv/Scripts/python.exe -m mosamiat.storage PATH_TO_DATASET --dry-run
& ./.venv/Scripts/python.exe -m mosamiat.storage PATH_TO_DATASET --verify
```

`--dry-run` validates without connecting or importing Hopsworks. `--verify`
uploads, then reads back each snapshot and compares all fields and row counts.
Repeat the verified upload to test idempotence against the real service.
The command `mosamiat-store` is equivalent after environment sync.

The uploader processes one source snapshot at a time and shares one connection
and feature-group handle. It validates raw arrays using the existing collector,
preserves nulls, and verifies backfill artifact hashes before upload. A failed
later window may leave earlier windows stored; rerun safely using the same keys.

## Schema and update rules

Feature group: `mosamiat_provider_data`, version `1`, offline Hudi storage.
Primary key: `record_id`. Event time: `valid_at_utc` (UTC timestamp).
Online storage and automatic statistics are disabled for this historical slice.

Each row retains the collector's city, requested and returned grid coordinates,
source, product, data kind, metric, value, unit, quality status, AQI scale and
retrieval time. `is_ground_truth` is always false. The raw JSON stays locally;
the remote table holds validated normalized source records, not raw response blobs.

Record IDs are SHA-256 hashes of city, requested coordinates, source/product/data
kind, valid time, retrieval time and metric. Reuploading the same snapshot uses
the same IDs and `upsert` updates those records. New retrieval times represent
separate revisions and keep separate IDs. Do not count them as independent
observations or merge them into training data without a revision-selection rule.

Historical rain probability has no source records; the missing column in the
derived backfill does not become a fabricated measurement in Hopsworks.
Explicit float conversion ensures even an all-null metric uses a numeric column.
Writes wait for the offline job; existing key mismatches fail, and the SDK
validates the remaining schema. No destructive overwrite option is used.

Readback filters by city, valid-time range and retrieval-time range, then checks
record IDs, duplicate counts, timestamps and values. It reads one snapshot's
range, rather than the entire feature group. A readback mismatch fails the command.
Remote failures print the exception type rather than SDK exception text that could
contain credentials. Diagnose account permissions/network settings locally if needed.

## Completion gates

Local tests cover repeatable keys, revision separation, missing values, invalid
metadata/units, corrupt backfills, SDK write arguments, key incompatibility,
readback mismatches and redaction. Cloud authentication, instance compatibility,
actual storage upsert behavior and offline readback remain unverified until the
user supplies credentials. Training-source acceptance remains a separate decision.

Local verification on 19 November 2026: 62 tests passed; Ruff lint and formatting
passed. The storage dry run revalidated 624 records from the existing live
two-day backfill. These checks do not establish SDK/server compatibility or prove
remote idempotence; the credentialed `--verify` run remains the final gate.

References used for the adapter:

- https://docs.hopsworks.ai/4.8/python-api/hsfs/feature_group/
- https://docs.hopsworks.ai/4.8/python-api/hsfs/feature_store/
