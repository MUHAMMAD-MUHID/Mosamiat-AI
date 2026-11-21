# Module 03: Hopsworks provider-data storage

This implements the first storage slice agreed for the current collector:
one versioned provider-data feature group, upload and readback verification.
The roadmap's feature, forecast, ledger, agent and drift tables remain later work
when their producing modules and contracts exist. Credentialed verification
status is recorded under Completion gates below.

## Setup

Install the optional storage dependencies from the project root:

The storage dependencies are installed and `uv.lock` is updated. On 20 November
2026 the resolved SDK was Hopsworks 4.8.8, including the Python profile. Windows
successfully built the required twofish dependency. A locked offline sync passed.

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

Use the hostname supplied by your Hopsworks instance. An HTTPS root URL is also
accepted and normalized; project paths, credentials in URLs and query strings
are rejected. Certificates are materialized under the ignored project-local
`.cache/hopsworks/` directory because the SDK's `/tmp` default fails on Windows.
Certificate directories are isolated by process so one client's cleanup cannot
remove another running client's credentials.
The adapter reuses the project's PEM certificates downloaded by the SDK at login.
This avoids an unnecessary JKS conversion that fails on Java 21 keystores with
released pyjks 20.0.0, and also avoids that conversion's hardcoded `/tmp` output.
Only the authenticated project's own keystore/truststore paths are intercepted;
unrelated connector credentials retain the SDK's original handling. This narrow
compatibility adapter depends on the pinned 4.8 client internals and must be
reviewed when upgrading the SDK. Upstream issue and proposed fix:
https://github.com/kurtbrose/pyjks/issues/79 and
https://github.com/kurtbrose/pyjks/pull/82.
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
range, rather than the entire feature group. Retrieval filter bounds are rounded
outward to whole seconds because server-side filters can lose fractional precision.
Stored timestamps are not rounded: exact record IDs and field comparisons still
apply after the query. A readback mismatch fails the command.
Remote failures print the exception type rather than SDK exception text that could
contain credentials. Diagnose account permissions/network settings locally if needed.

## Completion gates

Local tests cover repeatable keys, revision separation, missing values, invalid
metadata/units, corrupt backfills, SDK write arguments, key incompatibility,
readback mismatches, fractional timestamp bounds, certificate handling and
redaction. On 20 November 2026 all 84 project tests passed; Ruff lint and format
checks passed. The final storage-only regression run passed all 17 storage tests.

Credentialed authentication, offline materialization and full-field readback of
624 records from the existing 1–2 October Islamabad backfill passed on 20 November
2026. A separate bounded read confirmed exactly 624 unique record IDs, with all
240 ERA5 and 384 CAMS records and unchanged retrieval timestamps. Both final
upload/readback passes verified all 624 records, including a repeated upsert with
no duplicate IDs or changed fields. The sanitized result is recorded in the
local ignored `data/hopsworks-smoke.json`.
Training-source acceptance remains a separate decision; this test establishes
provider-data storage, not station observations or model readiness.

References used for the adapter:

- https://docs.hopsworks.ai/4.8/python-api/hsfs/feature_group/
- https://docs.hopsworks.ai/4.8/python-api/hsfs/feature_store/
