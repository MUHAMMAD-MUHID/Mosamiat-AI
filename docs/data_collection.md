# Step 02: Collect source data for Islamabad

For resumable historical downloads and the unified hourly dataset, see
[Module 02: historical reference dataset](data_dictionary.md).

## Purpose and boundaries

Build and verify one local collector before adding storage services, feature
engineering, model training, agents, or scheduling. Use the existing environment;
this step needs no API keys or additional runtime dependencies.

The initial source is Open-Meteo. Its weather forecast and CAMS air-quality feeds
are provider model outputs, not station observations. ERA5 history is reanalysis.
Keep those distinctions in the saved data. This collector alone does not establish
an observation dataset suitable for evaluating our future forecasting models.

## Process

1. Read Islamabad's configurable city ID, coordinates, and display timezone from
   `configs/city.toml`. Keep requests and stored timestamps in UTC.
2. In recent mode, request the previous 24 and next 24 hourly values from the
   weather and global CAMS APIs. Use one HTTP client and explicit metric units.
3. For an explicit historical date range, request ERA5 weather and archived global
   CAMS data. Limit each invocation to 31 inclusive UTC days, ending at least six
   days before today to allow ERA5 publication. Bulk backfill is a later step.
4. Retain each successful raw response with request parameters and retrieval time.
   Check response shape, units, hourly ordering, finite values, physical bounds,
   and alignment between sources. Keep nulls; do not impute or fabricate values.
5. Write a source-specific long-form CSV and a manifest with source metadata,
   missing-value counts, and collection status. Preserve forecast snapshots in
   separate run directories; never append revisions as if they were observations.
6. Verify offline failure cases and run a small live collection. Stop for review
   before building the next module.

## Source mapping

| Project metric | Recent data | Historical data | Canonical unit |
| --- | --- | --- | --- |
| Temperature | Weather forecast | ERA5 reanalysis | degC |
| Humidity | Weather forecast | ERA5 reanalysis | % |
| Precipitation amount | Weather forecast | ERA5 reanalysis | mm per preceding hour |
| Precipitation probability | Ensemble-derived provider probability | Unavailable in ERA5 | 0-1 |
| Wind speed / direction | Weather forecast | ERA5 reanalysis | m/s / degrees |
| AQI | Global CAMS model | Archived global CAMS model | US AQI |
| UV index | Global CAMS model | Archived global CAMS model | dimensionless |
| PM2.5, PM10, CO, NO2, SO2, O3 | Global CAMS model | Archived global CAMS model | ug/m3 |

The probability refers to precipitation exceeding 0.1 mm in the preceding hour;
it is not a measured historical outcome or a calibrated probability from our model.
The eventual rain-probability model needs an event label derived from suitable
precipitation reference data. That label is outside this collection step.

CAMS global data has a coarser underlying grid and native temporal resolution than
the returned hourly arrays. Hourly API output is not evidence of hourly station
measurements. Store returned grid coordinates as well as requested coordinates.

## Minimal implementation

- `configs/city.toml`: city settings, initially Islamabad.
- `src/mosamiat/collect.py`: source requests, validation, local persistence, CLI.
- `tests/test_collect.py`: deterministic tests using HTTPX mock responses.
- Update packaging so the environment exposes `mosamiat-collect`.

## Commands

Run from the repository root after syncing the environment:

```powershell
& ./.venv/Scripts/python.exe -m mosamiat.collect
& ./.venv/Scripts/python.exe -m mosamiat.collect --start 2026-10-01 --end 2026-10-02
& ./.venv/Scripts/python.exe -m mosamiat.collect --help
```

Optional `--config` selects another city TOML file and `--output` changes the local
destination. Defaults are `configs/city.toml` and `data/collection`, respectively.
History dates are inclusive UTC dates. Both dates must be provided together.

Each run saves raw JSON responses, `hourly.csv`, and `manifest.json` under a unique
directory in `data/collection/`. All data remain excluded from Git. On a source or
validation failure, keep available raw evidence and a failed manifest, return a
nonzero exit status, and do not publish a partial CSV as a successful result.
Null values are retained and reported as a partial collection. Historical runs
also report the unavailable precipitation probability explicitly.

CSV identity within a snapshot is `(city_id, source, product, valid_at_utc, metric)`.
Different runs are distinct forecast revisions. `retrieved_at_utc` is collection
time, not the provider's original model issuance time. Training must not assume
that a retrospectively retrieved value was available at its valid timestamp.

## Acceptance checks

- Configurable coordinates and UTC timestamps; no embedded credentials.
- All eight forecast fields covered in recent mode, plus pollutant concentrations.
- Raw provenance and US AQI scale retained; probability converted from % to 0-1.
- Missing values, invalid units/ranges, unequal arrays, and timestamp mismatches
  handled explicitly.
- Bounded timeouts and at most three attempts for transient transport/server errors
  or rate limiting; other client errors fail immediately.
- Failed runs preserve diagnostics without a published `hourly.csv`.
- Offline tests and Ruff pass; recent and historical live samples are inspected.

## Evidence and remaining decisions

Verified on 16 November 2026:

- Offline tests: 45 passed, covering source failures, retries, malformed data,
  missingness, provenance, date windows, and snapshot preservation.
- Ruff lint and formatting checks passed; installed dependencies are compatible.
- Recent live collection: 48 hourly timestamps, 672 metric rows, no null values.
- Historical live collection for 1-2 October 2026: 48 hourly timestamps, 624 metric
  rows, no null values in available fields. Status is explicitly `partial` because
  historical precipitation probability is unavailable.
- Inspected both saved CSVs for required fields, distinct keys, probability bounds,
  UTC timestamps, and source labels. Verified that data and references stay ignored
  by Git. No commits or pushes were performed.

Tests run with `& ./.venv/Scripts/pytest.exe -q`. If a restricted Windows session
cannot access its default temporary directory, use an accessible temporary
directory, such as `--basetemp .cache/pytest-run`. pytest removes existing contents
of its specified base temporary directory, so reserve it exclusively for tests.

These live samples are a small feasibility check, not a multi-year coverage audit.

Before final training: confirm the mentor accepts any proposed reanalysis/model
reference data, investigate measured weather/AQI sources, audit seasonal coverage,
and agree the ground-truth and rain-event definitions. Hopsworks persistence,
idempotent dataset upserts, bulk backfill, and hourly automation follow later.

## References and attribution

Data: Open-Meteo, ECMWF ERA5, and Copernicus Atmosphere Monitoring Service (CAMS).
Preserve attribution in downstream reports and any public dashboard. Free API
access is for noncommercial use, subject to the provider's limits and terms.

- [Weather forecast API](https://open-meteo.com/en/docs)
- [Historical weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [Air-quality API and CAMS attribution](https://open-meteo.com/en/docs/air-quality-api)
- [Usage terms](https://open-meteo.com/en/terms)
