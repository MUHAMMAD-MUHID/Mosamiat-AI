# Historical reference dataset (Module 02)

## Source contract

This module builds a local research/reference dataset, not verified station
observations. Source acceptance for supervised training remains unresolved.
The existing two-day collection evidence does not establish multi-year coverage.
Do not describe this module as completing the observation-source requirement.

| Output metric | Unit | Historical product | Intended interpretation |
| --- | --- | --- | --- |
| temperature_c | degC | ERA5 | Reanalysis temperature at 2 m |
| humidity_pct | % | ERA5 | Reanalysis relative humidity at 2 m |
| precipitation_mm | mm | ERA5 | Preceding-hour precipitation, including snow |
| wind_speed_mps | m/s | ERA5 | Reanalysis wind speed at 10 m |
| wind_direction_deg | degree | ERA5 | Reanalysis wind direction at 10 m |
| aqi_value | US_AQI | CAMS global | Model-derived US AQI |
| uv_index | dimensionless | CAMS global | Model UV index |
| pm2_5_ug_m3, pm10_ug_m3 | ug/m3 | CAMS global | Model particulate concentrations |
| carbon_monoxide_ug_m3, nitrogen_dioxide_ug_m3 | ug/m3 | CAMS global | Model gas concentrations |
| sulphur_dioxide_ug_m3, ozone_ug_m3 | ug/m3 | CAMS global | Model gas concentrations |
| rain_probability | 0–1 | Unavailable | Always missing in this historical dataset |

Weather comes exclusively from the existing ERA5 archive request; air quality and
UV come exclusively from its CAMS global request. No fallback source is mixed in.
Each wide row carries a snapshot name linking to the raw files and normalized
long-form source rows, which retain coordinates, retrieval time, source product,
data kind, units and AQI scale. Retrieval time is not model issuance time.
Hourly API values do not imply hourly station measurements.

The provider describes ERA5 coverage from 1940, but the usable joint date range
also depends on CAMS archive availability and each metric. The collector requires
the end date to be at least six UTC days old and validates every response against
the requested hours. No universal earliest joint date is asserted or hardcoded.
Requests for unavailable history may fail or return nulls; inspect the report.
Provider documentation could not be re-fetched during this implementation.

## Run and resume

From the repository root:

```powershell
& ./.venv/Scripts/python.exe -m mosamiat.backfill --start 2026-10-01 --end 2026-10-02
```

Both dates are inclusive UTC dates. Optional `--config` and `--output` behave like
the collector; the default output is `data/backfill`. After environment sync the
equivalent command is `mosamiat-backfill`.

Downloads are sequential calendar-month windows (at most 31 days) using one HTTP
client and the collector's bounded retries. Memory usage is bounded to a window;
the combined CSV is streamed. Keep bulk requests within current provider quotas.

Output layout:

```text
data/backfill/<city>_<configuration hash>/
  raw/<snapshot>/                  # Original collector outputs, including failures
  <window start>_<window end>.json # Checkpoint with hashes of all four raw artifacts
  <requested start>_<requested end>/
    hourly.csv                    # One row per UTC hour and city
    quality.json                  # Missing counts, percentages and longest streaks
    manifest.json                 # Status, windows, units, attribution, output hashes
```

Unchanged successful windows are reused, including snapshots with genuine nulls.
Missing or corrupted checkpoint artifacts cause that window to be downloaded
again. Failed windows have no successful checkpoint. A changed city configuration
or schema version gets its own cache. Exact window matches are reused across
requests; differently clipped edge windows are fetched separately. Existing
`data/collection` snapshots are not automatically imported.

Duplicate policy: one validated snapshot per requested window; raw revisions stay
separate. Windows in a dataset never overlap. Duplicate metric/hour rows and
off-hour timestamps are errors, not averaged or silently resolved. Rebuilding a
date range replaces its derived CSV rather than appending. Different requested
ranges are separate datasets; do not concatenate overlapping ranges blindly.

## Missing values and publication

All metrics preserve nulls, with a `<metric>_missing` boolean column. No forward
fill or interpolation is applied: there is no justified per-metric filling rule
yet. No imputer is fitted, so no training/validation boundary can be crossed.
Model training must later choose suitable labels and exclude unavailable ones;
do not substitute provider rain probabilities for measured rain outcomes. A rain
event derived from ERA5 would still be a reanalysis-based label and needs an
explicit event threshold and acceptance decision.

Quality percentages use all requested hours, and longest missing streaks continue
across monthly boundaries. The manifest also records missing counts per window;
source identity is fixed by metric in the table above. Historical runs have
`status: partial` because rain probability is unavailable, even if all 13 other
metrics have values. `is_ground_truth` remains false.

Consumers must require a finished `partial` manifest and matching output hashes.
During a rebuild or failure an older CSV may remain on disk but is not the current
published result. Files use temporary writes and rename; the final manifest is
the commit marker. A lock prevents concurrent writers for the same city/config.
After a force-killed process, remove its `.lock` only after confirming that no
backfill process is running, then rerun the same command. Graceful errors release
the lock and preserve diagnostics.

The checkpoint cache preserves the first successful snapshot; nulls are not
automatically refreshed forever. To deliberately fetch a new revision, use a
different `--output` directory and retain both datasets for comparison.

## References

Verification on 19 November 2026: all 53 tests passed, including interrupted
downloads, cache corruption, leap-year windows, duplicate/gap rejection, city
cache isolation, locking and missing streaks across windows. Ruff lint and format
checks passed. A live Islamabad download for 1–2 October 2026 produced 48 unique
hourly rows with all 13 available metrics populated and rain probability missing.
Output hashes matched the manifest. Repeating that run with network requests
explicitly rejected reused the cache and produced a byte-identical CSV.

```powershell
& ./.venv/Scripts/python.exe -m pytest -q
& ./.venv/Scripts/ruff.exe check src tests
& ./.venv/Scripts/ruff.exe format --check src tests
```

Restricted Windows sessions may need an accessible dedicated pytest temporary
directory; this verification required execution outside the filesystem sandbox.

- https://open-meteo.com/en/docs/historical-weather-api
- https://open-meteo.com/en/docs/air-quality-api
- https://open-meteo.com/en/terms

Retain Open-Meteo, ECMWF ERA5 and Copernicus CAMS attribution. Confirm current
terms, quotas, joint historical coverage and mentor acceptance before a large
download or final model evaluation. Hopsworks, training and scheduling are later
modules.
