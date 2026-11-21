# Module 04: reference features and direct hourly targets

The deterministic transform creates calendar, lag, rolling, change, missingness
and gap-age features, with separate direct targets for horizons 1 through 24.
It uses the existing historical snapshot validation and integrity checks; it does
not download new data or train models. Configuration remains TOML, matching the
existing city configuration and avoiding an additional parser dependency.

## Working command

From the repository root, after `uv sync --extra storage`:

```powershell
& ./.venv/Scripts/python.exe -m mosamiat.features.pipeline data/backfill-smoke/islamabad_f709810ebfa7d5a5/2026-10-01_2026-10-02 --output data/features-smoke
```

The installed equivalent is `mosamiat-features`. Options are `--city-config`
(default `configs/city.toml`), `--config` (default `configs/features.toml`) and
`--output` (default `data/features`). A historical collector snapshot or a
published backfill directory is accepted. The configured city, coordinates and
timezone must match. Overlapping revisions, duplicate hours, gaps, non-hourly
timestamps and non-historical sources are rejected, rather than resolved silently.
Local feature generation does not require Hopsworks credentials.

## Feature definitions

One row represents reference time **t**, in UTC. Calendar columns use the city's
IANA timezone. All data-derived features use values strictly earlier than t:

- `hour`, `day_of_week` (Monday = 0), `month`, `hour_sin`, `hour_cos`.
- Per metric: lags at 1, 3 and 24 hours; rolling means and population variances
  over the preceding 3 and 24 hours; change from t-2 to t-1.
- `<metric>_missing` describes the preceding hour. `<metric>_age_hours` measures
  time since the latest non-null valid-time value strictly before t. It is null
  when no earlier value exists, and does not measure retrieval latency.
- Wind direction is replaced by sine and cosine before creating lag/rolling
  features. Raw direction in degrees stays in the separate target table.

Rolling windows require every value in the window. No forward fill, interpolation,
scaler or imputer is applied. Warm-up rows and the latest row remain in the feature
output for inference. For an observed series, t-1 must have become available by
t before these features can be used operationally. Statistical transforms are
identical when applied to a full series or its historical prefix.

## Targets and training boundaries

`targets.csv` has one row per reference time and `horizon_hours` (1–24), with
`target_at_utc = reference_at_utc + horizon_hours`. Each metric has a
`target_<metric>` value and a matching missing flag. The t+24 target is explicit;
it does not stand for a complete 24-hour path. Future unavailable labels stay
null; building targets never removes feature rows.

A future training pipeline must select suitable, non-null metric/horizon labels,
apply its warm-up policy, and split chronologically. It must exclude training
labels whose target timestamps cross the validation boundary. Fit any learned
preprocessing only on the training portion. Encode wind targets circularly before
model fitting. Historical rain probability remains unavailable; no rain-event
threshold or measured precipitation label is invented here.

## Publication and limitations

Outputs are local `features.csv`, `targets.csv` and a final `manifest.json` with
schema version, full feature definitions, source snapshots, source-manifest hash,
units, missing counts and output hashes. Require `status: complete` and matching
hashes before consuming artifacts. The manifest is marked building before writes
and published last. Use a separate destination for each dataset and one writer
per destination. Artifacts under `data/` remain ignored by Git. Remote feature
persistence is not part of this slice; the existing provider-data feature group
is unchanged by feature generation.

These are retrospective ERA5/CAMS provider reference values. The snapshots were
retrieved after their valid times; the pipeline does not establish what data was
available at a historical forecast issuance. Every output manifest explicitly
sets `is_ground_truth: false` and `operational_point_in_time_safe: false`.
Past-only arithmetic does not make these sources operational backtest evidence.
Source acceptance, real availability timestamps and measured labels remain
prerequisites for defensible model evaluation.

The existing two-day smoke dataset produces 48 feature rows and 1,152 target rows.
It is too short for a 24-hour warm-up plus complete 24-hour future labels and
does not establish seasonal coverage or model readiness. Tests cover future-value
perturbations, prefix equality, horizon alignment, circular encoding, missingness,
invalid timelines, configuration errors, source integrity and repeatable outputs.
On 20 November 2026 the full 84-test project suite and Ruff checks passed. The
local smoke artifacts are under `data/features-smoke/`.
