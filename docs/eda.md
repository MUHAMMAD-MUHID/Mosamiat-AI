# Module 05: reproducible EDA and data-quality report

This module audits historical provider reference data and publishes reusable
tables, eight static charts, written findings and a self-contained HTML report.
It uses the same raw snapshot validation as Module 04. No network requests,
Hopsworks writes, imputation or model fitting occur during an EDA run.

## Setup and working commands

From the repository root, install the analysis extra. Keep the storage extra when
you also want the existing Hopsworks environment available:

```powershell
& ./.tools/bootstrap/bin/uv.exe sync --locked --extra storage --extra analysis
& ./.venv/Scripts/python.exe -m mosamiat.eda.pipeline data/backfill-smoke/islamabad_f709810ebfa7d5a5/2026-10-01_2026-10-02 --output artifacts/eda-smoke
```

The installed equivalent is `mosamiat-eda`. The positional input is a published
backfill directory or an individual historical collector snapshot. Options are
`--city-config` (default `configs/city.toml`) and `--output` (default
`artifacts/eda`). Point the same command at a longer validated backfill to repeat
the analysis. Coverage is measured against the declared UTC range, including
missing edge hours and gaps between snapshots.

The loader validates source requests, units, city, coordinates, timezone and raw
arrays. Backfill artifacts and checkpoint hashes must agree. Corrupt inputs and
overlapping revisions fail explicitly. `src/mosamiat/reference.py` shares this
loading logic with the feature pipeline; EDA fills absent timestamps with nulls
for its audit, while feature generation continues to reject incomplete timelines.

## Artifacts and interpretation

| Output | Purpose |
| --- | --- |
| `report.html` | Portable report with all eight PNGs embedded |
| `findings.md`, `summary.json` | Human-readable findings and structured counts/limitations |
| `metrics.csv` | Coverage, longest missing streak, distribution, extremes and IQR support for every metric |
| `sources.csv` | Product, data kind, unit, AQI scale, non-null count and retrieval/valid-time ranges |
| `anomalies.csv` | IQR review candidates, with timestamps, values and fences |
| `correlations.csv`, `correlation_counts.csv` | Pairwise Pearson coefficients and supporting sample counts |
| `hourly_profile.csv`, `daily_profile.csv` | Local-time mean profiles with per-metric counts |
| `daily_coverage.csv` | Per-metric non-null percentage by UTC date |
| `daily_precipitation.csv` | Observed local-day sum, available hours, expected day length and completeness |
| `manifest.json` | Completion marker, source hashes, settings and artifact hashes |

Charts cover UTC daily coverage, weather time series, air-quality/UV time series,
all-metric distributions, local-hour means, local-date means, correlations and a
16-sector wind rose. Red series markers are IQR review flags. Missing data,
constant series and unavailable correlations are displayed explicitly. Plotting
uses a non-interactive backend and the ignored project `.cache/matplotlib/`
directory by default.

IQR flags use the conventional 1.5 IQR fences, at least four values and positive
IQR. Raw direction, all-null, constant and zero-IQR series have no IQR test. A flag
does not establish a bad measurement: none are removed. Do not reuse these
full-period exploratory fences as training preprocessing; fit any later learned
rules using training data only.

Correlations need at least three complete pairs and non-constant values. This is
a computational minimum, not evidence for statistical significance. Rainfall
with no variation cannot have a meaningful correlation coefficient. Autocorrelation,
shared provider models and small samples limit all correlations.

Wind direction is circular. Arithmetic degree means, quartiles and correlations
are omitted. Sine/cosine components support profiles and correlations. The rose
and all circular summaries, profiles and correlations use directions with known
speed greater than zero; calm and unknown-speed hours are excluded.
Mean resultant length describes concentration
from zero to one; opposing directions can have an undefined mean.

Precipitation exact-zero/positive counts describe the provider's hourly amounts,
which can include snow. They do not define a rain event or provide a rain
probability label. An all-missing precipitation series has an unknown sum, not a
zero sum. Daily profile values are **means of hourly amounts**; daily observed
totals are in the separate precipitation table, with partial local days marked.
Expected local-day lengths account for daylight-saving changes.

## Notebook views

The three clean source notebooks read published artifacts and verify their
hashes. They contain no duplicated transformation logic or network calls:

- `notebooks/00_raw_data_audit.ipynb`: source provenance, coverage, missingness and review flags.
- `notebooks/01_weather_eda.ipynb`: temperature, humidity, precipitation, wind and temporal profiles.
- `notebooks/02_air_quality_eda.ipynb`: CAMS AQI, UV, pollutants, correlations and drift risks.

They default to `artifacts/eda-smoke`; change `OUTPUT` in the setup cell when
reviewing another run. Execute all three using the project's interpreter:

```powershell
& ./.venv/Scripts/python.exe notebooks/execute.py
```

Executed copies are written under ignored `artifacts/eda-notebooks/`, preserving
clean source notebooks. The runner uses the current Python interpreter explicitly
and keeps IPython/Jupyter runtime files in the project cache.

## Evidence and limits

The committed [findings](eda_findings.md) describe the existing Islamabad
1–2 October 2026 UTC smoke dataset, not a newly downloaded long history. It has
48 continuous hourly timestamps, only one sampled calendar month, and unavailable
historical rain probability. The descriptive profiles cannot establish annual
seasonality, general extremes, representative wet-weather behavior or forecast
skill. The under-30-day warning is a reporting heuristic, not a readiness gate.

The manifest retains `is_ground_truth: false` and
`operational_point_in_time_safe: false`. ERA5 reanalysis and CAMS model products
were retrieved retrospectively; they are not verified station observations or
proof of historical forecast-time availability. AQI uses unit `US_AQI`; the
existing source schema records the corresponding scale as `aqi_scale = US`.
Do not mix incompatible AQI scales or silently relabel model products as actuals.

Seasonality, changing model versions/resolution, unequal provider history and any
future sensor changes can shift distributions. Later drift monitoring must
distinguish these from real weather changes and compare appropriate seasons and
sources. Source acceptance, longer representative history and defensible rain
labels remain prerequisites for final model evaluation. Module 06 is the next
implementation step; no training or baseline evaluation is included here.

Publication holds one lock per output directory. The manifest is marked building
before artifacts are written and complete only after hashes are recorded. Failed
rendering marks it failed. Readers must require complete status and verify
hashes. After a force-killed run, remove its `.lock` only once the writer is gone.
Generated artifacts remain under ignored `artifacts/`; the written findings and
notebook sources can be committed.

Suggested commit: `docs(eda): add data-quality analysis for all target metrics`.
