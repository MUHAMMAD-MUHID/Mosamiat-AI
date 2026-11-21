# Module 05: EDA findings for Islamabad

Reference window: 2026-10-01T00:00:00+00:00 through 2026-10-02T23:00:00+00:00.
Expected hourly timestamps: 48; source timestamps: 48; missing timestamps: 0.
Duplicate metric/hour rows: 0 (overlapping revisions are rejected).

## Metric coverage and extremes

| Metric | Unit | Present | Missing % | Longest gap (h) | Min | Max | IQR review flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| temperature_c | degC | 48 | 0.00 | 0 | 21.7 | 35 | 0 |
| humidity_pct | % | 48 | 0.00 | 0 | 27 | 75 | 0 |
| precipitation_mm | mm | 48 | 0.00 | 0 | 0 | 0 | 0 |
| wind_speed_mps | m/s | 48 | 0.00 | 0 | 0.74 | 2.89 | 0 |
| wind_direction_deg | degree | 48 | 0.00 | 0 | 13 | 356 | 0 |
| aqi_value | US_AQI | 48 | 0.00 | 0 | 85 | 145 | 5 |
| uv_index | 1 | 48 | 0.00 | 0 | 0 | 6.6 | 4 |
| pm2_5_ug_m3 | ug/m3 | 48 | 0.00 | 0 | 15.8 | 65 | 0 |
| pm10_ug_m3 | ug/m3 | 48 | 0.00 | 0 | 20 | 82.9 | 0 |
| carbon_monoxide_ug_m3 | ug/m3 | 48 | 0.00 | 0 | 244 | 2538 | 0 |
| nitrogen_dioxide_ug_m3 | ug/m3 | 48 | 0.00 | 0 | 4.4 | 137.6 | 0 |
| sulphur_dioxide_ug_m3 | ug/m3 | 48 | 0.00 | 0 | 3.7 | 19.4 | 0 |
| ozone_ug_m3 | ug/m3 | 48 | 0.00 | 0 | 0 | 199 | 0 |
| rain_probability | 1 | 0 | 100.00 | 48 | unavailable | unavailable | 0 |

## Rain, AQI and wind

Precipitation: 48 exact-zero hours and 0 positive hours among 48 available hours. Observed sum: 0.0 mm. Precipitation includes snow; a positive value is not a validated rain-event label. Daily sums in daily_precipitation.csv mark partial local days.
Historical rain probability is reported separately and is never imputed from precipitation.
AQI scales present: US. Source-specific coverage is in sources.csv; this is model-derived CAMS AQI, not a station feed.
Wind: 48 non-calm directions, 0 calm hours; circular mean 356.16282491450954 degrees, resultant length 0.5347079119195798. Arithmetic means and Pearson correlations of raw degrees are omitted. Wind sine/cosine components are used in profiles and correlations.

## Temporal patterns, correlations and review flags

The sample spans 3 local dates in 1 calendar month(s). Local-hour and daily means include per-metric sample counts in the exported tables. Daily precipitation means are mean hourly amounts, not daily totals. Edge dates can be incomplete. These are descriptive profiles, not evidence for annual seasonality or forecast skill.
Pairwise Pearson correlations require at least three complete pairs and non-constant values. correlation_counts.csv provides support for each coefficient. Autocorrelation, small samples and shared model sources limit interpretation.
IQR review flags use Q1 - 1.5 IQR and Q3 + 1.5 IQR, at least four values and nonzero IQR. No flags are computed for raw wind direction, all-null, constant or zero-IQR series. No flagged values are deleted. Review extremes against provenance before changing data.

## Limitations and drift risks

- ERA5/CAMS provider reference values are not verified station observations.
- Retrospective retrieval does not establish historical forecast-time availability.
- IQR flags identify values for review; legitimate weather extremes are not errors.
- Correlations are descriptive, autocorrelated and do not establish causation.
- Source/model changes, seasons and unequal historical access can mimic concept drift.
- Less than 30 days: hourly/day profiles are descriptive; seasonal inference is unsupported.
- Historical rain probability is unavailable; no event definition or probability label is inferred.
- Insufficient history for 24-hour feature warm-up plus a complete t+24 label.
- More history and accepted labels are needed before model comparison. Validate coverage and provider quotas before expanding the download.
- Changes in model versions, resolution, provider coverage or any future station sensor can shift distributions. Drift monitoring must distinguish these from real weather changes.
- Compare future training and monitoring windows by season and source. A short sample cannot provide production drift thresholds or representative extremes.

Attribution: Open-Meteo, ECMWF ERA5, Copernicus CAMS.

## Reproduction and evidence

See [the Module 05 runbook](eda.md) for the exact command, artifact definitions and notebook runner. The local report is `artifacts/eda-smoke/report.html`. This is the existing 48-hour smoke dataset; no bulk download was performed for this module.

The full project suite passed all 94 tests. Ruff lint and formatting checks passed. All 20 published artifact hashes matched the manifest, including eight charts and the self-contained HTML report. Three notebook views are included for the raw audit, weather and air quality.
