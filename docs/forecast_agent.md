# Islamabad Forecast Agent

The Forecast node reads the two Open-Meteo products from one recent collector snapshot, loads the active Hopsworks selection, and generates 24 hourly targets across eight fields. A second node writes the predictions to the `mosamiat_forecast_ledger` feature group, version 1, and reads the run back to verify the stored values, timestamps and model identities. Both nodes share the same LangGraph state. [LangGraph graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)

## Run one forecast

The first verified run was `islamabad_20261123T15Z`. It issued at 20:46 PKT on 23 November 2026 and covers 21:00 that evening through 20:00 on 24 November. All 192 Hopsworks rows matched the checkpoint, including the version used for each prediction.

Use the project root and the Hopsworks credentials described in the [storage guide](storage.md).

```powershell
& ./.tools/bootstrap/bin/uv.exe sync --locked --inexact --extra agents --extra storage --extra modeling --extra deep-learning
& ./.venv/Scripts/python.exe -m mosamiat.agents.graph
```

The command collects weather and air-quality data if the hour has no prepared forecast. To use an ingestion job's completed snapshot, pass its directory with `--snapshot`. Both products must have been retrieved in the current UTC hour. The graph uses the preceding 24 completed hours and masks the issue hour before calling the same feature transform used in training. The collector also returns future provider forecasts, which the node excludes. Live weather inputs come from the provider forecast product. The validation experiment used historical ERA5 weather and CAMS air quality.

## Which model supplies each field

Each promoted field and horizon points to a checksummed registry artifact. The active selection lives under `Resources/mosamiat_islamabad/serving/`. Unpromoted cells use the original Random Forest v1 and seasonal-naive policy, with historical rain frequency for rain chance. The Forecast Agent uses that fallback if the active selection or its artifacts are unavailable. At horizon 24, seasonal naive uses the value 24 hours before the issue hour.

Promotion requires at least 48 paired held-out cases, 95% coverage of eligible cases, and at least 1% lower error than both the incumbent and seasonal naive. Rain validation requires five wet and five dry cases. The gate scores continuous fields with RMSE, wind direction with circular MAE, and rain chance with Brier score. [Original model comparison](model_evaluation.md)

## Persistence and recovery

Each forecast row identifies the city, UTC reference hour, actual issue time, target time, horizon, field, value, unit, method and version. It also records the source snapshot, retrieval time, input hash, evaluation hash and policy version. The first target is the next whole UTC hour. An issuance partway through an hour therefore has less than 60 minutes of lead time to that first target.

### Forecast completion and retraining

The forecast receives a `forecast_completed` event after the Hopsworks readback matches all 192 checkpointed predictions. The scheduled worker publishes that checkpoint before bulletin publication or drift monitoring. A retry reads the same run's remote rows first. It skips a complete match, upserts only missing keys in a matching partial write, and rejects conflicting rows. It verifies the full run after any repair.

Drift assessment, drift report readback, and retraining have separate audit events. A failed retraining records the request and exception type, emits a warning in the workflow log, and publishes a drift record with `training_status=failed`. The forecast remains available. The dashboard loads the latest verified forecast and can compute its alerts before the bulletin is verified. It continues showing the latest available verified report.

On 27 November 2026, GitHub run `45965017499` successfully read back its input features and then failed before forecast readback. Hopsworks execution `309020` logged a Spark abort while Hudi committed metadata for the forecast upsert. Its exception chain ended with `ArrayIndexOutOfBoundsException` in the metadata write, including the `column_stats` partition. The captured logs identify a backend materialization failure. The table and model registry were left intact. [GitHub job](https://github.com/MUHAMMAD-MUHID/Mosamiat-AI/actions/runs/45965017499/job/140391523667) · [Hopsworks job history](https://eu-west.cloud.hopsworks.ai/p/59511/jobs/named/mosamiat_forecast_ledger_1_offline_fg_materialization/executions)

The local SQLite ledger at `data/forecasts/ledger.sqlite3` checkpoints the first complete forecast for a city and reference hour before uploading. An attempt log records starts, completed preparation, failures and successful readbacks. The remote primary key identifies one field and horizon in the hourly run. A repeat on the same ledger uses the saved payload, including the original issue time. The remote write uses upsert. Replaying a prepared forecast after an interruption works even after the original hour has passed.

```powershell
& ./.venv/Scripts/python.exe -m mosamiat.agents.graph --retry islamabad_20261123T15Z
```

For a recurring job, preserve the checkpoint directory between executions and use one publisher per city. Separate machines with separate local ledgers can prepare different revisions under the same hourly key.

### Inspect an interrupted run

Use the run ID printed by the forecast command to read its local event history in order. This also works for older runs that recorded fewer events.

```powershell
& ./.venv/Scripts/python.exe -m mosamiat.agents.graph --history islamabad_20261123T15Z
```

Each new attempt records the node name on entry. A node failure records the node name and exception type. A `verified` event confirms the forecast rows matched Hopsworks. A `completed` event means the entire graph finished, including drift. If a process stops abruptly, the last recorded event identifies how far it got.

Forecast and bulletin publication each allow three total attempts for recognized connection interruptions and timeouts. A readback mismatch stops the run immediately. After fixing the underlying issue, use `--retry RUN_ID` when a prepared forecast exists. The graph replays its saved values and checks the remote writes again. If the run failed before preparation, start a normal forecast for the current hour.

## Alert Agent

After Hopsworks confirms the forecast, the Alert Agent evaluates each of the 24 hourly targets. Adjacent hours crossing the same threshold form one episode. An episode includes the first and last affected target, the most extreme value and its timestamp, and guidance for the general public, sensitive groups and outdoor workers. A later forecast that clears a rule removes that episode from the current outlook. Earlier bulletins preserve what was forecast at the time.

Version 1 uses the following project screening thresholds. Both boundaries are inclusive. An episode receives the higher severity if any of its hours reaches the elevated threshold.

| Forecast condition | Advisory | Elevated |
| --- | ---: | ---: |
| US AQI | ≥101 | ≥151 |
| High air temperature | ≥35 °C | ≥40 °C |
| Low air temperature | ≤2 °C | ≤0 °C |
| Precipitation in one hour | ≥8 mm | ≥16 mm |
| Hourly rain probability | ≥60% | ≥80% |
| Relative humidity | ≥85% | ≥95% |
| UV index | ≥3 | ≥8 |
| Wind speed | ≥14 m/s | ≥21 m/s |

The AQI cutoffs use the starts of the US sensitive-group and unhealthy bands. WHO recommends sun protection at UV index 3. The wind thresholds correspond approximately to the start of near-gale and strong-gale speeds on the Met Office’s Beaufort table. The temperature, humidity, precipitation and rain-probability cutoffs are project choices for the internship demonstration. Official local warnings should come from the relevant public authority. [AirNow AQI bands](https://www.airnow.gov/aqi/aqi-basics/) · [WHO UV guidance](https://www.who.int/news-room/fact-sheets/detail/ultraviolet-radiation) · [Met Office Beaufort scale](https://weather.metoffice.gov.uk/guides/coast-and-sea/beaufort-scale)

## Weather Report Agent

The first processed forecast of each Islamabad calendar day generates a report. A new or more severe alert episode triggers an update within the same day. The agent treats overlapping or directly adjacent hourly targets for the same rule as a continuing episode. Quiet hours still record the alert evaluation and the reason the report was skipped.

The report uses deterministic prose based on the verified forecasts and alerts. It includes local times, temperature, AQI, precipitation, rain probability, humidity, UV, wind and the alert guidance. Both agents run without an LLM account. The graph saves one bulletin per run in SQLite and upserts it to Hopsworks feature group `mosamiat_weather_bulletins` v1. Each row contains the active alert episodes, new-episode count, report trigger, rule version and report text. A retry reuses the saved bulletin and checks the remote row against it.

## Drift Agent

The fourth node runs after the bulletin and writes one decision per hourly run to `mosamiat_drift_log` v1 in Hopsworks. The log includes the input and evaluation hashes, the check windows, sample counts, scores, thresholds, retraining request and candidate outcome. An incomplete window is recorded as `insufficient`. Replaying the same run reuses the checkpointed decision.

- **Input drift.** The node checks 14 rolling means from the registered forest's input schema using the same 24 completed hours as inference. Each mean has a lower and upper limit taken from the 830 eligible training rows. Wind direction is represented by its sine and cosine. One crossing triggers a candidate fit. These limits describe the training season in Islamabad. A seasonal change alone can cross them.
- **Forecast-error drift.** The first policy monitors the 24-hour lead for all eight fields. It matches previously verified forecasts to archived ERA5 weather and CAMS air-quality references by city, UTC target, method and version. Verification uses the latest seven-day archive window ending six UTC dates before the forecast's issue date. It requires 24 consecutive matched target hours per field. The upper limit is the largest 24-hour rolling validation error for the serving method. The scores are RMSE for continuous values, circular MAE for wind direction and Brier score against liquid-rain events for rain probability. Provider reference values can differ from local station observations.
- **Retraining.** The first crossing checkpoints a request and uses the same trainers, registry readback, and promotion gate as the daily job. Requests within 24 hours share a training checkpoint. A retry verifies and reuses its saved evaluation.

The checked-in monitoring profile belongs to the pinned v1 bundle. To reproduce it from the original local evaluations, run the following from the repository root.

```powershell
& ./.venv/Scripts/python.exe -m mosamiat.modeling.drift_profile artifacts/classical-2025-sep-oct data/features-2025-sep-oct configs/drift_islamabad_v1.json
```

Use the normal forecast command to run all four agents, or `--retry RUN_ID` to resume a saved forecast. Retain the source snapshot and the SQLite checkpoint for replay. New requests train locally on the graph worker. Hopsworks holds the team-visible decision record.

For the saved `islamabad_20261123T15Z` forecast, the temperature and the sine component of wind direction crossed their training ranges. No archived t+24 forecast pairs were available. The resulting candidate trained and evaluated on the 18 September–17 November 2026 archive window. Its t+24 validation AQI RMSE was 19.93, compared with 18.77 for seasonal naive on the same 269 cases.

## Scheduled pipelines

The **Mosamiat pipelines** GitHub Actions workflow runs from `main` on GitHub-hosted Ubuntu workers. Configure the repository Actions secrets `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT`, and `HOPSWORKS_HOST`. Each live step reads these settings from its environment. The **check** job uses offline fixtures and requires no credentials.

| Pipeline | Schedule | Result |
| --- | --- | --- |
| Hourly | Minute 12 of every UTC hour | Checkpoints the source snapshot and verifies the Random Forest feature row in `mosamiat_hourly_features` v1. Runs forecast, bulletin, and drift publication. |
| Daily | 01:37 UTC (06:37 Pakistan time) | Reads a verified Feature Store training snapshot, fits and evaluates Random Forest, SARIMAX, and LSTM, then registers candidates and evaluates promotion. |

Actions can start later than the cron time. The forecast uses the actual UTC hour of execution. A missed hour leaves a gap. [GitHub scheduled workflows](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)

Hosted runs keep their own ledger, snapshots and candidate evaluations under `Resources/mosamiat_islamabad/` in the Hopsworks project. The first hosted run starts its own history. Run one scheduled publisher per city. A new runner restores the latest checksummed ledger. It resumes a prepared run interrupted by worker loss before issuing the current forecast. Each completed upload is checked before publication, and the two newest ledger generations are retained. Source snapshots and candidate bundles accumulate under this project directory.

In **Actions → Mosamiat pipelines → Run workflow**, select **check** to exercise the workflows with fixtures, **daily** to train or verify today's candidate, or **hourly** to issue a current forecast. To replay a failed prepared forecast, select **hourly** and enter its run ID in the optional retry field. A daily retry on the same UTC date reuses the saved evaluation. Credentialed runs write to Hopsworks. The workflow log reports success only after readback.

## Training verification

The shared routine refreshes 61 days of retrospective provider references ending six UTC dates before the run. It writes past-derived features and matured hourly labels to `mosamiat_training_history` v1 and trains from the verified readback. Horizon joins discard targets that cross split boundaries. LSTM selects its stopping epoch within the training fold. SARIMAX covers temperature and humidity. Random Forest and LSTM cover all eight fields.

On 27 November 2026, a credentialed verification read **1,464 hours** from Hopsworks. Fit and evaluation took **54.891 s** for Random Forest, **5.375 s** for SARIMAX, and **27.250 s** for LSTM. Each 24-hour validation score below used 269 eligible cases. Scores use retrospective provider references.

| Model | AQI RMSE | Temperature RMSE (°C) | Humidity RMSE (percentage points) |
| --- | ---: | ---: | ---: |
| Seasonal naive | 23.94 | 0.91 | 6.11 |
| Random Forest | 21.84 | 1.81 | 6.02 |
| SARIMAX | — | 3.23 | 7.03 |
| LSTM | 20.89 | 0.92 | 6.77 |

Training artifacts contain full validation and test scores for every supported horizon. The `training_reports` folder in Hopsworks Resources records fit duration, verified registrations, and validation decisions. Daily model versions use `1000000000 + YYYYMMDD`. Drift versions use the UTC request timestamp in Unix seconds.

The 27 November daily checkpoint registered and reloaded all three families as **v1020261127**. Its published selection promoted 91 field-and-horizon combinations, with 69 for Random Forest, 19 for LSTM, and three for SARIMAX. [Hopsworks project](https://eu-west.cloud.hopsworks.ai/p/59511)
