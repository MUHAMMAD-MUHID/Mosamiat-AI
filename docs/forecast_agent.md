# Islamabad Forecast Agent

The Forecast node reads the two Open-Meteo products from one recent collector snapshot, downloads the pinned Hopsworks model bundle, and generates 24 hourly targets across eight fields. A second node writes the predictions to the `mosamiat_forecast_ledger` feature group, version 1, and reads the run back to verify the stored values, timestamps and model identities. Both nodes share the same LangGraph state. [LangGraph graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)

## Run one forecast

The first verified run was `islamabad_20261123T15Z`. It issued at 20:46 PKT on 23 November 2026 and covers 21:00 that evening through 20:00 on 24 November. All 192 Hopsworks rows matched the checkpoint, including the version used for each prediction.

Use the project root and the Hopsworks credentials described in the [storage guide](storage.md).

```powershell
& ./.tools/bootstrap/bin/uv.exe sync --locked --inexact --extra agents --extra storage --extra modeling
& ./.venv/Scripts/python.exe -m mosamiat.agents.graph
```

The command collects weather and air-quality data if the hour has no prepared forecast. To use an ingestion job's completed snapshot, pass its directory with `--snapshot`. Both products must have been retrieved in the current UTC hour. The graph uses the preceding 24 completed hours and masks the issue hour before calling the same feature transform used in training. The collector also returns future provider forecasts, which the node excludes. Live weather inputs come from the provider forecast product. The validation experiment used historical ERA5 weather and CAMS air quality.

## Which model supplies each field

The initial policy compares the registered Random Forest v1 and the previous-day benchmark at each field and horizon. It reads validation scores from the checksum-verified registry bundle. Lower RMSE wins for continuous fields, lower circular mean absolute error wins for wind direction, and a tie retains the previous-day forecast. At horizon 24, that forecast uses the value 24 hours before the reference hour because the reference hour itself is incomplete.

Rain probability uses the training-period liquid-rain event frequency, approximately 11.05%, for all horizons. It is a constant benchmark with no hour-specific weather information. The primary validation window contained zero liquid-rain events. The decision applies only to the models available in this integration. [Model comparison](model_evaluation.md)

## Persistence and recovery

Each forecast row identifies the city, UTC reference hour, actual issue time, target time, horizon, field, value, unit, method and version. It also records the source snapshot, retrieval time, input hash, evaluation hash and policy version. The first target is the next whole UTC hour. An issuance partway through an hour therefore has less than 60 minutes of lead time to that first target.

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
- **Retraining.** The first crossing checkpoints a request and runs the existing Random Forest trainer on the latest 61 archive days eligible at issue time, using chronological train, validation and test splits. Crossings within 24 hours share the request. A retry verifies and reuses a completed candidate. Its weights, scores, forecasts and manifest are stored together under `data/drift/candidates/<request_id>/evaluation/`. Serving continues with the registered version until a separate promotion decision compares the models on common validation cases.

The checked-in monitoring profile belongs to the pinned v1 bundle. To reproduce it from the original local evaluations, run the following from the repository root.

```powershell
& ./.venv/Scripts/python.exe -m mosamiat.modeling.drift_profile artifacts/classical-2025-sep-oct data/features-2025-sep-oct configs/drift_islamabad_v1.json
```

Use the normal forecast command to run all four agents, or `--retry RUN_ID` to resume a saved forecast. Retain the source snapshot and the SQLite checkpoint for replay. New requests train locally on the graph worker. Hopsworks holds the team-visible decision record.

For the saved `islamabad_20261123T15Z` forecast, the temperature and the sine component of wind direction crossed their training ranges. No archived t+24 forecast pairs were available. The resulting candidate trained and evaluated on the 18 September–17 November 2026 archive window. Its t+24 validation AQI RMSE was 19.93, compared with 18.77 for seasonal naive on the same 269 cases. The serving configuration remains pinned to v1.
