# Hopsworks incidents for provider investigation

The GitHub Actions runs used Hopsworks Python SDK 4.8.8 against backend 5.2.2. Both scheduled runs authenticated, collected source data, and verified the hourly input row. Times below are UTC.

| Execution | Feature group | Observed failure |
| --- | --- | --- |
| 309126 | `mosamiat_forecast_ledger` v1 | On 27 November 2026 at approximately 18:58, run #31 received an SDK `RestAPIError` while submitting materialization. Hopsworks recorded a failed execution with stderr reporting a 10,000 ms connection timeout to an internal HTTPS endpoint. The exact HTTP status was absent from the Actions log. |
| 309209 | `mosamiat_weather_bulletins` v1 | On 28 November 2026 at approximately 02:03, run #32 received a terminal materialization failure. Hudi failed to commit metadata for instant `20261128020217492`. Forecast execution 309208 had already passed readback for all 192 rows. |
| 309020 | `mosamiat_forecast_ledger` v1 | On 27 November 2026, an earlier forecast materialization failed during the metadata commit for instant `20261127082511914`. It produced the same Avro and Hudi errors as execution 309209. |

Executions 309209 and 309020 recorded `ArrayIndexOutOfBoundsException` for indices 9 and 7 with record length 7. The stack passed through Avro `GenericData.Record.put` and Hudi `HoodieMetadataPayload.getInsertValue` at lines 445 and 439. The job logs identify the Hudi metadata writer and include `column_stats` failures. The application did not determine which metadata state triggered those failures.

Ask Hopsworks to inspect the launch control plane for execution 309126 and the metadata commits for executions 309209 and 309020. Validate any provider remediation on staging feature groups with representative existing metadata. Keep these incident IDs available when checking subsequent scheduled execution logs.

[Run #31](https://github.com/MUHAMMAD-MUHID/Mosamiat-AI/actions/runs/46052993474) · [Run #32](https://github.com/MUHAMMAD-MUHID/Mosamiat-AI/actions/runs/46095594492) · [Forecast materialization history](https://eu-west.cloud.hopsworks.ai/p/59511/jobs/named/mosamiat_forecast_ledger_1_offline_fg_materialization/executions) · [Bulletin materialization history](https://eu-west.cloud.hopsworks.ai/p/59511/jobs/named/mosamiat_weather_bulletins_1_offline_fg_materialization/executions)
