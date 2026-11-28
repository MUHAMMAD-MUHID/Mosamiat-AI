# Mosamiat staging verification

Review branch `review/mosamiat-reliability`. Live Hopsworks evidence is pending an isolated staging project. A GitHub Actions workflow on the review branch verified the drift checkpoint handoff with synthetic storage. [Review branch](https://github.com/MUHAMMAD-MUHID/Mosamiat-AI/tree/review/mosamiat-reliability)

## Verified results

| Scenario | Evidence | Result |
| --- | --- | --- |
| Exact execution tracking | Offline tests and a smoke check using the installed SDK 4.8.8 `Job.run()` with a mocked launch response | A newer unrelated execution is ignored. The observed execution ID and SDK launch arguments are preserved. |
| Ambiguous submission | Offline accepted-write response loss, empty and partial rows, complete matching rows, and conflicts | An unresolved submission blocks additional writes. A later complete readback resolves the run. Diagnostics record the operation, group and HTTP category when available. |
| Forecast materialization and partial repair | Offline terminal failure, full readback and replay | A failed or incomplete forecast stays unverified. Replay repairs only matching missing rows after a known execution reaches a terminal state. |
| Bulletin and dashboard | Offline graph, SQLite and Streamlit tests | A failed bulletin leaves the 192-row forecast available and retains the previous verified report. The app shows separate pending and failed statuses, including legacy bulletin-failure events. |
| Persistent retraining budget | Synthetic checkpoint passed between two separate GitHub-hosted runners at commit `f4b252e` | Runner `1000000273` reserved attempt 1 and published the fixture checkpoint. Runner `1000000274` restored it, reserved attempt 2, and declined attempt 3. Both jobs succeeded. [Reserve job](https://github.com/MUHAMMAD-MUHID/Mosamiat-AI/actions/runs/46120443322/job/140903556576) · [Restore job](https://github.com/MUHAMMAD-MUHID/Mosamiat-AI/actions/runs/46120443322/job/140903660279) |
| Model Registry and promotion | Full offline suite and branch diff review | The tests cover daily training, reuse of existing candidates, and validated promotion. The branch has no changes to model implementation files, dependencies, the lockfile, or the production workflow. |

## Commands and exit codes

| Command | Exit | Result |
| --- | --- | --- |
| `.venv/Scripts/python.exe -B -m pytest -p no:cacheprovider --basetemp $testDir -q --tb=short` | 0 | 161 passed in 159.06 seconds |
| `.venv/Scripts/ruff.exe check --no-cache src tests streamlit_app.py` | 0 | All checks passed |
| `git diff --check main...HEAD` | 0 | No whitespace errors |
| `.venv/Scripts/python.exe -B tests/verify_drift_checkpoint.py reserve $stageDir` | 0 | Published the first synthetic reservation |
| `.venv/Scripts/python.exe -B tests/verify_drift_checkpoint.py restore $stageDir` | 0 | Restored the reservation and enforced the cap |
| `.venv/Scripts/python.exe -B tests/verify_hopsworks_staging.py --project-id 59511` | 1 | Expected rejection of the production project ID before login |

The PowerShell variables point to new temporary directories for the corresponding test runs.

~~~powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$testDir = Join-Path $env:TEMP ('mosamiat-review-final-' + [guid]::NewGuid().ToString('N'))
$stageDir = Join-Path $env:TEMP ('mosamiat-drift-review-' + [guid]::NewGuid().ToString('N'))
~~~

## Live Hopsworks sign-off

The connected account’s project listing returned HTTP 200 and contained one accessible project. The lookup for `mosamiat_staging` returned HTTP 404. The live verifier requires staging settings in ignored `.env.staging` and an independently supplied project ID. It checks the ID through the REST API before connecting with SDK 4.8.8, then checks the SDK project again before any writes. It uses uniquely named synthetic feature groups and a separate checkpoint namespace inside that project.

~~~powershell
.venv/Scripts/python.exe -B tests/verify_hopsworks_staging.py --project-id $StagingProjectId
~~~

| Live evidence to collect | Current status |
| --- | --- |
| Confirmed SDK launch ID and newer unrelated execution ID from the same fixture job history | Pending staging access |
| Accepted execution ID whose response the fixture drops before application code receives it | Pending staging access |
| Initial unresolved outcome, safe diagnostic, later full readback, and count of relaunches | Pending staging access |
| Forecast execution ID and 192 verified rows, stopped bulletin execution ID, and bulletin row count | Pending staging access |
| Staging dashboard readback showing the latest 24-hour forecast and prior verified report | Pending staging access |

The live verifier deliberately discards the SDK response **after** the staging server accepts the launch. It uses the actual execution and readback data for verification. It stops a synthetic bulletin execution to exercise terminal failure handling. That scenario verifies application behavior after a failure. The Hudi metadata errors and the internal connection timeout remain separate provider incidents. [Provider incident record](hopsworks_incidents.md)

The hosted restart test validates checkpoint serialization and the attempt cap with fixture storage. It provides no evidence about GitHub-to-Hopsworks connectivity. A submission that never becomes identifiable or yields complete rows still requires operator investigation. SDK 4.8.8 supplies a job handle from `insert()` and does not always expose the first submission’s execution ID to the application.
