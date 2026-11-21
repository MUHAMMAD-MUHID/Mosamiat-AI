# Module 07 statistical forecasts

Two SARIMAX models forecast temperature and relative humidity for Islamabad at hourly horizons 1–24. Each model fits a daily pattern with 24-hour and 12-hour sine and cosine terms. A first-order autoregressive error accounts for recent departures from that pattern. The implementation uses the regression-with-AR-errors formulation described in the [statsmodels SARIMAX FAQ](https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_faq.html).

## Reproduce the local experiment

Run from the repository root with Python 3.11 and uv. The initial backfill uses the public Open-Meteo weather archive and air-quality API. Subsequent steps use the saved, integrity-checked snapshots.

```powershell
& ./.tools/bootstrap/bin/uv.exe sync --locked --extra analysis --extra modeling
& ./.venv/Scripts/python.exe -m mosamiat.backfill --start 2025-09-01 --end 2025-10-31 --output data/backfill
$dataset = 'data/backfill/islamabad_f709810ebfa7d5a5/2025-09-01_2025-10-31'
& ./.venv/Scripts/python.exe -m mosamiat.eda.pipeline $dataset --output artifacts/eda-2025-sep-oct
& ./.venv/Scripts/python.exe -m mosamiat.features.pipeline $dataset --output data/features-2025-sep-oct
& ./.venv/Scripts/python.exe -m mosamiat.modeling.evaluate $dataset --output artifacts/baseline-2025-sep-oct
& ./.venv/Scripts/mosamiat-statistical.exe $dataset --validation-start 2025-10-07T14:00:00Z --test-start 2025-10-19T19:00:00Z --output artifacts/statistical-2025-sep-oct
```

EDA found 1,464 unique hourly timestamps across September and October 2025. Temperature ranged from 15.6 to 37.0 °C and humidity from 24% to 96%, with complete coverage for both. Historical rain probability has no verification labels. The feature pipeline produced 1,464 input rows and 35,136 horizon-specific target rows for subsequent model comparisons.

## Fit and evaluation

The specification is fixed before evaluation. Each model fits once on the 878 hours before validation, using maximum likelihood. Missing training values remain missing in the state-space fit. The minimum of 72 observed training hours with every hour of the day represented is an implementation guard. A constant training series or failed fit produces a diagnostic.

At each issue time, the model starts from the last available value strictly earlier than the issue. It projects the residual forward over the actual number of elapsed hours, including any gaps, and adds the expected daily value at the target hour. Forecasting t+1 from a last value at t−1 takes two steps. Parameters remain fixed as new completed observations arrive. Humidity forecasts are bounded to 0–100%.

Both models use the [baseline evaluation contract](baseline.md). The validation and test windows each cover 293 hours. Scores for each horizon use identical cases for SARIMAX and seasonal naive, and each label stays within its own split. The table shows the t+24 result with 269 paired cases per row.

| Split | Target | Baseline RMSE | SARIMAX RMSE | SARIMAX MAE | SARIMAX R² |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation | Temperature (°C) | 1.294 | 3.025 | 2.771 | 0.452 |
| Validation | Humidity (percentage points) | 5.737 | 5.992 | 4.745 | 0.875 |
| Test | Temperature (°C) | 0.805 | 3.893 | 3.653 | 0.226 |
| Test | Humidity (percentage points) | 6.677 | 10.185 | 8.664 | 0.602 |

Seasonal naive achieved lower RMSE for both variables at every validation and test horizon. Keep it as the benchmark for the next model family. These results describe two months of retrospective ERA5/CAMS provider data. Coverage of additional seasons and verified input availability at forecast time are needed to assess operational skill.

The output directory contains `cases.csv` with aligned predictions and timestamps, `metrics.csv` with both models' scores by split, metric and horizon, and `manifest.json` with fitted coefficients, model specification, package version, cutoffs, provenance and output hashes. The coefficients can be loaded as JSON and passed to `mosamiat.modeling.statistical.predict` with the corresponding named hourly series and forecast keys. Model Registry promotion and automated retraining belong to later milestones.
