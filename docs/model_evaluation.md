# Islamabad forecast model comparison

The previous-day forecast had the lowest validation temperature error at every forecast horizon from 1 to 24 hours. At 24 hours, Random Forest improved the AQI forecast from 15.566 to 11.509 US AQI units of root mean squared error (RMSE). Its advantage fell to 0.051 units on the test period.

This comparison supports the forecasting system and written report described on pages 13–16 of the SHINE Session 1 Data Science deck. Its six metric groups correspond to eight forecast fields here because precipitation includes amount and probability, and wind includes speed and direction.

## Validation results at 24 hours

All four approaches use the same Islamabad weather and air-quality snapshots from September and October 2025. Training ends on 7 October at 14:00 UTC, when validation begins. Testing begins on 19 October at 19:00 UTC. Each score below uses 269 paired forecasts with both the issue time and target time in the validation period. Lower scores indicate smaller errors. SARIMAX covers only temperature and relative humidity.

| Target and measure | Previous day | SARIMAX | Random Forest | LSTM |
| --- | ---: | ---: | ---: | ---: |
| US AQI, RMSE | 15.566 | — | **11.509** | 15.581 |
| Temperature, RMSE (°C) | **1.294** | 3.025 | 3.300 | 1.384 |
| Precipitation, RMSE (mm) | **0.1226** | — | 0.4534 | 0.1236 |
| Rain probability, Brier score | 0.0297 | — | 0.1691 | **0.0122** |
| Relative humidity, RMSE (percentage points) | **5.737** | 5.992 | 9.256 | 5.771 |
| UV index, RMSE | **0.2171** | — | 0.2466 | 0.2221 |
| Wind speed, RMSE (m/s) | 0.6819 | — | **0.6293** | 0.6819 |
| Wind direction, circular MAE (degrees) | 20.714 | — | 26.139 | **20.532** |

## Implications for the Forecast Agent

Use the previous-day forecast as the initial benchmark for each field and horizon. Compare candidates using validation RMSE for continuous values, circular mean absolute error for wind direction and Brier score for rain probability. Use the test period to describe how well the validation differences held up.

- **Temperature and precipitation amount.** The previous-day forecast had the lowest validation RMSE at every horizon for both fields. The precipitation result describes a dry validation period.
- **AQI and wind.** Random Forest is the leading 24-hour AQI candidate. Its 24-hour wind-speed advantage reversed on test. LSTM had the smallest wind-direction error at all 24 validation horizons, with a 0.182° advantage at 24 hours. Check these differences across more seasons before promoting a version for operational use.
- **Rain probability.** The primary validation and test periods contained zero liquid-rain events. On an earlier test window containing wet hours, the LSTM scored 0.0491 against 0.0496 for a constant probability estimated from training data. Defer rain-model promotion until an independent window includes both wet and dry hours.

The data are retrospective Open-Meteo reference estimates from ECMWF ERA5 and Copernicus CAMS. The LSTM also used validation loss for early stopping. Live input availability and performance across seasons still need verification.

## Verified Hopsworks candidate

On 23 November 2026, the Islamabad Random Forest was registered as `mosamiat_islamabad_random_forest`, version **1**. Its registry entry contains the fitted model, the evaluation manifest with the ordered feature names, and the per-horizon scores. For the issue time of 7 October 2025 at 14:00 UTC, the downloaded model reproduced all 192 local forecast values exactly across 24 hours and eight fields. This version is available for Forecast Agent integration. Production selection per field and horizon remains open.

To register or recheck this version, use the project-root environment configured for Hopsworks and the modeling and storage dependencies. The command reuses an existing version and checks its contents against the specified local evaluation. Changed evaluations require a new version number. [Hopsworks model registration guide](https://docs.hopsworks.ai/4.8/user_guides/mlops/registry/frameworks/python/)

```powershell
& ./.venv/Scripts/python.exe -m mosamiat.modeling.registry artifacts/classical-2025-sep-oct --features data/features-2025-sep-oct --version 1
```

An inference caller can load the verified version with the recorded manifest checksum. Pass a DataFrame of engineered features indexed by UTC issue hour to `model.predict`. The output identifies the issue time, horizon and target time for each row.

```python
from mosamiat.modeling.registry import load_version
from mosamiat.storage import connect_project

model = load_version(
    connect_project().get_model_registry(),
    "mosamiat_islamabad_random_forest",
    1,
    "1487f5d407c6a4df2c6f5af7e5abd99d9a7abeebd04fb2b6d4b81cfe415ac726",
)
```

## Reproduce all horizons

From the repository root, run this command after generating the three evaluation directories. It verifies the input file hashes and checks that the runs share source snapshots, split boundaries, timestamps, labels, eligibility and baseline predictions before calculating scores.

```powershell
& ./.venv/Scripts/python.exe -m mosamiat.modeling.compare --classical artifacts/classical-2025-sep-oct --statistical artifacts/statistical-2025-sep-oct --lstm artifacts/lstm-2025-sep-oct --output artifacts/comparison-2025-sep-oct
```

The output directory contains `metrics.csv` with validation and test scores, case counts and horizons for each model and field, and `manifest.json` with the source provenance, split boundaries and file hashes. The CSV includes MAE and R² for continuous values. R² is undefined when the reference values are constant.
