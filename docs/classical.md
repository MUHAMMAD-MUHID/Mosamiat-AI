# Module 08 · Random Forest forecasts for Islamabad

The classical model produces eight fields at every hourly lead from 1 to 24. It uses one multi-output forest per field, so the 24 predictions share trees. The benchmark uses the same 1,464 hours and the same October split boundaries as the [SARIMAX experiment](statistical.md).

## Inputs and training

The [Module 04 transform](features.md) computes calendar, lag, rolling, change and missingness features. At issue hour t it can use reference values only through t−1. The first 24 hours warm up the features. Every label from t+1 through t+24 must precede the validation boundary for that example to enter training. Training-only feature selection drops constant and empty columns. One median imputer is fitted on the remaining training inputs and shared by the eight forests. A missing label excludes that issue for the affected forest.

The primary fit used 830 eligible training issue hours and 117 features. Each forest has 64 trees, a maximum depth of 10, a minimum of three samples per leaf, and a fixed seed of 42. The regressor for wind direction fits sine and cosine jointly, omits calm or unknown-speed labels, and converts its predictions back to degrees. For a vanishing predicted direction vector it uses the last non-calm training bearing.

The rain classifier estimates the probability that the ERA5 liquid-rain amount for the hour ending at the target timestamp exceeds zero. This separately collected archive field includes showers. Each raw response is bound to its weather snapshot and checked for matching request parameters, grid, UTC timestamps and units. Missing rain amounts stay unknown. The classifier uses the derived binary events only as labels. All weather and air-quality scores use retrospective ERA5 or CAMS reference data. [Open-Meteo hourly variable definitions](https://open-meteo.com/en/docs/historical-weather-api)

## Reproduce the experiment

From the repository root, with the September–October backfill described in the statistical runbook:

```powershell
& ./.tools/bootstrap/bin/uv.exe sync --locked --extra modeling
$dataset = 'data/backfill/islamabad_f709810ebfa7d5a5/2025-09-01_2025-10-31'
$rain = 'data/era5-rain-2025-sep-oct.json'
& ./.venv/Scripts/python.exe -m mosamiat.modeling.rain_labels $dataset --output $rain
& ./.venv/Scripts/mosamiat-classical.exe $dataset --rain-labels $rain --validation-start 2025-10-07T14:00:00Z --test-start 2025-10-19T19:00:00Z --output artifacts/classical-2025-sep-oct
```

The rain collector reuses verified local responses on repeat runs. Training and evaluation then run offline. The output manifest holds the feature definitions, input hashes, hyperparameters and training counts. The published model includes the imputer and the eight estimators. To reproduce inference using the saved feature CSV:

```python
import joblib
import pandas as pd

forest = joblib.load("artifacts/classical-2025-sep-oct/model.joblib")
features = pd.read_csv("data/features-2025-sep-oct/features.csv", parse_dates=["reference_at_utc"])
forecast = forest.predict(features.set_index("reference_at_utc").tail(1))
print(forecast)
```

## Primary 24-hour results

Each model has 269 scored forecasts per variable per split at t+24. Lower scores are better. The full per-horizon RMSE, MAE and R² for the continuous targets, circular direction errors and probability scores are in `metrics.csv`. The aligned forecasts and reference labels are in `cases.csv`.

| Target and score | Validation baseline | Validation forest | Test baseline | Test forest |
| --- | ---: | ---: | ---: | ---: |
| AQI, RMSE (US AQI) | 15.566 | 11.509 | 12.682 | 12.631 |
| Temperature, RMSE (°C) | 1.294 | 3.300 | 0.805 | 3.461 |
| Precipitation, RMSE (mm) | 0.123 | 0.453 | 0.000 | 0.492 |
| Rain event, Brier score | 0.030 | 0.169 | 0.000 | 0.197 |
| Humidity, RMSE (percentage points) | 5.737 | 9.256 | 6.677 | 6.751 |
| UV index, RMSE | 0.217 | 0.247 | 0.213 | 0.346 |
| Wind speed, RMSE (m/s) | 0.682 | 0.629 | 0.468 | 0.692 |
| Wind direction, circular MAE (degrees) | 20.714 | 26.139 | 16.654 | 22.413 |

The forest's AQI improvement carried through to test, although the test difference was small. Wind speed improved on validation and regressed on test. These comparisons cover two months in one city.

## Rain evaluation

The label audit found 59 positive hours in September and 38 in October, with complete coverage. All 97 occurred before the primary validation boundary. Raw probabilities, the fraction of positive training hours, and the last available same-hour binary event are compared on the same eligible cases. At t+24 the seasonal forecast uses the event at t−24 because the issue-hour value at t is still unavailable. The validation and test periods contain no wet hours. A fitted probability calibration curve on these windows would have no positive examples.

As a diagnostic, a second run retained the model settings and moved the validation start to 15 September and the test start to 25 September, both at 00:00 UTC. Use the command above with those two boundaries and `--output artifacts/classical-earlier-split-2025-sep-oct`. The three windows contained 44, 12 and 41 wet hours respectively. This diagnostic reuses the same source period and was selected after checking event coverage.

| t+24 Brier score | Validation (216 pairs) | Test (864 pairs) |
| --- | ---: | ---: |
| Seasonal naive event | 0.1019 | 0.0880 |
| Random Forest | 0.0721 | 0.0746 |
| Constant training rain frequency | 0.0547 | 0.0496 |

The training-frequency forecast performed best in this diagnostic. Reliability tables record mean predicted probability, event rate and count for each occupied probability bin at each horizon. More wet and dry periods are needed to fit and verify calibration before promoting a rain model.
