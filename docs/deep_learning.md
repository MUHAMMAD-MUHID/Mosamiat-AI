# Module 09 · LSTM forecasts for Islamabad

One PyTorch LSTM learns all eight forecast fields from scratch on a CPU. Every issue produces 24 hourly predictions per field. It uses the same ERA5/CAMS source snapshots, liquid-rain event labels and eligible issue/target pairs as the [Random Forest experiment](classical.md).

## Train and inspect

From the repository root, use the September–October 2025 backfill and verified rain labels from the classical runbook.

```powershell
& ./.tools/bootstrap/bin/uv.exe sync --locked --extra deep-learning
$dataset = 'data/backfill/islamabad_f709810ebfa7d5a5/2025-09-01_2025-10-31'
$rain = 'data/era5-rain-2025-sep-oct.json'
& ./.venv/Scripts/mosamiat-lstm.exe $dataset --rain-labels $rain --validation-start 2025-10-07T14:00:00Z --test-start 2025-10-19T19:00:00Z --output artifacts/lstm-2025-sep-oct
```

The second rain diagnostic uses `--validation-start 2025-09-15T00:00:00Z --test-start 2025-09-25T00:00:00Z --output artifacts/lstm-earlier-split-2025-sep-oct`. The two experiments overlap and use identical hyperparameters.

## What the model learns

- At issue hour t the encoder reads the 24 completed hours from t−24 through t−1. Inputs include weather, pollutants, encoded wind direction, missingness indicators and the local hour as sine and cosine. The training period supplies the means and standard deviations. Missing numeric inputs take their training means and retain a missingness indicator.
- A single recurrent layer with 32 hidden units feeds a direct prediction head. For six continuous variables and two wind components the network learns a correction to the most recent completed daily cycle. At lead 24 the matching available input is t−24. A separate logit per horizon estimates the probability of positive ERA5 liquid rain. Its initial bias uses the training rain frequency.
- The objective combines standardized squared errors for the six scalar targets, squared errors for sine and cosine (half weight each), and binary cross-entropy for rain. Missing labels and direction labels at calm or unknown wind speeds are masked. At inference the outputs are converted back to native units, bounded as appropriate, and wind components are converted back to compass degrees.
- Only complete in-fold 24-hour target paths enter training or checkpoint selection. Training windows are shuffled with a fixed seed while the hourly sequence inside every window stays ordered. Validation batches are chronological. Adam runs for up to 40 epochs with gradient clipping. Six consecutive epochs without a lower validation loss trigger early stopping. The lowest-loss validation checkpoint is restored before test scoring. These validation scores include the effect of checkpoint selection.

## Reload for inference

The checkpoint contains the state dictionary, architecture settings, normalization statistics and feature order. Supply at least 24 consecutive verified UTC hourly rows in the project's reference schema. The example below forecasts from the end of the archived sample.

```python
from pathlib import Path
import pandas as pd
from mosamiat.collect import City
from mosamiat.reference import load_reference
from mosamiat.modeling.lstm import Forecaster

dataset = Path("data/backfill/islamabad_f709810ebfa7d5a5/2025-09-01_2025-10-31")
values = load_reference(dataset, City.load(Path("configs/city.toml"))).values.tail(24)
model = Forecaster.load(Path("artifacts/lstm-2025-sep-oct/model.pt"))
issue = pd.DatetimeIndex([values.index[-1] + pd.Timedelta(hours=1)])
print(model.predict(values, issue))
```

Each run publishes weights in `model.pt`, epoch losses in `history.csv`, matched forecasts and labels in `cases.csv`, and per-horizon scores for the seasonal baseline and LSTM in `metrics.csv`. The example 24-hour path is in `forecast_example.csv`. The manifest records the split boundaries, best epoch, settings, source provenance and artifact hashes. These retrospective experiments use provider reference data from two months in one city.

## Islamabad results

The primary run used 830 training windows. Epoch 1 had the lowest validation loss (0.1822). Training loss continued to fall through epoch 7 while validation loss rose to 0.2036. For the earlier rain diagnostic the best epoch was 4 of 10.

Each primary t+24 test score below uses 269 eligible cases. Lower is better. SARIMAX was fitted for temperature and humidity only.

| t+24 test score | Seasonal naive | SARIMAX | Random Forest | LSTM |
| --- | ---: | ---: | ---: | ---: |
| US AQI, RMSE | 12.682 | — | 12.631 | 12.694 |
| Temperature, RMSE (°C) | 0.805 | 3.893 | 3.461 | 0.746 |
| Precipitation, RMSE (mm) | 0.000 | — | 0.492 | 0.010 |
| Rain event, Brier score | 0.0000 | — | 0.1968 | 0.0121 |
| Humidity, RMSE (percentage points) | 6.677 | 10.185 | 6.751 | 6.716 |
| UV index, RMSE | 0.2133 | — | 0.3462 | 0.2126 |
| Wind speed, RMSE (m/s) | 0.4676 | — | 0.6919 | 0.4674 |
| Wind direction, circular MAE (degrees) | 16.654 | — | 22.413 | 16.611 |

The temperature improvement on test did not repeat on validation. At t+24 the validation RMSE was 1.384 °C for the LSTM and 1.294 °C for seasonal naive. Several other differences were small.

The primary test contains no liquid-rain events. On the earlier split, which includes wet hours, the t+24 test Brier scores were 0.0491 for the LSTM, 0.0496 for the constant training rain frequency, 0.0746 for the forest and 0.0880 for seasonal naive. The small LSTM advantage over the constant probability needs independent seasonal verification.
