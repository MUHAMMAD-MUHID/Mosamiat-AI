# Module 06 baseline and evaluation

Run the seasonal naive benchmark against a historical collector snapshot or a published backfill directory. From the project root in PowerShell, the existing Islamabad sample can be evaluated offline.

```powershell
& ./.venv/Scripts/python.exe -m mosamiat.modeling.evaluate data/backfill-smoke/islamabad_f709810ebfa7d5a5/2026-10-01_2026-10-02 --output artifacts/baseline-smoke
```

The installed entry point is `mosamiat-baseline`. Use `--city-config` for a different city configuration. By default the first 60% of the hourly index is used for training, the next 20% for validation and the rest for testing, with boundary positions rounded down. To reproduce a comparison as a dataset grows, pin both boundaries using timezone-aware hours, for example `--validation-start 2026-10-02T04:00:00Z --test-start 2026-10-02T14:00:00Z`. Each fold retains only issue and target timestamps inside its own half-open interval. Each interval must contain at least two hours.

## Forecast rule

At issue hour **t**, observations through **t−1** are eligible, matching the feature pipeline. The forecast repeats the 24 most recently completed hours at the corresponding UTC hour of day. For horizons 1–23 this reads the target hour minus 24 hours. At horizon 24 that hour would be **t**, so the latest completed matching hour is **t−24**. A missing source hour or value produces a missing prediction. The baseline needs no fitted parameters and can use completed hours from earlier folds.

## Comparing models

- `cases.csv` records the split, issue time, target time, horizon, metric, baseline source time, reference label, seasonal prediction and eligibility flag. Eligibility requires a usable label and seasonal prediction. Calm or unknown wind speed makes its associated direction unusable.
- Use `split_targets` for training partitions and `score_predictions(cases, predictions)` on the held-out cases for model comparisons. Index both inputs by the five fields in `mosamiat.modeling.baseline.CASE_KEY` and use the same timestamp types. The scorer aligns by key and rejects duplicate keys, unknown cases and missing predictions on eligible cases.
- `metrics.csv` includes a row for every held-out split, forecast variable and horizon from 1 through 24. Compare the counts of in-window pairs, available labels and scored cases alongside each result. Continuous variables use RMSE, MAE and R². A constant target or a single case leaves R² blank. Direction uses mean absolute shortest-arc error in degrees. Brier score requires a probability in [0, 1] and a verified binary rain event. The historical sources lack that event label, so rain probability has zero scored cases.
- `manifest.json` records split boundaries, evaluation time, city, units, source provenance and output hashes. Consume a run only when its status is `complete` and its hashes match.

## Islamabad smoke result

The 48-hour sample yields 45 issue/target pairs in each held-out fold. Seven variables have usable labels, giving 315 scored cases per fold. Both held-out intervals are ten hours long, so horizons 10–24 have zero pairs. The source values come from retrospectively retrieved ERA5/CAMS products. This run verifies the evaluation workflow. Estimating operational forecast skill requires longer held-out windows, accepted verification labels and evidence of what inputs were available at issuance.
