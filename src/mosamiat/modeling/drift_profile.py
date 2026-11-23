"""Build monitoring limits from the deployed evaluation's training and validation splits."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mosamiat.backfill import digest, write_json
from mosamiat.collect import CollectionError
from mosamiat.modeling.baseline import FORECAST_METRICS
from mosamiat.modeling.inference import MANIFEST, MODEL
from mosamiat.storage import read_json


def build(evaluation: Path, features: Path):
    manifest = read_json(evaluation / "manifest.json")
    source = read_json(features / "manifest.json")
    if (
        digest(evaluation / "manifest.json") != MANIFEST
        or source["source_manifest_sha256"] != manifest["provenance"]["source_manifest_sha256"]
        or source["definitions"] != manifest["feature_definitions"]
        or digest(features / "features.csv") != source["sha256"]["features.csv"]
        or any(
            digest(evaluation / f) != manifest["sha256"][f] for f in ("metrics.csv", "cases.csv")
        )
    ):
        raise CollectionError("Monitoring reference differs from the registered evaluation")
    frame = pd.read_csv(features / "features.csv", index_col="reference_at_utc", parse_dates=True)
    train = frame[frame.index < pd.Timestamp(manifest["validation_start_utc"])].iloc[
        manifest["warmup_hours"] : -24
    ]
    inputs = {}
    for feature in manifest["features"]:
        if feature.endswith("_mean_24h"):
            s = train[feature].dropna()
            if len(s) >= 24:
                inputs[feature.removesuffix("_mean_24h")] = dict(
                    lower=float(s.min()), upper=float(s.max()), n=len(s)
                )
    scores = pd.read_csv(evaluation / "metrics.csv").query(
        "split == 'validation' and horizon_hours == 24"
    )
    cases = pd.read_csv(evaluation / "cases.csv", parse_dates=["target_at_utc"]).query(
        "split == 'validation' and horizon_hours == 24 and eligible"
    )
    errors = {}
    for metric in FORECAST_METRICS:
        group = cases[cases.metric == metric].set_index("target_at_utc").sort_index().asfreq("h")
        measure = "circular_mae_deg" if metric == "wind_direction_deg" else "rmse"
        if metric == "rain_probability":
            model = "training_rain_frequency"
            diff = manifest["rain_training_event_rate"] - group.actual
        else:
            table = scores[scores.metric == metric].set_index("model")[measure]
            model = (
                "random_forest" if table.random_forest < table.seasonal_naive else "seasonal_naive"
            )
            diff = group[model] - group.actual
        if metric == "wind_direction_deg":
            rolling = ((diff + 180) % 360 - 180).abs().rolling(24).mean()
        else:
            rolling = diff.pow(2).rolling(24).mean()
            if metric != "rain_probability":
                rolling = np.sqrt(rolling)
        errors[metric] = dict(
            model=MODEL if model == "random_forest" else model,
            measure="brier" if metric == "rain_probability" else measure,
            upper=float(rolling.max()),
            n=int(rolling.count()),
        )
    return dict(
        version=1,
        evaluation_sha256=MANIFEST,
        city_id=manifest["city_id"],
        inputs=inputs,
        errors=errors,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evaluation", "features", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    write_json(args.output, build(args.evaluation, args.features))
