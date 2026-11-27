import json
import shutil
from importlib.metadata import version
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from mosamiat.backfill import digest
from mosamiat.collect import CollectionError
from mosamiat.modeling.baseline import FORECAST_METRICS


def test_registry_round_trip_reuses_version_and_rejects_changed_bytes(tmp_path, monkeypatch):
    registry_module = pytest.importorskip("mosamiat.modeling.registry")
    from mosamiat.modeling import forest

    times = pd.date_range("2025-01-01", periods=96, freq="h", tz="UTC", name="reference_at_utc")
    features = pd.DataFrame({"lag": np.arange(len(times))}, index=times)
    values = pd.DataFrame({name: features.lag for name in FORECAST_METRICS})
    monkeypatch.setitem(forest.PARAMETERS, "n_estimators", 2)
    trained = forest.fit(features, values, features.lag % 2, times[80], 3)
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    registry_module.joblib.dump(trained, evaluation / "model.joblib")
    (evaluation / "metrics.csv").write_text(
        "model,split,metric,horizon_hours,rmse\n"
        "random_forest,validation,aqi_value,24,1.0\nseasonal_naive,validation,aqi_value,24,2.0\n"
    )
    common = dict(city_id="islamabad", timezone="Asia/Karachi", status="complete")
    manifest = dict(
        common,
        kind="random_forest_evaluation",
        features=trained.columns,
        feature_definitions={},
        sklearn_version=version("scikit-learn"),
        provenance={"source_manifest_sha256": "demo"},
        validation_start_utc=times[80].isoformat(),
        sha256={f: digest(evaluation / f) for f in ("metrics.csv", "model.joblib")},
    )
    (evaluation / "manifest.json").write_text(json.dumps(manifest))
    features.to_csv(tmp_path / "features.csv")
    feature_manifest = dict(
        common,
        definitions={},
        source_manifest_sha256="demo",
        sha256={"features.csv": digest(tmp_path / "features.csv")},
    )
    (tmp_path / "manifest.json").write_text(json.dumps(feature_manifest))

    registry = MagicMock()
    remote = tmp_path / "remote"
    entry = registry.python.create_model.return_value
    entry.version = 3
    entry.save.side_effect = lambda path, **kwargs: shutil.copytree(path, remote)
    entry.download.side_effect = lambda local_path: shutil.copytree(
        remote, local_path, dirs_exist_ok=True
    )
    registry.get_models.side_effect = lambda name: [entry] if remote.exists() else []
    registry.get_model.return_value = entry
    args = (evaluation, tmp_path, 3, lambda: registry)
    first = registry_module.register(*args)
    assert first == ("mosamiat_islamabad_random_forest", digest(evaluation / "manifest.json"))
    assert registry_module.register(*args) == first
    entry.save.assert_called_once()
    registry.get_model.assert_called_with(first[0], version=3)
    assert {p.name for p in remote.iterdir()} == {"model.joblib", "metrics.csv", "manifest.json"}
    model = Path(remote / "model.joblib")
    model.write_bytes(b"changed")
    with pytest.raises(CollectionError, match="checksum mismatch"):
        registry_module.register(*args)
    entry.save.assert_called_once()


@pytest.mark.parametrize("challenger, expected", [(10.5, "lstm"), (13.0, None), (11.0, None)])
def test_promotion_accepts_only_improvement_on_shared_validation_cases(challenger, expected):
    from mosamiat.modeling.compare import validation_winner

    frame = pd.DataFrame(
        {
            "eligible": [True] * 64,
            "actual": 10.0,
            "seasonal_naive": 12.0,
            "incumbent": 11.0,
            "lstm": challenger,
            "random_forest": 11.0,
        }
    )
    result = validation_winner("temperature_c", frame, ["lstm", "random_forest"])
    assert result["winner"] == expected
    frame.loc[:10, "lstm"] = np.nan
    assert validation_winner("temperature_c", frame, ["lstm", "random_forest"])["winner"] is None
