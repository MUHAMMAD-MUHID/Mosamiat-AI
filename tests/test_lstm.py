import numpy as np
import pandas as pd
import pytest

from mosamiat.backfill import METRICS
from mosamiat.collect import CollectionError
from mosamiat.modeling.baseline import FORECAST_METRICS


def test_window_edges_and_circular_seasonal_reference():
    torch = pytest.importorskip("torch")
    from mosamiat.modeling.lstm import Network
    from mosamiat.modeling.lstm_data import REGRESSION, HourlyWindows

    values = torch.arange(100, dtype=torch.float32).view(-1, 1).expand(-1, 10)
    dataset = HourlyWindows(values, values, 0, 80)
    assert len(dataset) == 32
    inputs, targets = dataset[0]
    assert inputs[:, 0].tolist() == list(range(24))
    assert targets[:, 0].tolist() == list(range(25, 49))
    assert dataset[len(dataset) - 1][1][-1, 0].item() == 79
    with pytest.raises(IndexError):
        dataset[len(dataset)]
    with pytest.raises(CollectionError, match="Insufficient"):
        HourlyWindows(values, values, 80, 100)
    with torch.no_grad():
        output = Network(10, 4, 0.25)(inputs.unsqueeze(0))[0]
    expected = torch.cat([inputs[1:], inputs[:1]])[:, : len(REGRESSION)]
    torch.testing.assert_close(output[:, :-1], expected)
    torch.testing.assert_close(output[:, -1].sigmoid(), torch.full((24,), 0.25))


def test_cpu_fit_is_repeatable_ignores_test_data_and_reloads(tmp_path):
    torch = pytest.importorskip("torch")
    from mosamiat.modeling.lstm import Forecaster, Settings, fit

    hours = pd.date_range("2025-06-01", periods=180, freq="h", tz="UTC")
    rng = np.random.default_rng(13)
    values = pd.DataFrame(
        rng.uniform(1, 5, (len(hours), len(METRICS))), index=hours, columns=list(METRICS)
    )
    values["wind_direction_deg"] = np.resize([359.0, 1.0], len(hours))
    values["rain_probability"] = np.nan
    values.loc[hours[8:12], "temperature_c"] = np.nan
    values.loc[hours[66], "wind_speed_mps"] = 0
    values.loc[hours[84:89], "aqi_value"] = np.nan
    rain = pd.Series((np.arange(len(hours)) % 9 == 0).astype(float), index=hours)
    rain.iloc[61] = np.nan
    settings = Settings(hidden_size=4, max_epochs=3, batch_size=16, patience=2)
    model, history, epoch = fit(values, rain, hours[90], hours[140], "Asia/Karachi", settings)
    assert epoch == history.validation_loss.idxmin()
    assert model.prep.means[model.prep.columns.index("temperature_c")] == pytest.approx(
        values.temperature_c.iloc[:90].mean()
    )
    assert model.prep.rain_rate == pytest.approx(rain.iloc[:90].mean())
    altered = values.copy()
    altered.iloc[140:] = 888
    other, other_history, _ = fit(altered, rain, hours[90], hours[140], "Asia/Karachi", settings)
    pd.testing.assert_frame_equal(history, other_history)
    for key, weight in model.model.state_dict().items():
        torch.testing.assert_close(weight, other.model.state_dict()[key], rtol=0, atol=0)
    saved = tmp_path / "model.pt"
    model.save(saved)
    restored = Forecaster.load(saved)
    issue = hours[140:141]
    expected = model.predict(values, issue)
    pd.testing.assert_frame_equal(restored.predict(altered, issue), expected)
    pd.testing.assert_frame_equal(restored.predict(values.iloc[116:140], issue), expected)
    assert list(expected.target_at_utc) == list(hours[141:165])
    assert np.isfinite(expected[list(FORECAST_METRICS)]).all().all()
    assert (
        expected.rain_probability.between(0, 1).all()
        and expected.humidity_pct.between(0, 100).all()
    )
    assert (expected.wind_direction_deg.ge(0) & expected.wind_direction_deg.lt(360)).all()
    with pytest.raises(CollectionError, match="24 completed"):
        restored.predict(values, hours[8:9])
