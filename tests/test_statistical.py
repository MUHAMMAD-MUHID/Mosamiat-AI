import json

import numpy as np
import pandas as pd
import pytest

from mosamiat.collect import CollectionError
from mosamiat.modeling.statistical import daily_terms, fit, predict


def keys(issue):
    return pd.MultiIndex.from_arrays(
        [[issue] * 24, pd.date_range(issue + pd.Timedelta(hours=1), periods=24, freq="h")],
        names=["reference_at_utc", "target_at_utc"],
    )


def test_statistical_fit_needs_usable_daily_history():
    times = pd.date_range("2025-01-01", periods=100, freq="h", tz="UTC")
    series = pd.Series(1.0, index=times, name="temperature_c")
    with pytest.raises(CollectionError, match="72 observed training"):
        fit(series.iloc[:48])
    with pytest.raises(CollectionError, match="constant"):
        fit(series)
    with pytest.raises(CollectionError, match="24 UTC hours"):
        fit(series[series.index.hour != 5])


def test_serialized_forecast_matches_statsmodels_with_gaps_and_no_future_access():
    SARIMAX = pytest.importorskip("statsmodels.tsa.statespace.sarimax").SARIMAX
    times = pd.date_range("2025-01-01", periods=200, freq="h", tz="UTC")
    rng = np.random.default_rng(72)
    series = pd.Series(
        20 + 4 * np.sin(2 * np.pi * times.hour / 24) + rng.normal(size=200),
        index=times,
        name="temperature_c",
    )
    training = series.iloc[:144].copy()
    training.iloc[[50, 51, -2, -1]] = np.nan
    params = json.loads(json.dumps(fit(training)))
    model = SARIMAX(training, order=(1, 0, 0), trend="n", exog=daily_terms(training.index))
    filtered = model.filter([params[name] for name in model.param_names])
    # Step 1 predicts the unavailable issue hour; step 2 is horizon 1.
    expected = filtered.forecast(25, exog=daily_terms(times[144:169])).iloc[1:]
    future = pd.concat([training, series.iloc[144:] + 1000])
    actual = predict(future, keys(times[144]), params)
    np.testing.assert_allclose(actual, expected, rtol=1e-7)
    pd.testing.assert_series_equal(actual, predict(training, keys(times[144]), params))
    # The next issue can use the newly completed hour, without refitting parameters.
    assert not np.allclose(
        predict(future, keys(times[145]), params), predict(training, keys(times[145]), params)
    )


def test_humidity_forecast_is_bounded():
    times = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    history = pd.Series(50.0, index=times, name="humidity_pct")
    params = {term: 0.0 for term in daily_terms(times)}
    params.update({"const": 120.0, "ar.L1": 0.0})
    assert predict(history, keys(times[-1]), params).eq(100).all()
    with pytest.raises(CollectionError, match="chronological"):
        predict(history.iloc[::-1], keys(times[-1]), params)
