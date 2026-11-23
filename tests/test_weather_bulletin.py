from datetime import timedelta

import pandas as pd
import pytest
from test_collect import CITY

from mosamiat.agents.alerts import RULES, evaluate, forecast_hours, new_episodes
from mosamiat.agents.ledger import Ledger
from mosamiat.backfill import METRICS
from mosamiat.collect import CollectionError

MILD = dict(
    temperature_c=20,
    humidity_pct=55,
    aqi_value=70,
    precipitation_mm=0,
    rain_probability=0.1,
    uv_index=2,
    wind_speed_mps=3,
    wind_direction_deg=90,
)


def payload(reference, **changes):
    stamp = pd.Timestamp(reference)
    values = MILD | changes
    return dict(
        run_id=stamp.isoformat(),
        reference_at_utc=stamp.isoformat(),
        issued_at_utc=(stamp + timedelta(minutes=5)).isoformat(),
        forecasts=[
            dict(
                target_at_utc=(stamp + timedelta(hours=h)).isoformat(),
                metric=k,
                value=v,
                unit=METRICS[k],
                model_name="candidate",
            )
            for k, v in values.items()
            for h in range(1, 25)
        ],
    )


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.key)
def test_threshold_includes_exact_boundary_and_preserves_distinct_episodes(rule):
    hours = forecast_hours(payload("2026-11-23T15:00Z"))
    assert not evaluate(hours)
    hours.loc[hours.index[[1, 2, 4]], rule.field] = rule.threshold
    hours.loc[hours.index[2], rule.field] = rule.elevated
    alerts = [a for a in evaluate(hours) if a["rule"] == rule.key]
    assert [(a["severity"], a["first"], a["last"]) for a in alerts] == [
        ("elevated", hours.index[1].isoformat(), hours.index[2].isoformat()),
        ("advisory", hours.index[4].isoformat(), hours.index[4].isoformat()),
    ]
    assert alerts[0]["peak"] == rule.elevated
    assert alerts[0]["peak_at"] == hours.index[2].isoformat()
    assert new_episodes(alerts, alerts) == []
    assert all(a["public"] and a["sensitive"] and a["workers"] for a in alerts)


def test_daily_alert_escalation_expiry_and_replay(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")

    def issue(time, aqi):
        data = payload(time, aqi_value=aqi)
        return ledger.bulletin(CITY, data, evaluate(forecast_hours(data)))

    first = issue("2026-11-23T13:00Z", 70)
    assert first["report_trigger"] == "daily" and "None of" in first["report"]
    new = issue("2026-11-23T14:00Z", 101)
    assert new["report_trigger"] == "new_alert" and "101" in new["report"]
    assert issue("2026-11-23T15:00Z", 102)["report"] == ""
    elevated = issue("2026-11-23T16:00Z", 151)
    assert elevated["new_alert_count"] == 1
    assert issue("2026-11-23T16:00Z", 151) == elevated
    assert issue("2026-11-23T17:00Z", 70)["alerts"] == []
    assert issue("2026-11-23T18:00Z", 151)["report_trigger"] == "new_alert"
    assert issue("2026-11-23T19:00Z", 151)["report_trigger"] == "daily"


def test_incomplete_forecast_cannot_be_reported_as_clear():
    data = payload("2026-11-23T15:00Z")
    data["forecasts"].pop()
    with pytest.raises(CollectionError, match="complete"):
        forecast_hours(data)
