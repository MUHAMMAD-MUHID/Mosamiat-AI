"""Render the verified forecast and alert evidence in the city's local time."""

import pandas as pd


def render(city, payload, hours, alerts):
    def local(stamp):
        return pd.Timestamp(stamp).tz_convert(city.timezone).strftime("%d %b %H:%M")

    peak = hours.temperature_c.idxmax()
    windy = hours.wind_speed_mps.idxmax()
    bearing = hours.at[windy, "wind_direction_deg"]
    direction = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")[int((bearing + 22.5) // 45) % 8]
    lines = [
        f"# {city.name} weather outlook",
        f"Forecast issued {local(payload['issued_at_utc'])} ({city.timezone}). "
        f"Hourly outlook from {local(hours.index[0])} through {local(hours.index[-1])}.",
        f"{city.name} is forecast to reach {hours.temperature_c.max():.1f} °C at {local(peak)}, "
        f"with a low of {hours.temperature_c.min():.1f} °C. "
        f"The highest forecast US AQI is {hours.aqi_value.max():.0f} at {local(hours.aqi_value.idxmax())}.",
        f"Total precipitation over the 24 hourly targets is forecast at {hours.precipitation_mm.sum():.1f} mm. "
        f"Relative humidity ranges from {hours.humidity_pct.min():.0f}% to {hours.humidity_pct.max():.0f}%, "
        f"and the UV index peaks at {hours.uv_index.max():.1f}.",
    ]
    if hours.wind_speed_mps.max() > 0:
        lines.append(
            f"The strongest wind is forecast from the {direction} at "
            f"{hours.wind_speed_mps.max():.1f} m/s around {local(windy)}."
        )
    else:
        lines.append("Winds are forecast to be calm throughout the period.")
    if all(
        r["model_name"] == "training_rain_frequency"
        for r in payload["forecasts"]
        if r["metric"] == "rain_probability"
    ):
        lines.append(
            f"The hourly rain estimate of {100 * hours.rain_probability.iloc[0]:.1f}% "
            "uses the training-period rain frequency and cannot identify when rain is most likely."
        )
    else:
        lines.append(
            f"The highest hourly rain probability is {100 * hours.rain_probability.max():.0f}%."
        )
    if alerts:
        lines.append("## Forecast alerts")
        for alert in alerts:
            scale = 100 if alert["metric"] == "rain_probability" else 1
            unit = {"US_AQI": "US AQI", "degC": "°C", "1": "%" if scale == 100 else "UV index"}.get(
                alert["unit"], alert["unit"]
            )
            lines.append(
                f"**{alert['label']}** · {local(alert['first'])} to {local(alert['last'])}. "
                f"The forecast reaches {alert['peak'] * scale:.1f} {unit} at {local(alert['peak_at'])}. "
                f"{alert['public']}\n\n"
                f"- **Sensitive groups** · {alert['sensitive']}\n"
                f"- **Outdoor workers** · {alert['workers']}"
            )
    else:
        lines.append("None of the project's alert thresholds were reached in this forecast.")
    lines.append(
        "Based on Mosamiat forecasts using Open-Meteo weather and Copernicus CAMS air-quality inputs. "
        "Conditions can change between hourly updates."
    )
    return "\n\n".join(lines) + "\n"
