"""Interactive, read-only Islamabad forecast desk."""

from base64 import b64encode
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from mosamiat.collect import City
from mosamiat.dashboard import FIELDS, load_outlook

CITY = City.load(Path("configs/city.toml"))
st.set_page_config(page_title=f"{CITY.name} weather · Mosamiat", page_icon="🌤", layout="wide")
st.html(Path("assets/dashboard.css"))
# Bundle the local illustration with the stylesheet for offline previews too.
photo = b64encode(Path("assets/islamabad-dusk.webp").read_bytes()).decode("ascii")
st.html(f"<style>.hero {{ --city-photo: url(data:image/webp;base64,{photo}); }}</style>")

PRECISION = {"aqi_value": 0, "rain_probability": 0, "humidity_pct": 0, "wind_direction_deg": 0}


@st.cache_data(ttl=300, show_spinner=False)
def latest():
    return load_outlook(CITY)


def stamp(t):
    return t.strftime("%d %b %Y, %H:%M PKT")


def local(t):
    return pd.Timestamp(t).tz_convert(CITY.timezone)


def display_units(hours, temperature, wind):
    converted = hours.copy()
    units = {key: unit for key, (_, unit) in FIELDS.items()}
    if temperature == "°F":
        converted["temperature_c"] = converted.temperature_c * 1.8 + 32
    factor = {"m/s": 1, "km/h": 3.6, "mph": 2.236936292}[wind]
    converted["wind_speed_mps"] *= factor
    units.update(temperature_c=temperature, wind_speed_mps=wind)
    return converted, units


def label(key):
    return FIELDS[key][0]


def pretty(key, value, units):
    unit = units[key]
    number = f"{value:.{PRECISION.get(key, 1)}f}"
    if key == "uv_index":
        return number
    return number + ("" if unit in {"%", "°"} else " ") + unit


def alert_status(alert, now):
    hour = now.floor("h")
    if local(alert["first"]) > hour:
        return "Upcoming"
    if local(alert["last"]) < hour:
        return "Ended"
    return "Active"


def trend(hours, metric, units):
    clock = hours.index.strftime("%d %b %H:%M").tolist()
    values = hours[metric]
    rows = pd.DataFrame({"Hour": clock, "Value": values.to_numpy()})
    title = f"{label(metric)} ({units[metric]})"
    bearing = metric == "wind_direction_deg"
    ticks = clock[:: max(1, (len(clock) + 5) // 6)]
    # Use explicit local labels because browser timezone settings may differ.
    x = {
        "field": "Hour",
        "type": "ordinal",
        "sort": clock,
        "axis": {
            "title": None,
            "values": ticks,
            "labelExpr": "slice(datum.label, -5)",
            "labelAngle": 0,
            "labelPadding": 9,
        },
    }
    y = {
        "field": "Value",
        "type": "quantitative",
        "title": title,
        "scale": {"domain": [0, 360]} if bearing else {"zero": False},
        "axis": {"tickCount": 5, "titlePadding": 10},
    }
    tip = [
        {"field": "Hour", "type": "nominal", "title": "Forecast hour (PKT)"},
        {"field": "Value", "type": "quantitative", "title": title, "format": ".2f"},
    ]
    layers = []
    if not bearing:
        layers.append(
            {
                "mark": {
                    "type": "area",
                    "clip": True,
                    "color": {
                        "gradient": "linear",
                        "x1": 0,
                        "x2": 0,
                        "y1": 0,
                        "y2": 1,
                        "stops": [
                            {"offset": 0, "color": "#9cbdf0"},
                            {"offset": 1, "color": "#f4f8fe"},
                        ],
                    },
                },
                "encoding": {"y2": {"datum": float(values.min())}},
            }
        )
        layers.append({"mark": {"type": "line", "color": "#285bb0", "strokeWidth": 3}})
    layers.append(
        {
            "mark": {
                "type": "point",
                "filled": True,
                "size": 55,
                "fill": "#285bb0",
                "stroke": "white",
                "strokeWidth": 1.3,
            },
            "encoding": {"tooltip": tip},
        }
    )
    layers.append(
        {
            "mark": {"type": "bar", "opacity": 0},
            "encoding": {"y": {"value": 0}, "y2": {"value": 170}, "tooltip": tip},
        }
    )
    st.vega_lite_chart(
        rows,
        {
            "height": 170,
            "background": "transparent",
            "encoding": {"x": x, "y": y},
            "layer": layers,
            "config": {
                "view": {"stroke": None},
                "axis": {
                    "domain": False,
                    "ticks": False,
                    "gridColor": "#e1e7ef",
                    "labelColor": "#42576b",
                    "titleColor": "#42576b",
                    "labelFontSize": 12,
                },
            },
        },
        theme=None,
        width="stretch",
    )


def overview(outlook, hours, units, now):
    valid = hours.index[hours.index >= now.floor("h")]
    focus = valid[0] if len(valid) else hours.index[-1]
    hour = hours.loc[focus]
    left, right = st.columns([1.8, 1], gap="medium")
    with left:
        st.html(f"""
        <section class="hero"><div><div class="kicker">24-hour forecast · Pakistan</div>
        <h1>{escape(CITY.name)}</h1></div><div>
        <div class="hero-value">{hour.temperature_c:.1f}<span>{units["temperature_c"]}</span></div>
        <p class="moment">Forecast for {focus:%a, %d %b · %H:%M} PKT</p>
        <div class="range">24-hour low <b>{hours.temperature_c.min():.1f}°</b>
        &nbsp; · &nbsp; High <b>{hours.temperature_c.max():.1f}°</b></div></div></section>""")
    with right:
        items = (
            ("Peak air quality index", "aqi_value", hours.aqi_value.max()),
            ("Total precipitation", "precipitation_mm", hours.precipitation_mm.sum()),
            ("Peak rain chance", "rain_probability", hours.rain_probability.max()),
            ("Strongest wind", "wind_speed_mps", hours.wind_speed_mps.max()),
        )
        summary = "".join(
            f"<div><dt>{name}</dt><dd>{pretty(key, value, units)}</dd></div>"
            for name, key, value in items
        )
        alerts = outlook.alerts
        pending = [a for a in alerts if alert_status(a, now) != "Ended"]
        priority = sorted(
            pending,
            key=lambda a: (
                alert_status(a, now) != "Active",
                a["severity"] != "elevated",
                a["first"],
            ),
        )
        if priority:
            first = priority[0]
            note = f"{alert_status(first, now)} · {escape(first['label'])}<br>{local(first['first']):%d %b %H:%M} – {local(first['last']):%d %b %H:%M} PKT"
        else:
            note = (
                f"All {len(alerts)} forecast alert periods have ended."
                if alerts
                else "No alert thresholds crossed in this outlook."
            )
        st.html(f"""<section class="desk-card signals"><div class="kicker">{hours.index[0]:%d %b} – {hours.index[-1]:%d %b} · PKT</div>
            <h2>Forecast highlights</h2><dl>{summary}</dl>
            <p class="notice {"" if priority else "quiet"}">{note}</p></section>""")
    with st.container(border=True, key="panel_trend"):
        heading, control = st.columns([2, 1])
        heading.subheader("Forecast trend")
        heading.caption(f"{hours.index[0]:%a %d %b, %H:%M} – {hours.index[-1]:%a %d %b, %H:%M} PKT")
        metric = control.selectbox(
            "Trend metric", list(FIELDS), format_func=label, key="trend_metric"
        )
        trend(hours, metric, units)


def explorer(hours, units):
    st.subheader("Hourly explorer")
    st.caption("Explore the forecast from its first hour. All times are in Pakistan time (UTC+5).")
    with st.container(border=True, key="panel_hourly"):
        a, b = st.columns(2)
        metric = a.selectbox("Plot metric", list(FIELDS), format_func=label)
        count = b.segmented_control(
            "Forecast window",
            [6, 12, 24],
            format_func=lambda x: f"{x} hours",
            default=24,
            selection_mode="single",
            required=True,
        )
        window = hours.iloc[:count]
        st.caption(f"{stamp(window.index[0])} – {stamp(window.index[-1])}")
        trend(window, metric, units)
        table = window.rename(columns={k: f"{label(k)} ({units[k]})" for k in FIELDS})
        table.insert(0, "Forecast hour (PKT)", window.index.strftime("%Y-%m-%d %H:%M"))
        st.dataframe(
            table.round(2), hide_index=True, width="stretch", height=min(345, 36 * (count + 1) + 4)
        )
        st.download_button(
            "Download hourly CSV",
            table.to_csv(index=False, float_format="%.4f").encode(),
            file_name=f"{CITY.city_id}-{window.index[0]:%Y%m%d-%H%M}-{count}h.csv",
            mime="text/csv",
            on_click="ignore",
        )
    with st.container(border=True, key="panel_compare"):
        st.subheader("Compare two hours")
        times = hours.index.strftime("%a %d %b · %H:%M PKT").tolist()
        a, b = st.columns(2)
        first = a.selectbox("First hour", times, key="first_hour")
        second = b.selectbox("Second hour", times, index=12, key="second_hour")
        if first == second:
            st.caption("Both selections refer to the same forecast hour.")
        data = [
            {
                "Metric": label(k),
                "First hour": pretty(k, hours[k].iloc[times.index(first)], units),
                "Second hour": pretty(k, hours[k].iloc[times.index(second)], units),
            }
            for k in FIELDS
        ]
        st.dataframe(pd.DataFrame(data), hide_index=True, width="stretch", height="content")


def alert_center(outlook, now):
    st.subheader("Alert center")
    st.caption("Forecast threshold crossings and the guidance published for each period.")
    stage, level, person = st.columns(3)
    period = stage.selectbox("Alert status", ["All periods", "Active", "Upcoming", "Ended"])
    severity = level.selectbox("Severity", ["All levels", "Advisory", "Elevated"])
    audience = person.selectbox(
        "Guidance for", ["General public", "Sensitive groups", "Outdoor workers"]
    )
    group = {
        "General public": "public",
        "Sensitive groups": "sensitive",
        "Outdoor workers": "workers",
    }[audience]
    matches = [
        a
        for a in outlook.alerts
        if (period == "All periods" or alert_status(a, now) == period)
        and (severity == "All levels" or a["severity"].title() == severity)
    ]
    if not matches:
        st.info(
            "No alerts match these filters."
            if outlook.alerts
            else "No alert thresholds were crossed in this forecast."
        )
    for i, a in enumerate(matches):
        with st.container(border=True, key=f"panel_alert_{i}"):
            severity_name = escape(a["severity"])
            st.html(
                f'<div class="alert-row"><strong class="alert-title">{escape(a["label"])}</strong>'
                f'<span class="tag {severity_name}">{severity_name.title()}</span>'
                f'<span class="tag">{alert_status(a, now)}</span></div>'
            )
            st.caption(f"{stamp(local(a['first']))} – {stamp(local(a['last']))}")
            st.write(a[group])
            with st.expander("Forecast evidence"):
                k = a["metric"]
                scale = 100 if k == "rain_probability" else 1
                unit = FIELDS[k][1]
                st.write(
                    f"{label(k)} reached {a['peak'] * scale:.1f} {unit} in the forecast for "
                    f"{stamp(local(a['peak_at']))}. The alert threshold is {a['comparison']} {a['threshold'] * scale:g} {unit}."
                )


@st.fragment(run_every="5m")
def page():
    brand, options = st.columns([3, 1], vertical_alignment="center")
    brand.html(
        '<div class="mast"><i class="insignia" aria-hidden="true"></i><strong>Mosamiat</strong><small>Weather desk</small></div>'
    )
    with options.popover("Desk settings", width="stretch"):
        temperature = st.radio("Temperature unit", ["°C", "°F"], horizontal=True)
        wind = st.radio("Wind speed unit", ["m/s", "km/h", "mph"], horizontal=True)
        if st.button("Refresh forecast", width="stretch"):
            latest.clear()
    try:
        with st.spinner("Loading verified forecast…"):
            outlook, source = latest()
    except Exception:
        st.warning("Forecast unavailable. Use Desk settings → Refresh forecast to check again.")
        return
    now = pd.Timestamp.now(tz=CITY.timezone)
    hours, units = display_units(outlook.hours, temperature, wind)
    expired = now.floor("h") > hours.index[-1]
    late = now.floor("h") > outlook.issued.floor("h") + pd.Timedelta(hours=1)
    status = (
        "Forecast period ended" if expired else "Update overdue" if late else "Verified forecast"
    )
    status_class = "aged" if expired or late else ""
    st.html(
        f'<div class="statusline"><strong class="{status_class}">{status}</strong>'
        f"<span>Issued {stamp(outlook.issued)}</span><span>· &nbsp;{escape(source)}</span></div>"
    )
    home, hourly, alerts, report = st.tabs(["Overview", "Hourly explorer", "Alerts", "Bulletin"])
    with home:
        overview(outlook, hours, units, now)
    with hourly:
        explorer(hours, units)
    with alerts:
        alert_center(outlook, now)
    with report:
        with st.container(border=True, key="panel_report"):
            if outlook.bulletin_status == "failed":
                st.warning("Bulletin publication failed for this forecast.")
            elif outlook.bulletin_status == "pending":
                st.caption("Bulletin verification is pending for this forecast.")
            if outlook.report:
                st.caption(
                    f"Published for the {stamp(outlook.report_issued)} forecast · Original units"
                )
                st.markdown(outlook.report)
            elif outlook.bulletin_status == "verified":
                st.subheader("Weather bulletin")
                st.info("A verified weather report is not yet available.")
    with st.expander("Forecast sources and credits"):
        st.write(
            f"Verified {stamp(outlook.verified)} · {source}. Weather inputs from Open-Meteo and air-quality inputs from Copernicus CAMS."
        )
        st.caption(f"Forecast ID · {outlook.run_id}")
        st.markdown(
            "City photograph by [Imrankhakwani](https://commons.wikimedia.org/wiki/File:Faisal_Mosque_at_dusk_seen_from_Margalla_Hills.jpg), licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Resized and cropped for display."
        )


page()
