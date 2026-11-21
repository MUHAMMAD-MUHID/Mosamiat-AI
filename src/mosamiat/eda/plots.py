"""Static scientific figures; rendering works without a desktop display."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".cache" / "matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from mosamiat.backfill import METRICS
from mosamiat.eda.analysis import AIR_METRICS, WEATHER_METRICS, Analysis

BLUE = "#22789a"
RED = "#bb4a38"


def save(figure, output: Path, name: str) -> str:
    figure.savefig(output / name, dpi=130, facecolor="white")
    plt.close(figure)
    return name


def facets(metrics: list[str], title: str):
    rows = (len(metrics) + 1) // 2
    figure, axes = plt.subplots(
        rows, 2, figsize=(12, rows * 2.3), squeeze=False, layout="constrained"
    )
    figure.suptitle(title, fontsize=14)
    for axis, metric in zip(axes.flat, metrics, strict=False):
        axis.set_title(metric.replace("_", " "), fontsize=10, loc="left")
        axis.tick_params(labelsize=8)
        axis.grid(alpha=0.18)
    for axis in list(axes.flat)[len(metrics) :]:
        axis.set_visible(False)
    return figure, list(axes.flat)[: len(metrics)]


def empty(axis, text="No available values") -> None:
    axis.text(0.5, 0.5, text, transform=axis.transAxes, ha="center", va="center", fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])


def series_plot(analysis: Analysis, output: Path, metrics: list[str], name: str, title: str) -> str:
    figure, axes = facets(metrics, title + " | red marks: IQR review flags")
    for axis, metric in zip(axes, metrics, strict=True):
        values = analysis.values[metric]
        axis.set_ylabel(METRICS[metric], fontsize=8)
        if not values.notna().any():
            empty(axis)
            continue
        if metric == "wind_direction_deg":
            axis.scatter(values.index, values, s=6, color=BLUE)
            axis.set_ylim(0, 360)
        else:
            axis.plot(values.index, values, color=BLUE, linewidth=1)
            flagged = analysis.anomalies[analysis.anomalies.metric.eq(metric)]
            axis.scatter(flagged.valid_at_utc, flagged.value, color=RED, s=15, zorder=3)
        if values.dropna().eq(0).all():
            axis.set_ylim(0, 1)
            axis.text(
                0.5,
                0.5,
                "All available values = 0",
                transform=axis.transAxes,
                ha="center",
                fontsize=9,
            )
        locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axis.set_xlabel("UTC valid time", fontsize=8)
    return save(figure, output, name)


def distributions(analysis: Analysis, output: Path) -> str:
    figure, axes = facets(list(METRICS), "Reference-value distributions | no imputation or removal")
    for axis, metric in zip(axes, METRICS, strict=True):
        values = analysis.values[metric].dropna()
        if metric == "wind_direction_deg":
            empty(axis, "Circular variable: see the wind rose")
        elif values.empty:
            empty(axis)
        else:
            bins = [0, 1] if values.eq(0).all() else min(20, max(1, int(np.sqrt(len(values)))))
            axis.hist(values, bins=bins, color=BLUE, alpha=0.85)
            axis.set_xlabel(METRICS[metric], fontsize=8)
            axis.set_ylabel("Hourly value count", fontsize=8)
    return save(figure, output, "distributions.png")


def profile_plot(analysis: Analysis, output: Path, daily: bool) -> str:
    table = analysis.daily_profile if daily else analysis.hourly_profile
    label = "daily mean of hourly values" if daily else "mean by local hour"
    note = "partial edge dates possible" if daily else "limited sample"
    figure, axes = facets(list(METRICS), f"Descriptive {label} | {note}; not seasonal evidence")
    for axis, metric in zip(axes, METRICS, strict=True):
        x = list(map(str, table.index)) if daily else table.index
        if metric == "wind_direction_deg":
            for component, color in [("sin", BLUE), ("cos", RED)]:
                axis.plot(x, table[f"wind_direction_{component}"], label=component, color=color)
            axis.legend(fontsize=7, loc="best")
            axis.set_ylabel("Mean circular components", fontsize=8)
        elif table[metric].notna().any():
            axis.plot(x, table[metric], color=BLUE, marker="." if len(table) < 10 else None)
            axis.set_ylabel(METRICS[metric], fontsize=8)
            if table[metric].dropna().eq(0).all():
                axis.set_ylim(0, 1)
                axis.text(
                    0.5,
                    0.5,
                    "All available values = 0",
                    transform=axis.transAxes,
                    ha="center",
                    fontsize=9,
                )
        else:
            empty(axis)
        axis.set_xlabel(
            f"Local date ({analysis.summary['timezone']})"
            if daily
            else f"Local hour ({analysis.summary['timezone']})",
            fontsize=8,
        )
        if daily and len(table) > 5:
            axis.xaxis.set_major_locator(plt.MaxNLocator(5))
        axis.tick_params(axis="x", labelrotation=25 if daily else 0)
    return save(figure, output, "daily_profile.png" if daily else "hourly_profile.png")


def heatmap(analysis: Analysis, output: Path, correlation: bool) -> str:
    table = analysis.correlations if correlation else analysis.daily_coverage.T
    figure, axis = plt.subplots(figsize=(12, 9), layout="constrained")
    cmap = plt.get_cmap("RdBu_r" if correlation else "Blues").with_extremes(bad="#dedede")
    matrix = np.ma.masked_invalid(table.to_numpy(dtype=float))
    plot = axis.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=-1 if correlation else 0,
        vmax=1 if correlation else 100,
    )
    axis.set_yticks(
        range(len(table.index)), [str(name).replace("_", " ") for name in table.index], fontsize=8
    )
    step = max(1, len(table.columns) // 16)
    ticks = list(range(0, len(table.columns), step))
    axis.set_xticks(
        ticks,
        [str(table.columns[i]).replace("_", " ") for i in ticks],
        rotation=60,
        ha="right",
        fontsize=8,
    )
    axis.set_title(
        "Pairwise Pearson correlation | grey: unavailable/constant; see sample counts"
        if correlation
        else "Non-null coverage by UTC date | percentage of included hours",
        fontsize=11,
    )
    figure.colorbar(
        plot, ax=axis, shrink=0.7, label="Correlation" if correlation else "Coverage (%)"
    )
    return save(figure, output, "correlations.png" if correlation else "coverage.png")


def wind_rose(analysis: Analysis, output: Path) -> str:
    figure, axis = plt.subplots(
        figsize=(7, 6), subplot_kw={"projection": "polar"}, layout="constrained"
    )
    moving = analysis.values.wind_speed_mps.gt(0) & analysis.values.wind_direction_deg.notna()
    radians = np.deg2rad(analysis.values.loc[moving, "wind_direction_deg"] % 360)
    # Center the north sector on zero, rather than splitting it at the wrap point.
    bins = 16
    width = 2 * np.pi / bins
    counts, _ = np.histogram(
        (radians + width / 2) % (2 * np.pi), bins=np.linspace(0, 2 * np.pi, bins + 1)
    )
    axis.bar(np.arange(bins) * width, counts, width=width * 0.9, color=BLUE, alpha=0.8)
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.set_title(
        f"Wind direction | {len(radians)} non-calm hourly values\n"
        "Calm and unknown-speed hours excluded",
        fontsize=11,
        pad=20,
    )
    axis.set_ylabel("Hourly value count", fontsize=8, labelpad=25)
    return save(figure, output, "wind_rose.png")


def render_charts(analysis: Analysis, output: Path) -> list[str]:
    with plt.rc_context(
        {"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False}
    ):
        return [
            heatmap(analysis, output, False),
            series_plot(
                analysis, output, WEATHER_METRICS, "weather_series.png", "Weather reference values"
            ),
            series_plot(
                analysis, output, AIR_METRICS, "air_quality_series.png", "CAMS air quality and UV"
            ),
            distributions(analysis, output),
            profile_plot(analysis, output, False),
            profile_plot(analysis, output, True),
            heatmap(analysis, output, True),
            wind_rose(analysis, output),
        ]
