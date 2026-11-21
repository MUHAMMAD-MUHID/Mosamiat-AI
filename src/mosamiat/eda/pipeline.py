"""Publish a reproducible EDA report from integrity-checked historical snapshots."""

import argparse
import base64
import html
import json
import sys
from pathlib import Path

from mosamiat.backfill import digest, write_json
from mosamiat.collect import City, CollectionError
from mosamiat.eda.analysis import Analysis, analyze
from mosamiat.reference import load_reference


def findings(analysis: Analysis, city: City) -> str:
    summary = analysis.summary
    rain = summary["rain"]
    wind = summary["wind"]
    lines = [
        f"# Module 05: EDA findings for {city.name}",
        "",
        f"Reference window: {summary['valid_from_utc']} through {summary['valid_until_utc']}.",
        f"Expected hourly timestamps: {summary['expected_hours']}; source timestamps: "
        f"{summary['source_hours']}; missing timestamps: {summary['missing_timestamps']}.",
        f"Duplicate metric/hour rows: {summary['duplicate_metric_hours']} (overlapping revisions are rejected).",
        "",
        "## Metric coverage and extremes",
        "",
        "| Metric | Unit | Present | Missing % | Longest gap (h) | Min | Max | IQR review flags |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric, row in analysis.metrics.iterrows():
        low = f"{row.minimum:.4g}" if row.present else "unavailable"
        high = f"{row.maximum:.4g}" if row.present else "unavailable"
        lines.append(
            f"| {metric} | {row.unit} | {int(row.present)} | {row.missing_pct:.2f} | "
            f"{int(row.longest_gap_hours)} | {low} | {high} | {int(row.iqr_review_count)} |"
        )
    lines += [
        "",
        "## Rain, AQI and wind",
        "",
        f"Precipitation: {rain['zero_hours']} exact-zero hours and {rain['positive_hours']} "
        f"positive hours among {rain['present_hours']} available hours. Observed sum: "
        f"{rain['observed_sum_mm']} mm. Precipitation includes snow; a positive value is not a "
        "validated rain-event label. Daily sums in daily_precipitation.csv mark partial local days.",
        "Historical rain probability is reported separately and is never imputed from precipitation.",
        f"AQI scales present: {', '.join(summary['aqi_scales']) or 'none'}. "
        "Source-specific coverage is in sources.csv; this is model-derived CAMS AQI, not a station feed.",
        f"Wind: {wind['count']} non-calm directions, {wind['calm_hours']} calm hours; "
        f"circular mean {wind['mean_direction_deg']} degrees, resultant length {wind['resultant_length']}. "
        "Arithmetic means and Pearson correlations of raw degrees are omitted. "
        "Wind sine/cosine components are used in profiles and correlations.",
        "",
        "## Temporal patterns, correlations and review flags",
        "",
        f"The sample spans {summary['local_dates']} local dates in "
        f"{len(summary['calendar_months'])} calendar month(s). Local-hour and daily means include "
        "per-metric sample counts in the exported tables. Daily precipitation means are mean hourly "
        "amounts, not daily totals. Edge dates can be incomplete. These are descriptive profiles, "
        "not evidence for annual seasonality or forecast skill.",
        "Pairwise Pearson correlations require at least three complete pairs and non-constant "
        "values. correlation_counts.csv provides support for each coefficient. Autocorrelation, "
        "small samples and shared model sources limit interpretation.",
        "IQR review flags use Q1 - 1.5 IQR and Q3 + 1.5 IQR, at least four values and nonzero "
        "IQR. No flags are computed for raw wind direction, all-null, constant or zero-IQR series. "
        "No flagged values are deleted. Review extremes against provenance before changing data.",
        "",
        "## Limitations and drift risks",
        "",
        *[f"- {warning}" for warning in summary["warnings"]],
        "- More history and accepted labels are needed before model comparison. Validate coverage "
        "and provider quotas before expanding the download.",
        "- Changes in model versions, resolution, provider coverage or any future station sensor "
        "can shift distributions. Drift monitoring must distinguish these from real weather changes.",
        "- Compare future training and monitoring windows by season and source. A short sample "
        "cannot provide production drift thresholds or representative extremes.",
        "",
        "Attribution: Open-Meteo, ECMWF ERA5, Copernicus CAMS.",
        "",
    ]
    return "\n".join(lines)


def html_report(analysis: Analysis, city: City, text: str, output: Path, charts: list[str]) -> str:
    images = "\n".join(
        f'<figure><img alt="{html.escape(name.replace("_", " "))}" '
        f'src="data:image/png;base64,{base64.b64encode((output / name).read_bytes()).decode()}">'
        f"<figcaption>{html.escape(name)}</figcaption></figure>"
        for name in charts
    )
    sections = []
    for paragraph in text.split("\n\n"):
        if paragraph.startswith("# ") or paragraph.startswith("|"):
            continue
        if paragraph.startswith("## "):
            sections.append(f"<h2>{html.escape(paragraph[3:])}</h2>")
            if paragraph.startswith("## Metric coverage"):
                columns = [
                    "unit",
                    "present",
                    "missing_pct",
                    "longest_gap_hours",
                    "minimum",
                    "mean",
                    "maximum",
                    "iqr_review_count",
                ]
                sections.append(
                    '<div class="scroll">'
                    + analysis.metrics[columns].round(4).to_html(na_rep="unavailable", border=0)
                    + "</div>"
                )
        elif paragraph.startswith("- "):
            sections.append(
                "<ul>"
                + "".join(f"<li>{html.escape(line[2:])}</li>" for line in paragraph.splitlines())
                + "</ul>"
            )
        else:
            sections.append(f"<p>{html.escape(paragraph)}</p>")
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Mosamiat EDA — {html.escape(city.name)}</title>"
        "<style>body{font:16px/1.6 system-ui;max-width:1180px;margin:40px auto;padding:0 24px;"
        "color:#173342}h1{line-height:1.2}.note{background:#fff1ce;padding:16px;border-radius:8px}"
        "img{max-width:100%;height:auto}figure{margin:32px 0}figcaption{color:#526671}"
        "table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border-bottom:1px solid #ddd;"
        "text-align:right}th:first-child,td:first-child{text-align:left}pre{white-space:pre-wrap;"
        "font:14px/1.6 system-ui}.scroll{overflow-x:auto}</style>"
        f"<h1>Mosamiat · {html.escape(city.name)} EDA</h1>"
        '<p class="note">Retrospective ERA5/CAMS provider reference data. '
        "Seasonal patterns and model readiness require more evidence.</p>"
        + "\n".join(sections)
        + images
        + "</html>"
    )


def build(dataset: Path, output: Path, city: City) -> Path:
    reference = load_reference(dataset, city)
    reference.protect_output(dataset, output)
    analysis = analyze(reference, city.timezone)
    # Import plotting only when publishing, so audit calculations stay lightweight.
    from mosamiat.eda.plots import render_charts

    manifest = {
        "schema_version": 1,
        "status": "building",
        "city_id": city.city_id,
        "is_ground_truth": False,
        "operational_point_in_time_safe": False,
        "provenance": reference.provenance(dataset),
        "settings": {
            "iqr_multiplier": analysis.summary["iqr_multiplier"],
            "minimum_correlation_pairs": analysis.summary["minimum_correlation_pairs"],
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".lock"
    try:
        lock_stream = lock.open("x")
    except FileExistsError as error:
        raise CollectionError("EDA output is locked by another writer") from error
    try:
        with lock_stream:
            write_json(output / "manifest.json", manifest)
            tables = {
                "metrics.csv": analysis.metrics,
                "anomalies.csv": analysis.anomalies,
                "sources.csv": analysis.sources,
                "correlations.csv": analysis.correlations,
                "correlation_counts.csv": analysis.correlation_counts,
                "hourly_profile.csv": analysis.hourly_profile,
                "daily_profile.csv": analysis.daily_profile,
                "daily_coverage.csv": analysis.daily_coverage,
                "daily_precipitation.csv": analysis.daily_precipitation,
            }
            for name, table in tables.items():
                table.to_csv(output / name, index=name not in {"anomalies.csv", "sources.csv"})
            (output / "summary.json").write_text(
                json.dumps(analysis.summary, indent=2, allow_nan=False), encoding="utf-8"
            )
            charts = render_charts(analysis, output)
            text = findings(analysis, city)
            (output / "findings.md").write_text(text, encoding="utf-8")
            (output / "report.html").write_text(
                html_report(analysis, city, text, output, charts), encoding="utf-8"
            )
            files = [*tables, "summary.json", "findings.md", "report.html", *charts]
            manifest.update(
                status="complete",
                expected_hours=analysis.summary["expected_hours"],
                charts=charts,
                sha256={name: digest(output / name) for name in files},
            )
            write_json(output / "manifest.json", manifest)
    except Exception:
        manifest["status"] = "failed"
        write_json(output / "manifest.json", manifest)
        raise
    finally:
        lock.unlink()
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/eda"))
    parser.add_argument("--city-config", type=Path, default=Path("configs/city.toml"))
    args = parser.parse_args(argv)
    try:
        destination = build(args.dataset, args.output, City.load(args.city_config))
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        print(f"EDA failed: {error}", file=sys.stderr)
        return 1
    print(f"Published EDA report: {destination / 'report.html'}")
    print("REFERENCE DATA ONLY: inspect findings and sample support before modeling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
