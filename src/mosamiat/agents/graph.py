"""Issue an hourly Islamabad forecast and persist it in the Hopsworks Forecast Ledger."""

import argparse
import hashlib
import json
import sys
from functools import cache
from pathlib import Path
from typing import TypedDict

import httpx
import pandas as pd
import requests
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from mosamiat.agents.alerts import evaluate, forecast_hours
from mosamiat.agents.ledger import Ledger, publish, publish_bulletin
from mosamiat.agents.retraining import run_drift
from mosamiat.backfill import digest
from mosamiat.collect import City, CollectionError, collect
from mosamiat.modeling.inference import (
    MANIFEST,
    MODEL,
    VERSION,
    active_forecast,
    forecast,
    issue_features,
    read_policy,
    recent_history,
)
from mosamiat.modeling.registry import candidate_predictions, load_candidate, load_forecast_bundle
from mosamiat.storage import connect_project


class ForecastState(TypedDict, total=False):
    run_id: str
    reference_at_utc: str
    snapshot: str
    retry: bool
    payload: dict
    status: str
    alerts: list[dict]
    bulletin: dict
    drift_result: dict


def _publication_interrupted(error):
    if isinstance(error, (requests.exceptions.SSLError, requests.exceptions.ProxyError)):
        return False
    return isinstance(
        error,
        (
            ConnectionError,
            TimeoutError,
            httpx.NetworkError,
            httpx.TimeoutException,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ),
    )


def build_graph(
    city,
    ledger,
    project,
    now=lambda: pd.Timestamp.now(tz="UTC"),
    *,
    publish_inputs=None,
    trainer=None,
):
    def prepare(state):
        run_id = state["run_id"]
        existing = ledger.get(run_id)
        if existing:
            ledger.log(run_id, "forecast_reused", "Loaded the prepared forecast from the ledger")
            return {"payload": existing, "status": "prepared"}
        if state.get("retry"):
            raise CollectionError("No prepared forecast exists for this run ID")
        if state.get("snapshot"):
            path = Path(state["snapshot"])
        else:
            with httpx.Client(timeout=httpx.Timeout(20, connect=10)) as client:
                path = collect(city, Path("data/collection"), client)
        issued = now()
        reference = pd.Timestamp(state["reference_at_utc"])
        if issued.floor("h") != reference:
            raise CollectionError("The forecast hour expired during ingestion")
        history, retrieved = recent_history(path, city, issued)
        project_handle = project()
        registry = project_handle.get_model_registry()
        bundle = load_forecast_bundle(registry, MODEL, VERSION, MANIFEST)
        sample = issue_features(history, reference, city, bundle[1]["feature_definitions"])
        rows = forecast(history, reference, city, bundle, sample=sample)
        evaluation_sha256 = MANIFEST
        try:
            policy, checksum = read_policy(project_handle.get_dataset_api(), city)
            if policy:
                predictions = {}
                for item in policy["models"].values():
                    key = (item["name"], item["version"])
                    if key not in predictions:
                        candidate = load_candidate(registry, item)
                        predictions[key] = candidate_predictions(
                            item["family"], candidate, history, reference, city
                        )
                rows = active_forecast(rows, policy, predictions)
                evaluation_sha256 = checksum
        except Exception as error:
            ledger.log(run_id, "serving_fallback", type(error).__name__)
        inputs = {}
        if publish_inputs is not None:
            selected = sample[bundle[1]["features"]].iloc[0]
            inputs["input_features"] = {
                key: float(value) if pd.notna(value) else None for key, value in selected.items()
            }
        issued = now().floor("us")
        if issued.floor("h") != reference:
            raise CollectionError("The forecast hour expired before inference completed")
        source_hash = hashlib.sha256(
            (digest(path / "weather.json") + digest(path / "air_quality.json")).encode()
        ).hexdigest()
        payload = ledger.prepare(
            dict(
                run_id=run_id,
                city_id=city.city_id,
                reference_at_utc=reference.isoformat(),
                issued_at_utc=issued.isoformat(),
                retrieved_at_utc=retrieved.isoformat(),
                snapshot_id=path.name,
                snapshot_path=str(path.resolve()),
                source_sha256=source_hash,
                evaluation_sha256=evaluation_sha256,
                policy_version=1,
                forecasts=rows,
                **inputs,
            )
        )
        ledger.log(run_id, "prepared", "24 horizons and eight fields")
        return {"payload": payload, "status": "prepared"}

    def persist(state):
        store = project().get_feature_store()
        if publish_inputs is not None:
            publish_inputs(store, state["payload"])
        publish(store, state["payload"])
        ledger.log(state["run_id"], "verified", "192 remote forecast rows matched")
        return {"status": "verified"}

    def check_alerts(state):
        alerts = evaluate(forecast_hours(state["payload"]))
        ledger.log(state["run_id"], "alerts_checked", f"{len(alerts)} forecast episodes")
        return {"alerts": alerts}

    def weather_report(state):
        bulletin = ledger.bulletin(city, state["payload"], state["alerts"])
        publish_bulletin(project().get_feature_store(), bulletin)
        ledger.log(state["run_id"], "bulletin_verified", bulletin["report_trigger"])
        return {"bulletin": bulletin}

    def drift(state):
        payload = state["payload"]
        policy = None
        if payload["evaluation_sha256"] != MANIFEST:
            policy, checksum = read_policy(
                project().get_dataset_api(), city, payload["evaluation_sha256"]
            )
            if policy is None or checksum != payload["evaluation_sha256"]:
                raise CollectionError("Missing drift profile for the forecast serving selection")
        return {
            "drift_result": run_drift(
                city, payload, ledger, project().get_feature_store(), trainer=trainer, policy=policy
            )
        }

    def audited(name, node):
        def run(state):
            ledger.log(state["run_id"], "node_started", name)
            try:
                return node(state)
            except Exception as error:
                ledger.log(state["run_id"], "node_failed", f"{name}: {type(error).__name__}")
                raise

        return run

    # Both publication steps upsert and read back the original checkpointed payload.
    publication_retry = RetryPolicy(max_attempts=3, retry_on=_publication_interrupted)
    graph = StateGraph(ForecastState)
    for name, node in (
        ("forecast", prepare),
        ("save_forecasts", persist),
        ("alert", check_alerts),
        ("weather_report", weather_report),
        ("drift", drift),
    ):
        policy = publication_retry if name in {"save_forecasts", "weather_report"} else None
        graph.add_node(name, audited(name, node), retry_policy=policy)
    graph.add_edge(START, "forecast")
    graph.add_edge("forecast", "save_forecasts")
    graph.add_edge("save_forecasts", "alert")
    graph.add_edge("alert", "weather_report")
    graph.add_edge("weather_report", "drift")
    graph.add_edge("drift", END)
    return graph.compile()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--snapshot", type=Path, help="Completed collector snapshot from this UTC hour"
    )
    mode.add_argument("--retry", help="Republish the original prepared forecast by run ID")
    mode.add_argument("--history", metavar="RUN_ID", help="Read the local event history as JSON")
    parser.add_argument("--ledger", type=Path, default=Path("data/forecasts/ledger.sqlite3"))
    args = parser.parse_args(argv)
    if args.history is not None:
        events = Ledger.history(args.ledger, args.history)
        print(json.dumps(events, indent=2))
        return 0 if events else 1
    city = City.load(Path("configs/city.toml"))
    reference = pd.Timestamp.now(tz="UTC").floor("h")
    run_id = args.retry or f"{city.city_id}_{reference:%Y%m%dT%HZ}"
    ledger = Ledger(args.ledger)
    try:
        from mosamiat.scheduled import train_on_drift

        ledger.log(run_id, "started", "retry" if args.retry else "hourly forecast")
        project = cache(connect_project)
        graph = build_graph(
            city,
            ledger,
            project,
            trainer=lambda city, request, root: train_on_drift(project(), city, request, root),
        )
        state = graph.invoke(
            dict(
                run_id=run_id,
                retry=bool(args.retry),
                snapshot=str(args.snapshot or ""),
                reference_at_utc=reference.isoformat(),
            )
        )
        ledger.log(run_id, "completed", "All four agents finished")
    except Exception as error:
        message = str(error) if isinstance(error, CollectionError) else type(error).__name__
        ledger.log(run_id, "failed", message)
        print(f"Forecast {run_id} failed: {message}", file=sys.stderr)
        return 1
    print(
        f"{state['status']}: {run_id} ({len(state['payload']['forecasts'])} forecast rows, "
        f"{len(state['alerts'])} alert episodes, report {state['bulletin']['report_trigger']}, "
        f"{state['drift_result']['drift_count']} drift thresholds crossed)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
