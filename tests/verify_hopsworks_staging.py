"""Run synthetic reliability checks against explicitly identified staging Hopsworks."""

import argparse
import os
import sys
from dataclasses import replace
from functools import partial
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.parse import quote
from uuid import uuid4

import httpx
from dotenv import dotenv_values
from hopsworks_common import client
from hopsworks_common.job import Job
from test_collect import CITY
from test_weather_bulletin import payload

from mosamiat.agents import ledger as publication
from mosamiat.agents.graph import build_graph
from mosamiat.dashboard import _cloud_outlook
from mosamiat.modeling.inference import MANIFEST
from mosamiat.scheduled import ScheduledLedger
from mosamiat.storage import connect_project, hostname

FORECAST = "mosamiat_forecast_ledger"
BULLETIN = "mosamiat_weather_bulletins"


def staging_settings(expected_id):
    if expected_id <= 0 or expected_id == 59511:
        raise SystemExit("Provide the isolated staging project ID. Production is excluded.")
    keys = ("HOPSWORKS_HOST", "HOPSWORKS_PROJECT", "HOPSWORKS_API_KEY")
    values = dotenv_values(".env.staging", interpolate=False)
    if not all(values.get(key) for key in keys):
        raise SystemExit("Set the three Hopsworks settings in the ignored .env.staging file.")
    settings = {key: values[key].strip() for key in keys}
    stage_host = hostname(settings["HOPSWORKS_HOST"])
    stage_project = settings["HOPSWORKS_PROJECT"]
    with httpx.Client(timeout=15) as http:
        response = http.get(
            f"https://{stage_host}/hopsworks-api/api/project/getProjectInfo/"
            f"{quote(stage_project, safe='')}",
            headers={"Authorization": f"ApiKey {settings['HOPSWORKS_API_KEY']}"},
        )
        print(f"Staging project lookup HTTP {response.status_code}", flush=True)
        response.raise_for_status()
        if int(response.json()["projectId"]) != expected_id:
            raise SystemExit("The staging project ID did not match the expected ID.")
    return settings


class StagingProject:
    def __init__(self, store, suffix):
        self.store = store
        self.names = {FORECAST: f"mosamiat_{suffix}_forecast", BULLETIN: f"mosamiat_{suffix}_bulletin"}

    def get_feature_store(self):
        return self

    def get_or_create_feature_group(self, *, name, **options):
        return self.store.get_or_create_feature_group(name=self.names[name], **options)


def verify(project, root):
    suffix = uuid4().hex[:8]
    city = replace(CITY, city_id=f"islamabad_staging_{suffix}")
    target = StagingProject(project.get_feature_store(), suffix)
    print(f"Staging groups {target.names}", flush=True)
    worker = root / "worker"
    worker.mkdir()
    ledger = ScheduledLedger(project.get_dataset_api(), city, worker)
    app = build_graph(city, ledger, lambda: target, checkpoint_forecast=ledger.save)

    def prepare(hour, **changes):
        item = payload(f"2026-11-24T{hour}:00Z", **changes)
        for i, row in enumerate(item["forecasts"]):
            row["horizon_hours"] = i % 24 + 1
        item.update(
            run_id=f"{city.city_id}_{hour}", city_id=city.city_id,
            retrieved_at_utc=item["issued_at_utc"], evaluation_sha256=MANIFEST,
        )
        # Seed a prepared synthetic forecast without adding an input snapshot or invoking a model.
        publication.Ledger.prepare(ledger, item)
        ledger.save()
        return item["run_id"]

    def invoke(run_id, retry=False):
        return app.invoke({"run_id": run_id, "retry": retry})

    job_run = Job.run
    wait = publication._wait_for_materialization

    def seed(job, *args, **kwargs):
        execution = job_run(job, *args, **kwargs)
        assert execution is not None and wait(job, job.name, execution.id)
        print(f"Fixture seed execution {execution.id} for {job.name}", flush=True)
        return execution

    with (
        patch("mosamiat.agents.graph.run_drift", return_value={"drift_count": 0}),
        patch.object(Job, "run", seed),
    ):
        baseline = prepare("07")
        assert invoke(baseline)["bulletin_status"] == "verified"

    accepted = []

    def lose_reply(job, *args, **kwargs):
        execution = job_run(job, *args, **kwargs)
        if job.name.startswith(target.names[FORECAST]):
            assert execution is not None
            accepted.append((job, execution.id))
            print(f"Server accepted execution {execution.id}. Reply discarded by fixture.", flush=True)
            raise ConnectionError("Synthetic lost materialization response")
        return execution

    with patch("mosamiat.agents.graph.run_drift", return_value={"drift_count": 0}):
        ambiguous = prepare("08")
        with patch.object(Job, "run", lose_reply):
            for retry in (False, True):
                try:
                    state = invoke(ambiguous, retry=retry)
                except publication.MaterializationError as error:
                    assert "submission outcome unknown" in str(error)
                    print(f"Ambiguous attempt stopped safely ({error})", flush=True)
                else:
                    assert state["status"] == "verified"
        (job, accepted_id), = accepted
        assert ledger.materialization(ambiguous, FORECAST) == (None,)
        assert wait(job, job.name, accepted_id)
        before_replay = {item.id for item in job.get_executions()}
        assert invoke(ambiguous, retry=True)["status"] == "verified"
        assert {item.id for item in job.get_executions()} == before_replay
        print(f"Recovered execution {accepted_id} with 192 matching rows and zero relaunches.", flush=True)

        stopped = []
        compared = []

        def stop_fixture_bulletin(job, *args, **kwargs):
            execution = job_run(job, *args, **kwargs)
            if job.name.startswith(target.names[BULLETIN]):
                assert execution is not None
                stopped.append(execution.id)
                execution.stop()
            return execution

        def observe_newer_execution(job, name, execution_id, **kwargs):
            if name != FORECAST:
                return wait(job, name, execution_id, **kwargs)
            assert wait(job, name, execution_id, **kwargs)
            newer = job_run(job, args=job.config.get("defaultArgs", ""), await_termination=False)
            assert newer is not None and newer.id != execution_id
            assert job.get_executions()[0].id == newer.id
            assert wait(job, name, execution_id, **kwargs)
            newer_succeeded = wait(job, name, newer.id, **kwargs)
            compared.append((execution_id, newer.id, newer_succeeded))
            return True

        failed_bulletin = prepare("09", aqi_value=154)
        with (
            patch.object(Job, "run", stop_fixture_bulletin),
            patch.object(publication, "_wait_for_materialization", observe_newer_execution),
        ):
            state = invoke(failed_bulletin)
        (forecast_id, unrelated_id, unrelated_succeeded), = compared
        (bulletin_id,) = stopped
        assert state["status"] == "verified" and state["bulletin_status"] == "failed"
        assert ledger.materialization(failed_bulletin, FORECAST) == (forecast_id,)
        assert ledger.materialization(failed_bulletin, BULLETIN) == (bulletin_id,)
        ledger.save()
        cache = root / "dashboard"
        cache.mkdir()
        outlook = _cloud_outlook(city, cache)
        assert outlook.run_id == failed_bulletin and outlook.hours.shape == (24, 8)
        assert outlook.bulletin_status == "failed" and outlook.report
        for name, expected in ((FORECAST, 192), (BULLETIN, None)):
            group = target.store.get_feature_group(target.names[name], version=1)
            rows = group.select_all().filter(group.run_id == failed_bulletin).read(
                read_options={"arrow_flight_config": {"timeout": 90}},
            )
            print(f"{target.names[name]} readback returned {len(rows)} row(s)", flush=True)
            if expected is not None:
                assert len(rows) == expected
        print(
            f"Forecast execution {forecast_id} verified 192 rows with unrelated execution "
            f"{unrelated_id} present (success={unrelated_succeeded}). "
            f"Bulletin execution {bulletin_id} failed. "
            "Dashboard loaded the latest 24-hour forecast and previous report.",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=int, required=True)
    args = parser.parse_args()
    if version("hopsworks") != "4.8.8":
        raise SystemExit("This verification requires Hopsworks SDK 4.8.8.")
    try:
        with patch.dict(os.environ, staging_settings(args.project_id)), TemporaryDirectory() as tmp:
            project = connect_project()
            if int(project.id) != args.project_id:
                raise SystemExit("SDK connected to an unexpected project.")
            print(f"Confirmed isolated Hopsworks project {args.project_id}", flush=True)
            session = client.get_instance()._session
            session.send = partial(session.send, timeout=(60, 120))
            verify(project, Path(tmp))
    except Exception as error:
        print(f"Staging verification failed ({publication.safe_error_detail(error)})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
