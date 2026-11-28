"""Exercise a synthetic checkpoint handoff between disposable review workers."""

import argparse
from pathlib import Path

from test_scheduled import CITY, DatasetFiles

from mosamiat.agents.retraining import reserve_attempt
from mosamiat.scheduled import ScheduledLedger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("reserve", "restore"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    worker = args.directory / args.phase
    worker.mkdir(parents=True)
    ledger = ScheduledLedger(DatasetFiles(args.directory / "remote"), CITY, worker)
    request = "islamabad_20261127T070134Z"
    if args.phase == "reserve":
        assert reserve_attempt(ledger, request) == (True, None)
        print("Reserved attempt 1 and published a checksummed synthetic checkpoint.")
    else:
        assert ledger.versions, "Expected a checkpoint from the first worker"
        assert reserve_attempt(ledger, request) == (True, None)
        generation = ledger.versions[-1]
        assert reserve_attempt(ledger, request) == (False, None)
        assert ledger.versions[-1] == generation
        print("Restored attempt 1, reserved attempt 2, and declined attempt 3.")


if __name__ == "__main__":
    main()
