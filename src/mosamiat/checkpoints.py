"""Filename convention for a published, checksum-addressed forecast ledger."""

import re

LEDGER_NAME = re.compile(r"ledger-(\d{10})-([0-9a-f]{64})\.zip")
