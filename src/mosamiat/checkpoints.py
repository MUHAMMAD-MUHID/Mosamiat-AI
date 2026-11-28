"""Shared conventions for published forecast ledger checkpoints."""

import re

LEDGER_NAME = re.compile(r"ledger-(\d{10})-([0-9a-f]{64})\.zip")

# Historical completed runs predate the explicit bulletin status.
VERIFIED_BULLETIN = """(
    EXISTS (SELECT 1 FROM events e WHERE e.run_id=b.run_id AND e.status='bulletin_verified')
    OR (
        EXISTS (SELECT 1 FROM events e WHERE e.run_id=b.run_id AND e.status='completed')
        AND NOT EXISTS (
            SELECT 1 FROM events e WHERE e.run_id=b.run_id AND e.status='bulletin_failed'
        )
    )
)"""
