"""Partition issue times chronologically and purge labels at fold boundaries."""

import pandas as pd

from mosamiat.collect import CollectionError


def split_targets(
    targets: pd.DataFrame,
    validation_start: pd.Timestamp,
    test_start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Use half-open UTC folds. The end is one hour after the final reference hour."""
    issues = targets.reference_at_utc
    if not issues.min() < validation_start < test_start < end:
        raise CollectionError("Expected nonempty chronological train, validation and test windows")
    folds = {}
    for name, start, stop in (
        ("train", issues.min(), validation_start),
        ("validation", validation_start, test_start),
        ("test", test_start, end),
    ):
        folds[name] = targets.loc[
            (issues >= start) & (issues < stop) & (targets.target_at_utc < stop)
        ]
        if folds[name].empty:
            raise CollectionError(f"{name} has no issue/target pairs inside its window")
    validate_split_order(folds)
    return folds


def validate_split_order(folds: dict[str, pd.DataFrame]) -> None:
    """Also usable to audit externally supplied partitions before model fitting."""
    previous_end = None
    for name in ("train", "validation", "test"):
        fold = folds[name]
        if fold.empty or not (fold.target_at_utc > fold.reference_at_utc).all():
            raise CollectionError(f"Invalid forecast timestamps in {name}")
        if previous_end is not None and fold.reference_at_utc.min() <= previous_end:
            raise CollectionError(f"Earlier labels overlap {name} issue times")
        previous_end = fold.target_at_utc.max()
