from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    train_start: int
    train_end: int
    test_end: int


def make_walk_forward_splits(
    data: pd.DataFrame,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> list[WalkForwardSplit]:
    """Create expanding walk-forward train/test splits."""
    splits: list[WalkForwardSplit] = []
    step = step_size or test_size
    start = 0
    while start + train_size + test_size <= len(data):
        train_end = start + train_size
        test_end = train_end + test_size
        splits.append(
            WalkForwardSplit(
                train=data.iloc[start:train_end].copy(),
                test=data.iloc[train_end:test_end].copy(),
                train_start=start,
                train_end=train_end,
                test_end=test_end,
            )
        )
        start += step
    return splits
