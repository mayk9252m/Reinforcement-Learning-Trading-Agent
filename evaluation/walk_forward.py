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


