from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from environments.trading_env import TradingEnv
from evaluation.metrics import performance_report


class Policy(Protocol):
    def predict(self, observation, deterministic: bool = True): ...


@dataclass
class BacktestResult:
    equity_curve: list[float]
    actions: list[int]
    trades: list
    metrics: dict[str, float]


class Backtester:
    """Run deterministic policy evaluation over a trading environment."""

    def __init__(self, env: TradingEnv) -> None:
        self.env = env