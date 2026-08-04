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

    def run(self, policy: Policy | None = None) -> BacktestResult:
        observation, info = self.env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            if policy is None:
                action = 0
            else:
                action, _state = policy.predict(observation, deterministic=True)
                action = int(np.asarray(action).item())
            observation, _reward, terminated, truncated, info = self.env.step(action)

        metrics = performance_report(info["equity_curve"], info["trades"])
        return BacktestResult(
            equity_curve=info["equity_curve"],
            actions=info["actions"],
            trades=info["trades"],
            metrics=metrics,
        )
