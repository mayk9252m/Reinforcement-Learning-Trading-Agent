from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    return_weight: float = 1.0
    sharpe_weight: float = 0.25
    drawdown_penalty: float = 0.35
    volatility_penalty: float = 0.10
    overtrading_penalty: float = 0.02
    transaction_cost_penalty: float = 1.0


class RiskAdjustedReward:
    """Risk-aware reward shaping for trading agents."""

    def __init__(self, config: RewardConfig | dict | None = None) -> None:
        if isinstance(config, dict):
            self.config = RewardConfig(**config)
        elif isinstance(config, RewardConfig):
            self.config = config
        else:
            self.config = RewardConfig()

    def __call__(
        self,
        daily_return: float,
        returns_window: list[float],
        drawdown: float,
        turnover: float,
        transaction_cost: float,
        portfolio_value: float,
    ) -> float:
        returns = np.asarray(returns_window, dtype=np.float64)
        volatility = float(np.std(returns)) if returns.size > 1 else 0.0
        sharpe = 0.0
        if volatility > 1e-12:
            sharpe = float(np.mean(returns) / volatility * np.sqrt(252))

        cost_ratio = transaction_cost / max(portfolio_value, 1e-12)
        reward = (
            self.config.return_weight * daily_return
            + self.config.sharpe_weight * sharpe / 252
            - self.config.drawdown_penalty * abs(min(drawdown, 0.0))
            - self.config.volatility_penalty * volatility
            - self.config.overtrading_penalty * turnover
            - self.config.transaction_cost_penalty * cost_ratio
        )
        return float(np.clip(reward, -10.0, 10.0))
