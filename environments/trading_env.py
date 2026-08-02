from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any
from torch import seed

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from rewards.risk_adjusted import RiskAdjustedReward


@dataclass
class Trade:
    step: int
    action: int
    price: float
    shares: float
    cash: float
    position: float
    portfolio_value: float
    transaction_cost: float


class TradingEnv(gym.Env):
    """Single-asset trading environment with realistic execution costs."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int = 30,
        initial_cash: float = 100000.0,
        commission_bps: float = 2.0,
        slippage_bps: float = 5.0,
        max_position_fraction: float = 1.0,
        reward_config: dict | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if len(data) <= window_size + 1:
            raise ValueError("Data length must exceed window_size + 1")

        self.data = data.reset_index(drop=True).copy()
        self.window_size = int(window_size)
        self.initial_cash = float(initial_cash)
        self.commission_bps = float(commission_bps)
        self.slippage_bps = float(slippage_bps)
        self.max_position_fraction = float(max_position_fraction)
        self.render_mode = render_mode
        self.reward_model = RiskAdjustedReward(reward_config)

        self.feature_columns = list(self.data.columns)
        self.close_index = self.feature_columns.index("close")
        self.action_space = spaces.Discrete(3)

        obs_size = self.window_size * len(self.feature_columns) + 6
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )
        self._returns_window: deque[float] = deque(maxlen=63)
        self.reset()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.position = 0.0
        self.avg_entry_price = 0.0
        self.previous_action = 0
        self.portfolio_value = self.initial_cash
        self.peak_value = self.initial_cash
        self.trades: list[Trade] = []
        self.equity_curve: list[float] = [self.initial_cash]
        self.actions: list[int] = [0]
        self._returns_window.clear()
        return self._get_observation(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        previous_value = self.portfolio_value
        price = self._current_price()
        transaction_cost, turnover = self._execute_trade(action, price)

        self.current_step += 1
        self.portfolio_value = self._mark_to_market(self._current_price())
        self.peak_value = max(self.peak_value, self.portfolio_value)
        daily_return = (self.portfolio_value - previous_value) / max(previous_value, 1e-12)
        drawdown = self.portfolio_value / max(self.peak_value, 1e-12) - 1.0
        self._returns_window.append(daily_return)

        reward = self.reward_model(
            daily_return=daily_return,
            returns_window=list(self._returns_window),
            drawdown=drawdown,
            turnover=turnover,
            transaction_cost=transaction_cost,
            portfolio_value=self.portfolio_value,
        )

        self.previous_action = action
        self.equity_curve.append(self.portfolio_value)
        self.actions.append(action)

        terminated = self.portfolio_value <= self.initial_cash * 0.2
        truncated = self.current_step >= len(self.data) - 1
        return self._get_observation(), reward, terminated, truncated, self._get_info()