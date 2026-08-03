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

    def render(self) -> None:
        print(
            f"step={self.current_step} value={self.portfolio_value:.2f}"
            f"cash={self.cash:.2f} position={self.position:.4f} "
        )

    def close(self) -> None:
        pass

    def _execute_trade(self, action: int, price: float) -> tuple[float, float]:
        if action == 0:
            return 0.0, 0.0

        commission = self.commission_bps / 10000
        slippage = self.slippage_bps / 10000
        transaction_cost = 0.0
        turnover = 0.0

        if action == 1:
            execution_price = price * (1 + slippage)
            target_exposure = self.portfolio_value * self.max_position_fraction
            current_exposure = self.position * execution_price
            trade_value = max(target_exposure - current_exposure, 0.0)
            trade_value = min(trade_value, self.cash / (1 + commission))
            shares = trade_value / execution_price if execution_price > 0 else 0.0
            transaction_cost = trade_value * commission
            if shares > 0:
                self.cash -= trade_value + transaction_cost
                self.position += shares
                self.avg_entry_price = execution_price
                turnover = trade_value / max(self.portfolio_value, 1e-12)
                self._record_trade(action, execution_price, shares, transaction_cost)

        elif action == 2 and self.position > 0:
            execution_price = price * (1 - slippage)
            shares = self.position
            trade_value = shares * execution_price
            transaction_cost = trade_value * commission
            self.cash += trade_value - transaction_cost
            self.position = 0.0
            self.avg_entry_price = 0.0
            turnover = trade_value / max(self.portfolio_value, 1e-12)
            self._record_trade(action, execution_price, -shares, transaction_cost)

        return transaction_cost, turnover

    def _record_trade(
        self,
        action: int,
        price: float,
        shares: float,
        transaction_cost: float,
    ) -> None:
        self.trades.append(
            Trade(
                step=self.current_step,
                action=action,
                price=price,
                shares=shares,
                cash=self.cash,
                position=self.position,
                portfolio_value=self.portfolio_value,
                transaction_cost=transaction_cost,
            )
        )