from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class ContinuousPortfolioEnv(gym.Env):
    """Multi-asset allocation environment with continuous weights.

    The action is a vector of asset allocations. Softmax normalization keeps the
    portfolio long-only and fully invested; this class is a foundation for the
    requested multi-stock optimization bonus.
    """

    def __init__(self, prices: pd.DataFrame, window_size: int = 30, initial_cash: float = 10000.0):
        super().__init__()
        self.prices = prices.ffill().dropna().copy()
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.n_assets = self.prices.shape[1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size * self.n_assets + self.n_assets,),
            dtype=np.float32,
        )
        self.reset()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_idx = self.window_size
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.value = self.initial_cash
        self.equity_curve = [self.value]
        return self.obs(), {"portfolio_value": self.value}

    def step(self, action):
        exp_action = np.exp(action - np.max(action))
        new_weights = exp_action / exp_action.sum()
        prev_prices = self.prices.iloc[self.step_idx - 1].to_numpy()
        next_prices = self.prices.iloc[self.step_idx].to_numpy()
        asset_returns = next_prices / prev_prices - 1
        turnover = np.abs(new_weights - self.weights).sum()
        portfolio_return = float(np.dot(new_weights, asset_returns) - 0.001 * turnover)
        self.value *= 1 + portfolio_return
        self.weights = new_weights
        self.step_idx += 1
        self.equity_curve.append(self.value)
        done = self.step_idx >= len(self.prices) - 1
        return self._obs(), portfolio_return, False, done, {"portfolio_value": self.value}

    