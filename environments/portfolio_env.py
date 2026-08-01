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