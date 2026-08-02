from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

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