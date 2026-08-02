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