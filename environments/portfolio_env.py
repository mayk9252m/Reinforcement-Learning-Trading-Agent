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