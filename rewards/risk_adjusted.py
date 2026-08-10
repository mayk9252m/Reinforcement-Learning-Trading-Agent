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