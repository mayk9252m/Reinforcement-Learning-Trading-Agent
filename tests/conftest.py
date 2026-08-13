from __future__ import annotations

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_data() -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    n = 120
    returns = rng.normal(0.0008, 0.01, size=n)
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, size=n)),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 3_000_000, size=n),
            "sma_10":close,
            "ema_12":close,
            "rsi": 50.0,
            "macd": 0.0,
            "atr": 1.0,
            "bb_width": 0.05,
            "momentum": returns,
            "log_return": returns,
            "volatility": 0.15,
            "drawdown": 0.0,
        }
    )