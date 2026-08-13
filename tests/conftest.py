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
            "low": close * (1 - rng.normal(0, 0.001, size=n)),
            "close": close,
        }
    )