from __future__ import annotations

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "open": [100, 101, 102],
        "high": [105, 106, 107],
        "low": [95, 96, 97],
        "close": [102, 103, 104],
        "volume": [1000, 1100, 1200]
    })