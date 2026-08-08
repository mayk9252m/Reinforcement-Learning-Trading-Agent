from __future__ import annotations

import pandas as pd
import numpy as np


def add_technical_indicators(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create a rich feature set using pandas-native technical indicators."""