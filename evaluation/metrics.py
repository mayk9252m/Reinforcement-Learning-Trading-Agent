from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def returns_from_equity(equity_curve: list[float] | np.ndarray | pd.Series) -> pd.Series:
    equity = pd.Series(equity_curve, dtype="float64")
    return equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def total_return(equity_curve) -> float:
    equity = pd.Series(equity_curve, dtype="float64")
    return float(equity.iloc[-1] / equity.iloc[0] - 1)

