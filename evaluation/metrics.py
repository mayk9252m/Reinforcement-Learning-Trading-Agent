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


def annual_return(equity_curve) -> float:
    equity = pd.Series(equity_curve, dtype="float64")
    years = max((len(equity) - 1) / TRADING_DAYS, 1 / TRADING_DAYS)
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def sharpe_ratio(returns, risk_free_rate: float = 0.0) -> float:
    series = pd.Series(returns, dtype="float64").dropna()
    if series.std() == 0 or series.empty:
        return 0.0
    excess = series - risk_free_rate / TRADING_DAYS
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / series.std())


def sortino_ratio(returns, risk_free_rate: float = 0.0) -> float:
    series = pd.Series(returns, dtype="float64").dropna()
    downside = series[series < 0]
    if downside.std() == 0 or downside.empty:
        return 0.0
    excess = series - risk_free_rate / TRADING_DAYS
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / downside.std())


def max_drawdown(equity_curve) -> float:
    equity = pd.Series(equity_curve, dtype="float64")
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min())


def calmar_ratio(equity_curve) -> float:
    mdd = abs(max_drawdown(equity_curve))
    return 0.0 if mdd == 0 else annual_return(equity_curve) / mdd