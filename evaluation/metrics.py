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


def trade_statistics(trades: list) -> dict[str, float]:
    sells = [trade for trade in trades if getattr(trade, "action", None) == 2]
    if not sells:
        return {
            "win_rate": 0.0,
            "average_profit": 0.0,
            "average_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
        }
    pnl = np.asarray([getattr(trade, "portfolio_value", 0.0) for trade in sells], dtype=float)
    pnl = np.diff(np.insert(pnl, 0, pnl[0]))
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    return {
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) else 0.0,
        "average_profit": float(wins.mean()) if len(wins) else 0.0,
        "average_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else 0.0,
        "expectancy": float(pnl.mean()) if len(pnl) else 0.0,
    }


def performance_report(equity_curve, trades: list | None = None) -> dict[str, float]:
    returns = returns_from_equity(equity_curve)
    report = {
        "total_return": total_return(equity_curve),
        "annual_return": annual_return(equity_curve),
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "calmar_ratio": calmar_ratio(equity_curve),
        "max_drawdown": max_drawdown(equity_curve),
    }
    report.update(trade_statistics(trades or []))
    return report
