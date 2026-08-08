from __future__ import annotations

import pandas as pd
import numpy as np


def add_technical_indicators(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create a rich feature set using pandas-native technical indicators."""
    data = frame.copy()
    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]

    for window in config.get("sma_windows", [10, 20, 50]):
        data[f"sma_{window}"] = close.rolling(window).mean()
    for window in config.get("ema_windows", [12, 26]):
        data[f"ema_{window}"] = close.ewm(span=window, adjust=False).mean()

    rsi_window = int(config.get("rsi_window", 14))
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_window).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_window).mean()
    rs = gain / loss.replace(0, np.nan)
    data["rsi"] = 100 - (100 / (1 + rs))

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema_fast - ema_slow
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    atr_window = int(config.get("atr_window", 14))
    true_range = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    data["atr"] = true_range.rolling(atr_window).mean()

    bb_window = int(config.get("bollinger_window", 20))
    bb_mid = close.rolling(bb_window).mean()
    bb_std = close.rolling(bb_window).std()
    data["bb_mid"] = bb_mid
    data["bb_upper"] = bb_mid + 2 * bb_std
    data["bb_lower"] = bb_mid - 2 * bb_std
    data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / bb_mid

    momentum_window = int(config.get("momentum_window", 10))
    volatility_window = int(config.get("volatility_window", 20))
    data["log_return"] = np.log(close / close.shift(1))
    data["momentum"] = close.pct_change(momentum_window)
    data["volatility"] = data["log_return"].rolling(volatility_window).std() * np.sqrt(252)
    data["volume_change"] = volume.pct_change()
    data["drawdown"] = close / close.cummax() - 1.0

    return data.replace([np.inf, -np.inf], np.nan)


def clean_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values after indicator construction."""
    return frame.ffill().bfill().dropna().copy()
