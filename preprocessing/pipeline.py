from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from preprocessing.data_loader import MarketDataLoader
from preprocessing.features import add_technical_indicators, clean_market_features


class PreprocessingPipeline:
    """End-to-end data download, feature engineering, and normalization."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.processed_dir = Path(config["data"].get("processed_dir", "data/processed"))
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def run(self, refresh: bool = False) -> dict[str, pd.DataFrame]:
        loader = MarketDataLoader(self.config["data"].get("cache_dir", "data/raw"))
        raw = loader.download(
            self.config["data"]["tickers"],
            start=self.config["data"]["start"],
            end=self.config["data"].get("end"),
            interval=self.config["data"].get("interval", "1d"),
            refresh=refresh,
        )
        processed: dict[str, pd.DataFrame] = {}
        for ticker, frame in raw.items():
            features = add_technical_indicators(frame, self.config["features"].get("indicators", {}))
            features = clean_market_features(features)
            normalized = normalize_features(features, self.config["features"].get("scaler", "standard"))
            normalized.to_csv(self.processed_dir / f"{ticker.replace('-', '_')}.csv")
            processed[ticker] = normalized
        return processed


def normalize_features(frame: pd.DataFrame, scaler_name: str = "standard") -> pd.DataFrame:
    """Normalize feature columns while preserving prices for execution."""
    data = frame.copy()
    execution_columns = ["open", "high", "low", "close", "volume"]
    feature_columns = [column for column in data.columns if column not in execution_columns]
    scaler = MinMaxScaler() if scaler_name == "minmax" else StandardScaler()
    if feature_columns:
        data[feature_columns] = scaler.fit_transform(data[feature_columns])
    return data
