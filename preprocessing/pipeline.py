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