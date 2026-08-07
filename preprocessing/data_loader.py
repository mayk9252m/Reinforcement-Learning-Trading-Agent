from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger(__name__)


class MarketDataLoader:
    """Download and load market data from Yahoo Finance."""

    def __init__(self, cache_dir: str | Path = "data/raw") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)