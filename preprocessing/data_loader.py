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

    def download(
        self,
        tickers: list[str],
        start: str,
        end: str | None = None,
        interval: str = "1d",
        refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        datasets: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            cache_file = self.cache_dir / f"{ticker.replace('-', '_')}_{interval}.csv"
            if cache_file.exists() and not refresh:
                LOGGER.info("Loading cached data for %s", ticker)
                frame = pd.read_csv(cache_file, parse_dates=["Date"], index_col="Date")
            else:
                LOGGER.info("Downloading %s from yfinance", ticker)
                frame = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                )
                if frame.empty:
                    raise ValueError(f"No data returned for ticker {ticker}")
                frame = frame.rename_axis("Date")
                frame.to_csv(cache_file)
            datasets[ticker] = normalize_ohlcv_columns(frame)
        return datasets


def normalize_ohlcv_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance columns and keep the core OHLCV schema."""
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    rename = {column: column.lower().replace(" ", "_") for column in data.columns}
    data = data.rename(columns=rename)
    required = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return data[required].sort_index()
    