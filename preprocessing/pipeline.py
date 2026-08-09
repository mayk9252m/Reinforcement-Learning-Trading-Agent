from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from preprocessing.data_loader import MarketDataLoader
from preprocessing.features import add_technical_indicators, clean_market_features