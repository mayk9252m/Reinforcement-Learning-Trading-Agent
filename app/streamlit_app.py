from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.ppo_agent import PPOTradingAgent
from environments.trading_env import TradingEnv
from evaluation.backtester import Backtester
from preprocessing.pipeline import PreprocessingPipeline
from utils.config import load_config
from visualization.plots import create_plotly_dashboard


st.set_page_config(page_title="PPO Trading Agent", layout="wide")
st.title("PPO Reinforcement Learning Trading Agent")

config_path = st.sidebar.text_input("Config", "configs/default.yaml")
model_path = st.sidebar.text_input("Model", "models/ppo_trading_agent.zip")
ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
run = st.sidebar.button("Run Backtest")