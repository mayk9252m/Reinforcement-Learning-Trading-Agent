from __future__ import annotations

import argparse
import json

from agents.ppo_agent import PPOTradingAgent
from environments.trading_env import TradingEnv
from evaluation.backtester import Backtester
from preprocessing.pipeline import PreprocessingPipeline
from utils.config import load_config
from visualization.plots import create_plotly_dashboard


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest a trained PPO trading agent")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="models/ppo_trading_agent.zip")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--dashboard", default="logs/backtest_dashboard.html")
    return parser.parse_args()


