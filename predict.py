from __future__ import annotations

import argparse

from yaml import parser

from agents.ppo_agent import PPOTradingAgent
from environments.trading_env import TradingEnv
from preprocessing.pipeline import PreprocessingPipeline
from utils.config import load_config

ACTION_LABELS = {0: "Hold", 1: "Buy", 2: "Sell"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-step PPO inference")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="models/ppo_trading_agent.zip")
    parser.add_argument("--ticker", default="AAPL")
    return parser.parse_args()