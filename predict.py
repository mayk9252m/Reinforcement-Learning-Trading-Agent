from __future__ import annotations

import argparse

from agents.ppo_agent import PPOTradingAgent
from environments.trading_env import TradingEnv
from preprocessing.pipeline import PreprocessingPipeline
from utils.config import load_config

ACTION_LABELS = {0: "Hold", 1: "Buy", 2: "Sell"}