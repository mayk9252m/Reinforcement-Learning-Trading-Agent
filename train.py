from __future__ import annotations

import argparse
import logging

import wandb

from agents.ppo_agent import PPOTradingAgent
from environments.trading_env import TradingEnv
from preprocessing.pipeline import PreprocessingPipeline
from utils.config import load_config
from utils.logging import configure_logging
from utils.seed import set_global_seed

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO trading agent")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--resume-from", default=None)
    return parser.parse_args()