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