from __future__ import annotations

import argparse
import json

from agents.ppo_agent import PPOTradingAgent
from environments.trading_env import TradingEnv
from evaluation.backtester import Backtester
from preprocessing.pipeline import PreprocessingPipeline
from utils.config import load_config
from visualization.plots import create_plotly_dashboard