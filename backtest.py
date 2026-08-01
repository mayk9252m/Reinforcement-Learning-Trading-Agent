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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data = PreprocessingPipeline(config).run(refresh=False)[args.ticker]
    env_cfg = config["environment"]
    env = TradingEnv(
        data,
        window_size=config["features"]["window_size"],
        initial_cash=env_cfg["initial_cash"],
        commission_bps=env_cfg["commission_bps"],
        slippage_bps=env_cfg["slippage_bps"],
        max_position_fraction=env_cfg["max_position_fraction"],
        reward_config=env_cfg["reward_config"],
    )
    agent = PPOTradingAgent.load(args.model, env=env, config=config, seed=config["project"].get("seed", 42))
    result = Backtester(env).run(policy=agent)
    create_plotly_dashboard(data["close"], result.equity_curve, result.actions, result.metrics, args.dashboard)
    print(json.dumps(result.metrics, indent=2))
    print(f"Dashboard written to {args.dashboard}")


if __name__ == "__main__":
    main()
