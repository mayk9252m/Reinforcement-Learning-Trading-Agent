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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data = PreprocessingPipeline(config).run(refresh=False)[args.ticker]
    env_cfg = config["environment"]
    env = TradingEnv(
        data,
        window_size=config["features"]["window_size"],
        initial_cash=env_cfg["initial_cash"],
        commission=env_cfg["commission_bps"],
        slippage=env_cfg["slippage_bps"],
        max_position_fraction=env_cfg["max_position_fraction"],
        reward_config=env_cfg["reward"],
    )
    agent = PPOTradingAgent.load(args.model, env=env, config=config)
    observation, _info = env.reset()
    action, _state = agent.predict(observation)
    print(ACTION_LABELS[int(action)])


if __name__ == "__main__":
    main()