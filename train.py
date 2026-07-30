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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config.get("logging", {}).get("level", "INFO"))
    set_global_seed(config["project"].get("seed", 42))

    if config.get("logging", {}).get("use_wandb", False):
        wandb.init(project=config["logging"].get("wandb_project", "ppo_trading-agent"), config=config)

    datasets = PreprocessingPipeline(config).run(refresh=args.refresh_data)
    ticker = args.ticker or config["data"]["tickers"][0]
    data = datasets[ticker]
    split = int(len(data) * 0.8)
    train_data = data.iloc[:split].copy()
    eval_data = data.iloc[split:].copy()

    env_cfg = config["environment"]
    train_env = TradingEnv(
        train_data,
        window_size=config["features"]["window_size"],
        initial_cash=env_cfg["initial_cash"],
        commission_bps=env_cfg["commission_bps"],
        slippage_bps=env_cfg["slippage_bps"],
        max_position_fraction=env_cfg["max_position_fraction"],
        reward_config=env_cfg["reward"],
    )
    eval_env = TradingEnv(
        eval_data,
        window_size=config["features"]["window_size"],
        initial_cash=env_cfg["initial_cash"],
        commission_bps=env_cfg["commission_bps"],
        slippage_bps=env_cfg["slippage_bps"],
        max_position_fraction=env_cfg["max_position_fraction"],
        reward_config=env_cfg["reward"],
    )

    LOGGER.info("Training PPO on %s for %s timesteps", ticker, config["ppo"]["total_timesteps"])
    agent = PPOTradingAgent(train_env, config=config, seed=config["project"].get("seed", 42))
    agent.train(eval_env=eval_env, resume_from=args.resume_from)
    agent.save("models/ppo_trading_agent.zip")
    LOGGER.info("Saved model to models/ppo_trading_agent.zip")


if __name__ == "__main__":
    main()
