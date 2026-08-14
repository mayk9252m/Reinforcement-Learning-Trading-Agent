from __future__ import annotations

import numpy as np

from environments.trading_env import TradingEnv

def test_environment_reset_and_step(sample_market_data):
    env = TradingEnv(sample_market_data)
    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape[0] == env.observation_space.shape[0]

    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)
    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape[0] == env.observation_space.shape[0]
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)