import os
import sys
from typing import Callable, Optional

import gymnasium as gym
import numpy as np

GAME_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if GAME_DIR not in sys.path:
    sys.path.insert(0, GAME_DIR)

from truco_env import TrucoEnv
from agents.random_agent import RandomAgent
from agents.rational_agent import RationalAgent


class TrucoSB3Env(gym.Env):
    """Single-agent wrapper for SB3. Controls player 0 and auto-plays opponent."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(self, opponent: str | object = "random"):
        super().__init__()
        self._env = TrucoEnv()
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self._opponent = self._make_opponent(opponent)

    def _make_opponent(self, opponent: str):
        if hasattr(opponent, "choose_action"):
            return opponent
        if opponent == "random":
            return RandomAgent()
        if opponent == "rational":
            return RationalAgent()
        raise ValueError(f"Unknown opponent: {opponent}")

    def set_opponent(self, opponent: str | object):
        self._opponent = self._make_opponent(opponent)

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        obs, info = self._env.reset(seed=seed)
        obs, _, done, info = self._auto_play_until_player(obs, info)
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._env.step(action, player_id=0)
        if terminated or truncated:
            return obs, reward, terminated, truncated, info
        obs, opp_reward, done, info = self._auto_play_until_player(obs, info)
        terminated = terminated or done
        return obs, reward + opp_reward, terminated, truncated, info

    def _auto_play_until_player(self, obs, info):
        done = False
        total_reward = 0.0
        while not done and self._env.get_current_player() != 0:
            player_id = self._env.get_current_player()
            mask = self._env.get_action_mask(player_id)
            if not any(mask):
                done = True
                break
            action = self._opponent.choose_action(mask, self._env, player_id)
            obs, reward, terminated, truncated, _ = self._env.step(action, player_id)
            total_reward += reward
            if terminated or truncated:
                done = True
                break
        return self._env.get_observation(0), total_reward, done, info

    def render(self, mode="human"):
        return self._env.render(mode=mode, player_id=0)

    def close(self):
        self._env.close()

    def action_masks(self):
        return np.array(self._env.get_action_mask(0), dtype=bool)
