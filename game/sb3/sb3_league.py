import os
import random
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np

from agents.random_agent import RandomAgent
from agents.rational_agent import RationalAgent
from sb3.sb3_agent import SB3Agent
from sb3.sb3_env import TrucoSB3Env


class SelfPlayAgent:
    def __init__(self, model=None):
        self.model = model

    def set_model(self, model):
        self.model = model

    def choose_action(self, action_mask, env=None, player_id=0):
        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        if not valid_actions:
            return None
        if self.model is None or env is None:
            return valid_actions[0]
        obs = env.get_observation(player_id)
        try:
            action, _ = self.model.predict(
                obs,
                action_masks=np.array(action_mask, dtype=bool),
                deterministic=True,
            )
        except TypeError:
            action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)
        if action_mask[action]:
            return action
        return valid_actions[0]


class SnapshotAgent:
    def __init__(self, model_path: str):
        self._agent = SB3Agent(model_path)

    def choose_action(self, action_mask, env=None, player_id=0):
        return self._agent.choose_action(action_mask, env, player_id)


@dataclass
class LeaguePool:
    self_play_weight: float = 0.5
    heuristic_weight: float = 0
    snapshot_weight: float = 0.5
    self_play_agent: SelfPlayAgent = field(default_factory=SelfPlayAgent)
    heuristics: list = field(default_factory=lambda: [RandomAgent(), RationalAgent()])
    snapshots: list[SnapshotAgent] = field(default_factory=list)

    def sample(self):
        r = random.random()
        if r < self.self_play_weight and self.self_play_agent.model is not None:
            return self.self_play_agent
        if r < self.self_play_weight + self.heuristic_weight:
            return random.choice(self.heuristics)
        if self.snapshots:
            return random.choice(self.snapshots)
        if self.heuristic_weight > 0:
            return random.choice(self.heuristics)
        else:
            return self.self_play_agent

    def add_snapshot(self, path: str):
        self.snapshots.append(SnapshotAgent(path))


class LeagueEnvWrapper(gym.Wrapper):
    def __init__(self, env: TrucoSB3Env, pool: Optional[LeaguePool] = None):
        super().__init__(env)
        self.pool = pool or LeaguePool()

    def set_current_model(self, model):
        self.pool.self_play_agent.set_model(model)

    def add_snapshot(self, path: str):
        self.pool.add_snapshot(path)

    def reset(self, **kwargs):
        opponent = self.pool.sample()
        if hasattr(self.env, "set_opponent"):
            self.env.set_opponent(opponent)
        else:
            self.env.opponent_agent = opponent
        return self.env.reset(**kwargs)


def find_league_env(env):
    if hasattr(env, "envs") and env.envs:
        current = env.envs[0]
    else:
        current = env
    while current is not None:
        if isinstance(current, LeagueEnvWrapper):
            return current
        if hasattr(current, "env"):
            current = current.env
        else:
            return None
    return None
