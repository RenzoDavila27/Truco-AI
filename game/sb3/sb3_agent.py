import os
import sys
from typing import Optional

import numpy as np

GAME_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if GAME_DIR not in sys.path:
    sys.path.insert(0, GAME_DIR)

from truco_env import TrucoEnv


class SB3Agent:
    """Load a trained SB3 model and act using action masks."""

    def __init__(self, model_path: str, use_maskable: bool = True):
        self.model_path = model_path
        self.use_maskable = use_maskable
        self.model = self._load_model()

    def _load_model(self):
        if self.use_maskable:
            try:
                from sb3_contrib import MaskablePPO

                return MaskablePPO.load(self.model_path)
            except Exception:
                pass
        from stable_baselines3 import PPO

        return PPO.load(self.model_path)

    def choose_action(self, action_mask, env: Optional[TrucoEnv] = None, player_id: int = 0):
        if env is None:
            raise ValueError("env is required to build the observation")
        obs = env.get_observation(player_id)
        if self.use_maskable:
            action, _ = self.model.predict(obs, action_masks=np.array(action_mask, dtype=bool), deterministic=True)
        else:
            action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
