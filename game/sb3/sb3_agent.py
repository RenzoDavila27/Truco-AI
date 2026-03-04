import os
import sys
from typing import Optional

import numpy as np
import torch

GAME_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if GAME_DIR not in sys.path:
    sys.path.insert(0, GAME_DIR)

from constantes import Acciones
from truco_env import TrucoEnv


class SB3Agent:
    """Load a trained SB3 model and act using action masks."""

    def __init__(self, model_path: str, use_maskable: bool = True, verbose: bool = False):
        self.model_path = model_path
        self.use_maskable = use_maskable
        self.verbose = verbose
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

    def _get_action_probs(self, obs, action_mask):
        """Extrae las probabilidades de cada acción desde la política."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(action_mask, dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_t, action_masks=mask_t)
            probs = dist.distribution.probs.squeeze(0).cpu().numpy()
        return probs

    def _print_probs(self, probs, action_mask):
        """Muestra las probabilidades de las acciones válidas."""
        print("  ┌─ Probabilidades del agente ─────────────┐")
        for i, (valid, prob) in enumerate(zip(action_mask, probs)):
            if valid:
                name = Acciones(i).name
                bar_len = int(prob * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"  │ {name:<16s} {bar} {prob:6.1%} │")
        print("  └─────────────────────────────────────────┘")

    def choose_action(self, action_mask, env: Optional[TrucoEnv] = None, player_id: int = 0):
        if env is None:
            raise ValueError("env is required to build the observation")
        obs = env.get_observation(player_id)
        mask_np = np.array(action_mask, dtype=bool)

        if self.verbose and self.use_maskable:
            probs = self._get_action_probs(obs, mask_np)
            self._print_probs(probs, mask_np)

        if self.use_maskable:
            action, _ = self.model.predict(obs, action_masks=mask_np, deterministic=True)
        else:
            action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
