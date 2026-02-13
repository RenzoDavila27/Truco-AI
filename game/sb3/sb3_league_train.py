import argparse
import os
import random
import sys

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

GAME_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if GAME_DIR not in sys.path:
    sys.path.insert(0, GAME_DIR)

from sb3.sb3_env import TrucoSB3Env
from sb3.sb3_league import LeagueEnvWrapper, find_league_env


class LeagueCallback(BaseCallback):
    def __init__(self, check_freq: int, history_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.history_dir = history_dir
        os.makedirs(self.history_dir, exist_ok=True)

    def _on_training_start(self) -> None:
        league_env = find_league_env(self.training_env)
        if league_env is not None:
            league_env.set_current_model(self.model)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True
        path = os.path.join(self.history_dir, f"ppo_snapshot_{self.num_timesteps}.zip")
        self.model.save(path)
        league_env = find_league_env(self.training_env)
        if league_env is not None:
            league_env.add_snapshot(path)
        return True


def make_env():
    env = TrucoSB3Env(opponent="random")
    return LeagueEnvWrapper(env)


def train(
    total_timesteps: int,
    seed: int | None,
    check_freq: int,
    history_dir: str,
    ent_coef: float,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    output_path: str,
):
    env = make_env()
    from sb3_contrib import MaskablePPO

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        ent_coef=ent_coef,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    callback = LeagueCallback(check_freq=check_freq, history_dir=history_dir)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO with league opponent pool.")
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--check-freq", type=int, default=100_000)
    parser.add_argument("--history-dir", type=str, default="game/sb3/models/history")
    parser.add_argument("--output", type=str, default="game/sb3/models/ppo_truco_league")
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    train(
        args.timesteps,
        args.seed,
        args.check_freq,
        args.history_dir,
        args.ent_coef,
        args.n_steps,
        args.batch_size,
        args.learning_rate,
        args.output,
    )
