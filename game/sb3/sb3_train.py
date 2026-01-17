import argparse
import os
import sys
from collections import deque

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

GAME_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if GAME_DIR not in sys.path:
    sys.path.insert(0, GAME_DIR)

from sb3.sb3_env import TrucoSB3Env


def make_env(opponent):
    initial_opponent = "random" if opponent == "selfplay" else opponent
    env = TrucoSB3Env(opponent=initial_opponent)
    try:
        from stable_baselines3.common.monitor import Monitor

        env = Monitor(env)
    except Exception:
        pass
    return env


def _resolve_model_path(path: str) -> str:
    if os.path.isfile(path):
        return path
    if os.path.isfile(f"{path}.zip"):
        return f"{path}.zip"
    return path


def _load_selfplay_opponent(snapshot_path: str):
    from agents.random_agent import RandomAgent
    from sb3.sb3_agent import SB3Agent

    resolved_path = _resolve_model_path(snapshot_path)
    if os.path.isfile(resolved_path):
        return SB3Agent(resolved_path)
    return RandomAgent()


def _get_base_env(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _get_match_result(env):
    base_env = _get_base_env(env)
    estado = base_env._env.logic.estado
    if estado.puntos_jugador > estado.puntos_oponente:
        return 1.0
    if estado.puntos_jugador < estado.puntos_oponente:
        return 0.0
    return 0.5


class SelfPlayWinrateCallback(BaseCallback):
    def __init__(self, env, snapshot_path: str, window_size: int, target_winrate: float, verbose: int = 0):
        super().__init__(verbose)
        self.env = env
        self.snapshot_path = snapshot_path
        self.window_size = window_size
        self.target_winrate = target_winrate
        self.results = deque(maxlen=window_size)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is None or not any(dones):
            return True
        result = _get_match_result(self.env)
        self.results.append(result)
        if len(self.results) < self.window_size:
            return True
        winrate = float(np.mean(self.results))
        if winrate >= self.target_winrate:
            os.makedirs(os.path.dirname(self.snapshot_path) or ".", exist_ok=True)
            self.model.save(self.snapshot_path)
            base_env = _get_base_env(self.env)
            base_env.set_opponent(_load_selfplay_opponent(self.snapshot_path))
            print(
                "Self-play: dificultad aumentada "
                f"(winrate={winrate:.2f}, ventana={self.window_size})."
            )
            self.results.clear()
        return True

def train(
    total_timesteps: int,
    opponent: str,
    output_path: str,
    seed: int | None,
    fresh: bool,
    selfplay_snapshot: str,
    selfplay_window: int,
    selfplay_winrate: float,
    learning_rate: float | None,
    force_learning_rate: bool,
):
    env = make_env(opponent)
    from sb3_contrib import MaskablePPO

    load_path = _resolve_model_path(output_path)
    if not fresh and os.path.isfile(load_path):
        model = MaskablePPO.load(load_path, env=env)
        if force_learning_rate and learning_rate is not None:
            model.learning_rate = learning_rate
            model.lr_schedule = lambda _: learning_rate
            if hasattr(model, "_setup_lr_schedule"):
                model._setup_lr_schedule()
    else:
        if learning_rate is None:
            learning_rate = 3e-4
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            learning_rate=learning_rate,
        )

    if opponent != "selfplay":
        model.learn(total_timesteps=total_timesteps)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        model.save(output_path)
        return

    base_env = _get_base_env(env)
    opponent_agent = _load_selfplay_opponent(selfplay_snapshot)
    base_env.set_opponent(opponent_agent)

    callback = SelfPlayWinrateCallback(
        env=env,
        snapshot_path=selfplay_snapshot,
        window_size=selfplay_window,
        target_winrate=selfplay_winrate,
    )
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, callback=callback)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO with action masking on Truco.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--opponent", choices=["random", "rational", "selfplay"], default="random")
    parser.add_argument("--output", type=str, default="game/sb3/models/ppo_truco")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--force-learning-rate",
        action="store_true",
        help="Sobrescribe el LR incluso al cargar un modelo existente.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Entrena desde cero aunque exista un modelo guardado.",
    )
    parser.add_argument(
        "--selfplay-window",
        type=int,
        default=100,
        help="Cantidad de partidas para medir winrate en self-play.",
    )
    parser.add_argument(
        "--selfplay-winrate",
        type=float,
        default=0.8,
        help="Winrate objetivo para actualizar el oponente en self-play.",
    )
    parser.add_argument(
        "--selfplay-snapshot",
        type=str,
        default="game/sb3/models/ppo_truco_opponent",
        help="Ruta del snapshot del oponente en self-play.",
    )
    args = parser.parse_args()

    train(
        args.timesteps,
        args.opponent,
        args.output,
        args.seed,
        args.fresh,
        args.selfplay_snapshot,
        args.selfplay_window,
        args.selfplay_winrate,
        args.learning_rate,
        args.force_learning_rate,
    )
