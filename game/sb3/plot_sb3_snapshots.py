import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

GAME_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if GAME_DIR not in sys.path:
    sys.path.insert(0, GAME_DIR)

from sb3.sb3_env import TrucoSB3Env


@dataclass
class ModelEval:
    name: str
    path: str
    timesteps: int
    winrate: float
    wins: int
    losses: int
    ties: int


def _format_millions(value: float) -> str:
    millions = value / 1_000_000
    if abs(millions - round(millions)) < 1e-6:
        return f"{int(round(millions))}M"
    return f"{millions:.1f}M"


def _parse_timesteps(name: str) -> Optional[int]:
    match = re.search(r"(\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def _load_model(path: str):
    try:
        from sb3_contrib import MaskablePPO

        return MaskablePPO.load(path)
    except Exception:
        from stable_baselines3 import PPO

        return PPO.load(path)


def _list_models(history_dir: str, final_model: Optional[str]) -> list[tuple[str, str, int]]:
    entries = []
    if os.path.isdir(history_dir):
        for fname in os.listdir(history_dir):
            if not fname.endswith(".zip"):
                continue
            timesteps = _parse_timesteps(fname)
            if timesteps is None:
                continue
            entries.append((fname, os.path.join(history_dir, fname), timesteps))
    entries.sort(key=lambda item: item[2])

    if final_model and os.path.isfile(final_model):
        final_name = os.path.basename(final_model)
        final_ts = entries[-1][2] if entries else 0
        entries.append((final_name, final_model, final_ts))

    return entries


def _match_result(env: TrucoSB3Env) -> str:
    estado = env._env.logic.estado
    if estado.puntos_jugador > estado.puntos_oponente:
        return "win"
    if estado.puntos_jugador < estado.puntos_oponente:
        return "loss"
    return "tie"


def _evaluate_model(path: str, opponent: str, games: int, seed: Optional[int]) -> tuple[float, int, int, int]:
    env = TrucoSB3Env(opponent=opponent)
    model = _load_model(path)
    wins = losses = ties = 0
    for i in range(games):
        game_seed = None if seed is None else seed + i
        obs, _ = env.reset(seed=game_seed)
        done = False
        while not done:
            action_mask = env.action_masks()
            try:
                action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
            except TypeError:
                action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        result = _match_result(env)
        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1
        else:
            ties += 1
    winrate = wins / games if games > 0 else 0.0
    return winrate, wins, losses, ties


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalua snapshots/politicas SB3 y grafica winrate."
    )
    parser.add_argument(
        "--opponent",
        choices=["random", "rational"],
        default="rational",
        help="Agente rival.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=200,
        help="Partidas por snapshot.",
    )
    parser.add_argument(
        "--history-dir",
        default=os.path.join(GAME_DIR, "sb3", "models", "history"),
        help="Directorio con snapshots.",
    )
    parser.add_argument(
        "--final-model",
        default=os.path.join(GAME_DIR, "sb3", "models", "ppo_truco_league.zip"),
        help="Ruta de la politica final.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed para evaluaciones reproducibles.",
    )
    parser.add_argument(
        "--csv-name",
        default=None,
        help="Nombre del CSV de salida (solo el nombre de archivo).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Nombre del grafico de salida (solo el nombre de archivo).",
    )
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(GAME_DIR, ".."))
    resultados_dir = os.path.join(project_root, "resultados")
    plots_dir = os.path.join(GAME_DIR, "plots")
    os.makedirs(resultados_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    models = _list_models(args.history_dir, args.final_model)
    if not models:
        raise FileNotFoundError("No se encontraron snapshots o politicas SB3.")

    evaluations: list[ModelEval] = []
    for name, path, timesteps in models:
        winrate, wins, losses, ties = _evaluate_model(path, args.opponent, args.games, args.seed)
        evaluations.append(
            ModelEval(
                name=name,
                path=path,
                timesteps=timesteps,
                winrate=winrate,
                wins=wins,
                losses=losses,
                ties=ties,
            )
        )
        print(f"{name} | winrate={winrate:.3f} | W={wins} L={losses} T={ties}")

    csv_name = args.csv_name or f"sb3_league_eval_vs_{args.opponent}.csv"
    csv_path = os.path.join(resultados_dir, os.path.basename(csv_name))
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "timesteps", "winrate", "wins", "losses", "ties"],
        )
        writer.writeheader()
        for e in evaluations:
            writer.writerow(
                {
                    "model": e.name,
                    "timesteps": e.timesteps,
                    "winrate": f"{e.winrate:.6f}",
                    "wins": e.wins,
                    "losses": e.losses,
                    "ties": e.ties,
                }
            )

    x_values = [e.timesteps for e in evaluations]
    y_values = [e.winrate for e in evaluations]

    output_name = args.output or f"sb3_league_eval_vs_{args.opponent}.png"
    output_path = os.path.join(plots_dir, os.path.basename(output_name))
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_values, marker="o", linewidth=2, color="#2C7FB8")
    plt.title(f"SB3 League vs {args.opponent}")
    plt.xlabel("Timesteps (millones)")
    plt.ylabel("Winrate")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    if x_values:
        min_x = min(x_values)
        max_x = max(x_values)
        if min_x == max_x:
            tick_values = [min_x]
        else:
            tick_count = 10 if len(x_values) > 10 else len(x_values)
            step = (max_x - min_x) / (tick_count - 1)
            tick_values = [min_x + step * i for i in range(tick_count)]
        tick_labels = [_format_millions(val) for val in tick_values]
        plt.xticks(tick_values, tick_labels, rotation=0, ha="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"CSV guardado en {csv_path}")
    print(f"Grafico guardado en {output_path}")


if __name__ == "__main__":
    main()
