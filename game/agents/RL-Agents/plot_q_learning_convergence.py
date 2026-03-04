import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional
import random
import numpy as np

import matplotlib.pyplot as plt
import csv

GAME_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if GAME_DIR not in sys.path:
    sys.path.insert(0, GAME_DIR)

from truco_env import TrucoEnv
from agents.registry import create_agent, get_agent_registry
from constantes import Acciones


@dataclass
class TableEval:
    name: str
    path: str
    episode: Optional[int]
    winrate: float
    wins: int
    losses: int
    ties: int


def _parse_episode(name: str) -> Optional[int]:
    match = re.search(r"_ep(\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def _format_millions(value: float) -> str:
    millions = value / 1_000_000
    if abs(millions - round(millions)) < 1e-6:
        return f"{int(round(millions))}M"
    return f"{millions:.1f}M"


def _list_q_tables(q_tables_dir: str) -> list[tuple[str, str, Optional[int], float]]:
    entries: list[tuple[str, str, Optional[int], float]] = []
    if not os.path.isdir(q_tables_dir):
        return entries
    for fname in os.listdir(q_tables_dir):
        if not fname.endswith(".pkl"):
            continue
        path = os.path.join(q_tables_dir, fname)
        episode = _parse_episode(fname)
        mtime = os.path.getmtime(path)
        entries.append((fname, path, episode, mtime))
    entries.sort(key=lambda item: (0 if item[2] is not None else 1, item[2] or 0, item[3]))
    return entries


def _play_game(env: TrucoEnv, agent_0, agent_1, seed=None) -> str:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    env.reset(seed=seed)
    done = False

    while not done:
        player_id = env.get_current_player()
        action_mask = env.get_action_mask(player_id)
        if not any(action_mask):
            break

        if player_id == 0:
            chosen_action = agent_0.choose_action(action_mask, env, player_id)
        else:
            chosen_action = agent_1.choose_action(action_mask, env, player_id)

        estado = env.logic.estado
        prev_cartas = list(estado.cartas_jugadas)
        prev_turno_responder_truco = estado.turno_responder_truco

        _, _, done, _, _ = env.step(chosen_action, player_id)

        if not done:
            mano_terminada = False
            if prev_cartas and not env.logic.estado.cartas_jugadas:
                mano_terminada = True
            elif chosen_action == Acciones.IR_AL_MAZO.value:
                mano_terminada = True
            elif chosen_action == Acciones.NO_QUIERO.value and prev_turno_responder_truco:
                mano_terminada = True

            if mano_terminada:
                continue

    points_j0 = env.logic.estado.puntos_jugador
    points_j1 = env.logic.estado.puntos_oponente
    if points_j0 > points_j1:
        return "J0"
    if points_j1 > points_j0:
        return "J1"
    return "Empate"


def _evaluate_table(
    q_table_path: str,
    opponent_name: str,
    games: int,
    q_player: int,
    seed: Optional[int],
) -> tuple[float, int, int, int]:
    env = TrucoEnv()
    opponent = create_agent(opponent_name)
    q_agent = create_agent("q_learning", q_table_path=q_table_path)
    if q_player == 0:
        agent_0, agent_1 = q_agent, opponent
    else:
        agent_0, agent_1 = opponent, q_agent

    wins = losses = ties = 0
    for i in range(games):
        game_seed = None if seed is None else seed + i
        winner = _play_game(env, agent_0, agent_1, seed=game_seed)
        if winner == "Empate":
            ties += 1
        elif (winner == "J0" and q_player == 0) or (winner == "J1" and q_player == 1):
            wins += 1
        else:
            losses += 1
    winrate = wins / games if games > 0 else 0.0
    return winrate, wins, losses, ties


def main() -> None:
    registry = get_agent_registry()
    opponent_choices = sorted(name for name in registry.keys() if name != "q_learning")

    parser = argparse.ArgumentParser(
        description="Grafica la convergencia de Q-Learning midiendo winrate vs un rival."
    )
    parser.add_argument(
        "--opponent",
        choices=opponent_choices,
        default="rational",
        help="Agente rival.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=200,
        help="Partidas por Q-table.",
    )
    parser.add_argument(
        "--q-player",
        type=int,
        choices=[0, 1],
        default=0,
        help="Posicion del agente Q-Learning (0 o 1).",
    )
    parser.add_argument(
        "--q-tables-dir",
        default=os.path.join(os.path.dirname(__file__), "q_tables"),
        help="Directorio con Q-tables.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Nombre del grafico de salida (solo el nombre de archivo).",
    )
    parser.add_argument(
        "--csv-name",
        default=None,
        help="Nombre del CSV de salida (solo el nombre de archivo).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed para simulaciones reproducibles.",
    )
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(GAME_DIR, ".."))
    resultados_dir = os.path.join(project_root, "resultados")
    plots_dir = os.path.join(GAME_DIR, "plots")
    os.makedirs(resultados_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    entries = _list_q_tables(args.q_tables_dir)
    if not entries:
        raise FileNotFoundError(f"No se encontraron Q-tables en {args.q_tables_dir}")

    evaluations: list[TableEval] = []
    for fname, path, episode, _ in entries:
        winrate, wins, losses, ties = _evaluate_table(
            path, args.opponent, args.games, args.q_player, args.seed
        )
        evaluations.append(
            TableEval(
                name=fname,
                path=path,
                episode=episode,
                winrate=winrate,
                wins=wins,
                losses=losses,
                ties=ties,
            )
        )
        print(
            f"{fname} | winrate={winrate:.3f} | W={wins} L={losses} T={ties}"
        )

    x_values = [e.episode if e.episode is not None else idx for idx, e in enumerate(evaluations)]
    y_values = [e.winrate for e in evaluations]

    csv_name = args.csv_name or f"q_learning_convergence_vs_{args.opponent}.csv"
    csv_path = os.path.join(resultados_dir, os.path.basename(csv_name))
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["q_table", "episode", "winrate", "wins", "losses", "ties"],
        )
        writer.writeheader()
        for e in evaluations:
            writer.writerow(
                {
                    "q_table": e.name,
                    "episode": "" if e.episode is None else e.episode,
                    "winrate": f"{e.winrate:.6f}",
                    "wins": e.wins,
                    "losses": e.losses,
                    "ties": e.ties,
                }
            )

    output_name = args.output or f"q_learning_convergence_vs_{args.opponent}.png"
    output_path = os.path.join(plots_dir, os.path.basename(output_name))
    plt.figure(figsize=(10, 5))
    plt.plot(
        x_values,
        y_values,
        marker="o",
        linewidth=2,
        color="#2C7FB8",
        label="Winrate",
    )
    plt.title(f"Convergencia Q-Learning vs {args.opponent}")
    plt.xlabel("Episodios (millones)")
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
    plt.legend(title="Eje X: episodios en millones (M)", loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"CSV guardado en {csv_path}")
    print(f"Grafico guardado en {output_path}")


if __name__ == "__main__":
    main()
