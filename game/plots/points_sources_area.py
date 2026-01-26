import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GAME_DIR = PROJECT_ROOT / "game"
if str(GAME_DIR) not in sys.path:
    sys.path.insert(0, str(GAME_DIR))

from constantes import Acciones  # noqa: E402
from truco_env import TrucoEnv  # noqa: E402
from agents.registry import create_agent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stacked area charts for point sources by matchup."
    )
    parser.add_argument("--focal-agent", required=True, help="Agent to analyze")
    parser.add_argument(
        "--opponents",
        required=True,
        help="Comma-separated list of opponents",
    )
    parser.add_argument(
        "--results-dir",
        default="resultados",
        help="Directory containing or storing source CSV files",
    )
    parser.add_argument(
        "--auto-games",
        type=int,
        default=100,
        help="Number of games to simulate if source CSV is missing",
    )
    return parser.parse_args()


def _simulate_matchup(agent_0: str, agent_1: str, games: int, output_path: Path) -> None:
    env = TrucoEnv()
    agent_0_inst = create_agent(agent_0)
    agent_1_inst = create_agent(agent_1)

    fieldnames = [
        "game",
        "agent_0",
        "agent_1",
        "envido_j0",
        "truco_j0",
        "cartas_j0",
        "abandono_j0",
        "envido_j1",
        "truco_j1",
        "cartas_j1",
        "abandono_j1",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for game_id in range(1, games + 1):
            env.reset()
            done = False

            totals = {
                "envido_j0": 0,
                "truco_j0": 0,
                "cartas_j0": 0,
                "abandono_j0": 0,
                "envido_j1": 0,
                "truco_j1": 0,
                "cartas_j1": 0,
                "abandono_j1": 0,
            }

            while not done:
                player_id = env.get_current_player()
                action_mask = env.get_action_mask(player_id)
                if not any(action_mask):
                    break

                if player_id == 0:
                    chosen_action = agent_0_inst.choose_action(action_mask, env, player_id)
                else:
                    chosen_action = agent_1_inst.choose_action(action_mask, env, player_id)

                estado = env.logic.estado
                prev_turno_responder_envido = estado.turno_responder_envido
                prev_turno_responder_truco = estado.turno_responder_truco
                prev_nivel_truco = estado.nivel_truco
                prev_puntos_j0 = estado.puntos_jugador
                prev_puntos_j1 = estado.puntos_oponente

                _, _, done, _, _ = env.step(chosen_action, player_id)

                after_puntos_j0 = env.logic.estado.puntos_jugador
                after_puntos_j1 = env.logic.estado.puntos_oponente
                delta_j0 = after_puntos_j0 - prev_puntos_j0
                delta_j1 = after_puntos_j1 - prev_puntos_j1

                if delta_j0 == 0 and delta_j1 == 0:
                    continue

                if prev_turno_responder_envido and chosen_action in [
                    Acciones.QUIERO.value,
                    Acciones.NO_QUIERO.value,
                ]:
                    source = "envido"
                elif chosen_action == Acciones.NO_QUIERO.value and prev_turno_responder_truco:
                    source = "abandono"
                elif chosen_action == Acciones.IR_AL_MAZO.value:
                    source = "abandono"
                else:
                    source = "truco" if prev_nivel_truco > 0 else "cartas"

                if delta_j0 > 0:
                    totals[f"{source}_j0"] += delta_j0
                if delta_j1 > 0:
                    totals[f"{source}_j1"] += delta_j1

            writer.writerow(
                {
                    "game": game_id,
                    "agent_0": agent_0,
                    "agent_1": agent_1,
                    **totals,
                }
            )


def ensure_sources_csv(
    results_dir: Path, agent_0: str, agent_1: str, auto_games: int
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / f"{agent_0}vs{agent_1}sources.csv"
    swapped = results_dir / f"{agent_1}vs{agent_0}sources.csv"
    if target.exists():
        return target
    if swapped.exists():
        return swapped

    _simulate_matchup(agent_0, agent_1, auto_games, target)
    return target


def load_series(csv_path: Path, focal_agent: str) -> dict[str, list[float]]:
    series = {
        "envido": [],
        "truco": [],
        "cartas": [],
        "abandono": [],
    }
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            agent_0 = row.get("agent_0", "")
            agent_1 = row.get("agent_1", "")
            if focal_agent == agent_0:
                suffix = "j0"
            elif focal_agent == agent_1:
                suffix = "j1"
            else:
                raise ValueError(
                    f"Focal agent {focal_agent} not found in {csv_path.name}"
                )

            series["envido"].append(float(row.get(f"envido_{suffix}", 0)))
            series["truco"].append(float(row.get(f"truco_{suffix}", 0)))
            series["cartas"].append(float(row.get(f"cartas_{suffix}", 0)))
            series["abandono"].append(float(row.get(f"abandono_{suffix}", 0)))
    return series


def main() -> None:
    args = parse_args()
    results_dir = PROJECT_ROOT / args.results_dir

    opponents = [name.strip() for name in args.opponents.split(",") if name.strip()]
    if not opponents:
        raise ValueError("Provide at least one opponent.")

    matchup_series = []
    for opponent in opponents:
        csv_path = ensure_sources_csv(results_dir, args.focal_agent, opponent, args.auto_games)
        series = load_series(csv_path, args.focal_agent)
        matchup_series.append((opponent, series))

    fig, axes = plt.subplots(len(matchup_series), 1, figsize=(10, 4 * len(matchup_series)))
    if len(matchup_series) == 1:
        axes = [axes]

    colors = ["#2CA25F", "#E34A33", "#3182BD", "#7F7F7F"]
    labels = ["Envido", "Truco", "Cartas", "Abandono"]

    for ax, (opponent, series) in zip(axes, matchup_series, strict=False):
        x = np.arange(1, len(series["envido"]) + 1)
        ax.stackplot(
            x,
            series["envido"],
            series["truco"],
            series["cartas"],
            series["abandono"],
            labels=labels,
            colors=colors,
            alpha=0.9,
        )
        ax.set_title(f"{args.focal_agent} vs {opponent}")
        ax.set_ylabel("Puntos por partida")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    axes[-1].set_xlabel("Partida")
    axes[0].legend(loc="upper left", frameon=False)
    plt.tight_layout()

    output_dir = Path("game") / "plots" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.focal_agent}_income_sources_area.png"
    plt.savefig(output_path, dpi=150)


if __name__ == "__main__":
    main()
