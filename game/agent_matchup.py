import argparse
import csv
import os
import random
import numpy as np
from constantes import Acciones
from truco_env import TrucoEnv
from agents.registry import create_agent, get_agent_registry

DEFAULT_QTABLE_NAME = "q_table.pkl"
RL_QTABLE_DIR = os.path.join(
    os.path.dirname(__file__),
    "agents",
    "RL-Agents",
    "q_tables",
)


def _resolve_qtable_path(name):
    if not name:
        name = DEFAULT_QTABLE_NAME
    root, ext = os.path.splitext(name)
    if not ext:
        name = f"{name}.pkl"
    if os.path.isabs(name) or os.sep in name or "/" in name:
        return name
    return os.path.join(RL_QTABLE_DIR, name)


def _play_game(env, agent_0, agent_1, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    env.reset(seed=seed)
    done = False

    hands_played = 0
    hands_won_j0 = 0
    hands_won_j1 = 0

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
        prev_puntos_j0 = estado.puntos_jugador
        prev_puntos_j1 = estado.puntos_oponente

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
                hands_played += 1
                delta_j0 = env.logic.estado.puntos_jugador - prev_puntos_j0
                delta_j1 = env.logic.estado.puntos_oponente - prev_puntos_j1
                if delta_j0 > 0:
                    hands_won_j0 += 1
                elif delta_j1 > 0:
                    hands_won_j1 += 1

    estado = env.logic.estado
    points_j0 = estado.puntos_jugador
    points_j1 = estado.puntos_oponente

    if points_j0 > points_j1:
        winner = "J0"
    elif points_j1 > points_j0:
        winner = "J1"
    else:
        winner = "Empate"

    return {
        "winner": winner,
        "points_j0": points_j0,
        "points_j1": points_j1,
        "points_lost_j0": points_j1,
        "points_lost_j1": points_j0,
        "hands_played": hands_played,
        "hands_won_j0": hands_won_j0,
        "hands_won_j1": hands_won_j1,
    }


def main(
    agent_0,
    agent_1,
    games,
    output_name=None,
    summary_name=None,
    q_table_name=DEFAULT_QTABLE_NAME,
    q_table_j0=None,
    q_table_j1=None,
    seed=None,
):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(project_root, "resultados")
    os.makedirs(results_dir, exist_ok=True)

    if output_name is None:
        output_name = f"{agent_0}vs{agent_1}results.csv"
    if summary_name is None:
        summary_name = f"{agent_0}vs{agent_1}summary.txt"
    output_path = os.path.join(results_dir, os.path.basename(output_name))
    summary_path = os.path.join(results_dir, os.path.basename(summary_name))
    agent_0_name = agent_0
    agent_1_name = agent_1

    fieldnames = [
        "game",
        "agent_0",
        "agent_1",
        "q_table_j0",
        "q_table_j1",
        "winner",
        "points_j0",
        "points_j1",
        "points_lost_j0",
        "points_lost_j1",
        "hands_played",
        "hands_won_j0",
        "hands_won_j1",
    ]

    totals = {
        "wins_j0": 0,
        "wins_j1": 0,
        "ties": 0,
        "points_j0": 0,
        "points_j1": 0,
        "hands_played": 0,
        "hands_won_j0": 0,
        "hands_won_j1": 0,
    }

    env = TrucoEnv()
    if agent_0_name == "q_learning":
        q_table_j0_path = _resolve_qtable_path(q_table_j0 or q_table_name)
    else:
        q_table_j0_path = None
    if agent_1_name == "q_learning":
        q_table_j1_path = _resolve_qtable_path(q_table_j1 or q_table_name)
    else:
        q_table_j1_path = None

    q_table_j0_label = q_table_j0_path if q_table_j0_path else "N/A"
    q_table_j1_label = q_table_j1_path if q_table_j1_path else "N/A"

    agent_0_inst = (
        create_agent(agent_0_name, q_table_path=q_table_j0_path)
        if agent_0_name == "q_learning"
        else create_agent(agent_0_name)
    )
    agent_1_inst = (
        create_agent(agent_1_name, q_table_path=q_table_j1_path)
        if agent_1_name == "q_learning"
        else create_agent(agent_1_name)
    )

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(1, games + 1):
            game_seed = None if seed is None else seed + i
            result = _play_game(env, agent_0_inst, agent_1_inst, seed=game_seed)
            result["game"] = i
            result["agent_0"] = agent_0_name
            result["agent_1"] = agent_1_name
            result["q_table_j0"] = q_table_j0_label
            result["q_table_j1"] = q_table_j1_label
            writer.writerow(result)
            if result["winner"] == "J0":
                totals["wins_j0"] += 1
            elif result["winner"] == "J1":
                totals["wins_j1"] += 1
            else:
                totals["ties"] += 1
            totals["points_j0"] += result["points_j0"]
            totals["points_j1"] += result["points_j1"]
            totals["hands_played"] += result["hands_played"]
            totals["hands_won_j0"] += result["hands_won_j0"]
            totals["hands_won_j1"] += result["hands_won_j1"]

    print(f"Resultados guardados en {output_path}")

    if games > 0:
        avg_points_j0 = totals["points_j0"] / games
        avg_points_j1 = totals["points_j1"] / games
        avg_hands_played = totals["hands_played"] / games
        avg_hands_won_j0 = totals["hands_won_j0"] / games
        avg_hands_won_j1 = totals["hands_won_j1"] / games
    else:
        avg_points_j0 = 0
        avg_points_j1 = 0
        avg_hands_played = 0
        avg_hands_won_j0 = 0
        avg_hands_won_j1 = 0

    with open(summary_path, "w", encoding="utf-8") as summary:
        summary.write(f"Agente J0: {agent_0_name}\n")
        summary.write(f"Agente J1: {agent_1_name}\n")
        summary.write(f"Q-table J0: {q_table_j0_label}\n")
        summary.write(f"Q-table J1: {q_table_j1_label}\n")
        summary.write(f"Partidas: {games}\n")
        summary.write(f"Victorias J0: {totals['wins_j0']}\n")
        summary.write(f"Victorias J1: {totals['wins_j1']}\n")
        summary.write(f"Empates: {totals['ties']}\n")
        summary.write(f"Promedio puntos J0: {avg_points_j0:.2f}\n")
        summary.write(f"Promedio puntos J1: {avg_points_j1:.2f}\n")
        summary.write(f"Promedio manos jugadas: {avg_hands_played:.2f}\n")
        summary.write(f"Promedio manos ganadas J0: {avg_hands_won_j0:.2f}\n")
        summary.write(f"Promedio manos ganadas J1: {avg_hands_won_j1:.2f}\n")

    print(f"Resumen guardado en {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simula multiples partidas entre dos agentes y guarda resultados."
    )
    parser.add_argument(
        "--agent-0",
        choices=sorted(get_agent_registry().keys()),
        default="random",
        help="Agente para J0.",
    )
    parser.add_argument(
        "--agent-1",
        choices=sorted(get_agent_registry().keys()),
        default="rational",
        help="Agente para J1.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Cantidad de partidas a simular.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Nombre del archivo CSV de salida.",
    )
    parser.add_argument(
        "--output-summary",
        default=None,
        help="Nombre del archivo de resumen.",
    )
    parser.add_argument(
        "--q-table-name",
        default=DEFAULT_QTABLE_NAME,
        help="Nombre o ruta de la Q-table a usar si hay agente q_learning.",
    )
    parser.add_argument(
        "--q-table-j0",
        default=None,
        help="Nombre o ruta de la Q-table para J0 (si es q_learning).",
    )
    parser.add_argument(
        "--q-table-j1",
        default=None,
        help="Nombre o ruta de la Q-table para J1 (si es q_learning).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed para simulaciones reproducibles.",
    )
    args = parser.parse_args()

    main(
        args.agent_0,
        args.agent_1,
        args.games,
        args.output_csv,
        args.output_summary,
        args.q_table_name,
        args.q_table_j0,
        args.q_table_j1,
        args.seed,
    )
