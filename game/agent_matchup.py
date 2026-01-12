import argparse
import csv
from constantes import Acciones
from truco_env import TrucoEnv
from agents.registry import create_agent, get_agent_registry


def _play_game(agent_0, agent_1):
    env = TrucoEnv()
    env.reset()
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


def main(agent_0, agent_1, games, output_name=None, summary_name=None):
    if output_name is None:
        output_name = f"{agent_0}vs{agent_1}results.csv"
    if summary_name is None:
        summary_name = f"{agent_0}vs{agent_1}summary.txt"
    agent_0_name = agent_0
    agent_1_name = agent_1

    fieldnames = [
        "game",
        "agent_0",
        "agent_1",
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

    with open(output_name, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(1, games + 1):
            agent_0_inst = create_agent(agent_0_name)
            agent_1_inst = create_agent(agent_1_name)
            result = _play_game(agent_0_inst, agent_1_inst)
            result["game"] = i
            result["agent_0"] = agent_0_name
            result["agent_1"] = agent_1_name
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

    print(f"Resultados guardados en {output_name}")

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

    with open(summary_name, "w", encoding="utf-8") as summary:
        summary.write(f"Agente J0: {agent_0_name}\n")
        summary.write(f"Agente J1: {agent_1_name}\n")
        summary.write(f"Partidas: {games}\n")
        summary.write(f"Victorias J0: {totals['wins_j0']}\n")
        summary.write(f"Victorias J1: {totals['wins_j1']}\n")
        summary.write(f"Empates: {totals['ties']}\n")
        summary.write(f"Promedio puntos J0: {avg_points_j0:.2f}\n")
        summary.write(f"Promedio puntos J1: {avg_points_j1:.2f}\n")
        summary.write(f"Promedio manos jugadas: {avg_hands_played:.2f}\n")
        summary.write(f"Promedio manos ganadas J0: {avg_hands_won_j0:.2f}\n")
        summary.write(f"Promedio manos ganadas J1: {avg_hands_won_j1:.2f}\n")

    print(f"Resumen guardado en {summary_name}")


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
    args = parser.parse_args()

    main(args.agent_0, args.agent_1, args.games, args.output_csv, args.output_summary)
