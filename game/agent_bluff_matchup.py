import argparse
import csv
import os

from constantes import Acciones
from truco_env import TrucoEnv
from agents.registry import create_agent, get_agent_registry


def _format_hand(hand):
    return "|".join([f"{carta[0]}-{carta[1]}" for carta in hand])


def _hand_strength_stats(logic, hand):
    if not hand:
        return {
            "best": 0.0,
            "avg": 0.0,
            "top2": 0.0,
            "ranks": [],
        }

    ranks = [logic.obtener_ranking(carta) for carta in hand]
    strengths = [15 - rank for rank in ranks]  # Higher is stronger
    strengths_sorted = sorted(strengths, reverse=True)
    top2 = strengths_sorted[:2]
    return {
        "best": float(max(strengths)),
        "avg": float(sum(strengths) / len(strengths)),
        "top2": float(sum(top2)),
        "ranks": ranks,
    }

def _envido_points_for_player(estado, logic, player_id):
    mano = estado.mano_jugador if player_id == 0 else estado.mano_oponente
    cartas_jugadas = [c[0] for c in estado.cartas_jugadas if c[1] == player_id]
    return logic.calcular_puntos_envido(mano + cartas_jugadas)


def _resolve_truco_events(events, active_indices, points_j0, points_j1):
    for idx in active_indices:
        event = events[idx]
        if event.get("resolved"):
            continue

        points_before_j0 = event["points_before_j0"]
        points_before_j1 = event["points_before_j1"]
        delta_j0 = points_j0 - points_before_j0
        delta_j1 = points_j1 - points_before_j1

        if delta_j0 > 0:
            hand_winner = "J0"
        elif delta_j1 > 0:
            hand_winner = "J1"
        else:
            hand_winner = "Empate"

        caller_id = event["caller_id"]
        caller_points_delta = delta_j0 if caller_id == 0 else delta_j1
        caller_won_hand = 1 if caller_points_delta > 0 else 0

        event.update(
            {
                "points_after_j0": points_j0,
                "points_after_j1": points_j1,
                "delta_j0": delta_j0,
                "delta_j1": delta_j1,
                "caller_points_delta": caller_points_delta,
                "hand_winner": hand_winner,
                "caller_won_hand": caller_won_hand,
                "resolved": True,
            }
        )


def _resolve_envido_events(events, active_indices, points_j0, points_j1):
    for idx in active_indices:
        event = events[idx]
        if event.get("resolved"):
            continue

        points_before_j0 = event["points_before_j0"]
        points_before_j1 = event["points_before_j1"]
        delta_j0 = points_j0 - points_before_j0
        delta_j1 = points_j1 - points_before_j1

        caller_id = event["caller_id"]
        caller_points_delta = delta_j0 if caller_id == 0 else delta_j1

        if delta_j0 > 0:
            envido_winner = "J0"
        elif delta_j1 > 0:
            envido_winner = "J1"
        else:
            envido_winner = "Empate"

        event.update(
            {
                "points_after_j0": points_j0,
                "points_after_j1": points_j1,
                "delta_j0": delta_j0,
                "delta_j1": delta_j1,
                "caller_points_delta": caller_points_delta,
                "envido_winner": envido_winner,
                "resolved": True,
            }
        )


def _play_game(env, agent_0, agent_1, agent_0_name, agent_1_name, game_id):
    env.reset()
    done = False

    hand_index = 1
    events = []
    active_truco_indices = []
    active_envido_indices = []

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
        prev_turno_responder_envido = estado.turno_responder_envido
        prev_puntos_j0 = estado.puntos_jugador
        prev_puntos_j1 = estado.puntos_oponente
        prev_numero_ronda = estado.numero_ronda
        prev_nivel_truco = estado.nivel_truco
        prev_envido_total = estado.envido_total
        prev_envido_total_anterior = estado.envido_total_anterior
        prev_envido_finalizado = estado.envido_finalizado
        prev_estado_canto_envido = estado.estado_canto_envido

        if prev_turno_responder_truco and chosen_action in [
            Acciones.QUIERO.value,
            Acciones.NO_QUIERO.value,
            Acciones.RETRUCO.value,
            Acciones.VALE_CUATRO.value,
        ]:
            for idx in reversed(active_truco_indices):
                if not events[idx].get("response_action"):
                    response_action = Acciones(chosen_action).name
                    response_type = (
                        "accept"
                        if chosen_action == Acciones.QUIERO.value
                        else "decline"
                        if chosen_action == Acciones.NO_QUIERO.value
                        else "raise"
                    )
                    responder = "J0" if player_id == 0 else "J1"
                    responder_agent = agent_0_name if player_id == 0 else agent_1_name
                    events[idx].update(
                        {
                            "response_action": response_action,
                            "response_type": response_type,
                            "responder": responder,
                            "responder_agent": responder_agent,
                        }
                    )
                    break

        if prev_turno_responder_envido and chosen_action in [
            Acciones.QUIERO.value,
            Acciones.NO_QUIERO.value,
            Acciones.ENVIDO.value,
            Acciones.ENVIDO_ENVIDO.value,
            Acciones.REAL_ENVIDO.value,
            Acciones.FALTA_ENVIDO.value,
        ]:
            for idx in reversed(active_envido_indices):
                if not events[idx].get("response_action"):
                    response_action = Acciones(chosen_action).name
                    response_type = (
                        "accept"
                        if chosen_action == Acciones.QUIERO.value
                        else "decline"
                        if chosen_action == Acciones.NO_QUIERO.value
                        else "raise"
                    )
                    responder = "J0" if player_id == 0 else "J1"
                    responder_agent = agent_0_name if player_id == 0 else agent_1_name
                    events[idx].update(
                        {
                            "response_action": response_action,
                            "response_type": response_type,
                            "responder": responder,
                            "responder_agent": responder_agent,
                        }
                    )
                    break

        if chosen_action in [
            Acciones.TRUCO.value,
            Acciones.RETRUCO.value,
            Acciones.VALE_CUATRO.value,
        ]:
            caller = "J0" if player_id == 0 else "J1"
            opponent = "J1" if player_id == 0 else "J0"
            caller_agent = agent_0_name if player_id == 0 else agent_1_name
            opponent_agent = agent_1_name if player_id == 0 else agent_0_name

            caller_hand = (
                estado.mano_jugador if player_id == 0 else estado.mano_oponente
            )
            opponent_hand = (
                estado.mano_oponente if player_id == 0 else estado.mano_jugador
            )

            caller_stats = _hand_strength_stats(env.logic, caller_hand)
            opponent_stats = _hand_strength_stats(env.logic, opponent_hand)
            strength_diff_avg = caller_stats["avg"] - opponent_stats["avg"]

            event = {
                "game": game_id,
                "hand_index": hand_index,
                "agent_0": agent_0_name,
                "agent_1": agent_1_name,
                "event_type": "truco",
                "caller": caller,
                "caller_id": player_id,
                "caller_agent": caller_agent,
                "opponent": opponent,
                "opponent_agent": opponent_agent,
                "action": Acciones(chosen_action).name,
                "round_number": prev_numero_ronda,
                "truco_level_before": prev_nivel_truco,
                "envido_level_before": "",
                "envido_total_before": "",
                "envido_total_previous": "",
                "caller_envido_points": "",
                "opponent_envido_points": "",
                "envido_points_diff": "",
                "caller_weaker_envido": "",
                "envido_winner": "",
                "caller_hand": _format_hand(caller_hand),
                "opponent_hand": _format_hand(opponent_hand),
                "caller_best_strength": caller_stats["best"],
                "caller_avg_strength": caller_stats["avg"],
                "caller_top2_strength": caller_stats["top2"],
                "opponent_best_strength": opponent_stats["best"],
                "opponent_avg_strength": opponent_stats["avg"],
                "opponent_top2_strength": opponent_stats["top2"],
                "strength_diff_avg": strength_diff_avg,
                "caller_weaker_hand": 1 if strength_diff_avg < 0 else 0,
                "response_action": "",
                "response_type": "",
                "responder": "",
                "responder_agent": "",
                "points_before_j0": prev_puntos_j0,
                "points_before_j1": prev_puntos_j1,
                "points_after_j0": "",
                "points_after_j1": "",
                "delta_j0": "",
                "delta_j1": "",
                "caller_points_delta": "",
                "hand_winner": "",
                "caller_won_hand": "",
                "resolved": False,
            }

            events.append(event)
            active_truco_indices.append(len(events) - 1)

        if chosen_action in [
            Acciones.ENVIDO.value,
            Acciones.ENVIDO_ENVIDO.value,
            Acciones.REAL_ENVIDO.value,
            Acciones.FALTA_ENVIDO.value,
        ]:
            caller = "J0" if player_id == 0 else "J1"
            opponent = "J1" if player_id == 0 else "J0"
            caller_agent = agent_0_name if player_id == 0 else agent_1_name
            opponent_agent = agent_1_name if player_id == 0 else agent_0_name

            caller_points = _envido_points_for_player(estado, env.logic, player_id)
            opponent_points = _envido_points_for_player(estado, env.logic, 1 - player_id)
            points_diff = caller_points - opponent_points

            event = {
                "game": game_id,
                "hand_index": hand_index,
                "agent_0": agent_0_name,
                "agent_1": agent_1_name,
                "event_type": "envido",
                "caller": caller,
                "caller_id": player_id,
                "caller_agent": caller_agent,
                "opponent": opponent,
                "opponent_agent": opponent_agent,
                "action": Acciones(chosen_action).name,
                "round_number": prev_numero_ronda,
                "truco_level_before": "",
                "envido_level_before": prev_estado_canto_envido,
                "envido_total_before": prev_envido_total,
                "envido_total_previous": prev_envido_total_anterior,
                "caller_envido_points": caller_points,
                "opponent_envido_points": opponent_points,
                "envido_points_diff": points_diff,
                "caller_weaker_envido": 1 if caller_points < 25 else 0,
                "envido_winner": "",
                "caller_hand": _format_hand(
                    estado.mano_jugador if player_id == 0 else estado.mano_oponente
                ),
                "opponent_hand": _format_hand(
                    estado.mano_oponente if player_id == 0 else estado.mano_jugador
                ),
                "caller_best_strength": "",
                "caller_avg_strength": "",
                "caller_top2_strength": "",
                "opponent_best_strength": "",
                "opponent_avg_strength": "",
                "opponent_top2_strength": "",
                "strength_diff_avg": "",
                "caller_weaker_hand": "",
                "response_action": "",
                "response_type": "",
                "responder": "",
                "responder_agent": "",
                "points_before_j0": prev_puntos_j0,
                "points_before_j1": prev_puntos_j1,
                "points_after_j0": "",
                "points_after_j1": "",
                "delta_j0": "",
                "delta_j1": "",
                "caller_points_delta": "",
                "hand_winner": "",
                "caller_won_hand": "",
                "resolved": False,
            }

            events.append(event)
            active_envido_indices.append(len(events) - 1)

        _, _, done, _, _ = env.step(chosen_action, player_id)

        if not prev_envido_finalizado and env.logic.estado.envido_finalizado:
            _resolve_envido_events(
                events,
                active_envido_indices,
                env.logic.estado.puntos_jugador,
                env.logic.estado.puntos_oponente,
            )
            active_envido_indices = []

        mano_terminada = False
        if prev_cartas and not env.logic.estado.cartas_jugadas:
            mano_terminada = True
        elif chosen_action == Acciones.IR_AL_MAZO.value:
            mano_terminada = True
        elif chosen_action == Acciones.NO_QUIERO.value and prev_turno_responder_truco:
            mano_terminada = True

        if mano_terminada:
            _resolve_truco_events(
                events,
                active_truco_indices,
                env.logic.estado.puntos_jugador,
                env.logic.estado.puntos_oponente,
            )
            active_truco_indices = []
            hand_index += 1

    if active_truco_indices:
        _resolve_truco_events(
            events,
            active_truco_indices,
            env.logic.estado.puntos_jugador,
            env.logic.estado.puntos_oponente,
        )
    if active_envido_indices:
        _resolve_envido_events(
            events,
            active_envido_indices,
            env.logic.estado.puntos_jugador,
            env.logic.estado.puntos_oponente,
        )

    return events


def main(agent_0, agent_1, games, output_name=None, summary_name=None):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(project_root, "resultados")
    os.makedirs(results_dir, exist_ok=True)

    if output_name is None:
        output_name = f"{agent_0}vs{agent_1}bluff_results.csv"
    if summary_name is None:
        summary_name = f"{agent_0}vs{agent_1}bluff_summary.txt"

    output_path = os.path.join(results_dir, os.path.basename(output_name))
    summary_path = os.path.join(results_dir, os.path.basename(summary_name))

    env = TrucoEnv()
    agent_0_inst = create_agent(agent_0)
    agent_1_inst = create_agent(agent_1)

    fieldnames = [
        "game",
        "hand_index",
        "agent_0",
        "agent_1",
        "event_type",
        "caller",
        "caller_id",
        "caller_agent",
        "opponent",
        "opponent_agent",
        "action",
        "round_number",
        "truco_level_before",
        "envido_level_before",
        "envido_total_before",
        "envido_total_previous",
        "caller_envido_points",
        "opponent_envido_points",
        "envido_points_diff",
        "caller_weaker_envido",
        "envido_winner",
        "caller_hand",
        "opponent_hand",
        "caller_best_strength",
        "caller_avg_strength",
        "caller_top2_strength",
        "opponent_best_strength",
        "opponent_avg_strength",
        "opponent_top2_strength",
        "strength_diff_avg",
        "caller_weaker_hand",
        "response_action",
        "response_type",
        "responder",
        "responder_agent",
        "points_before_j0",
        "points_before_j1",
        "points_after_j0",
        "points_after_j1",
        "delta_j0",
        "delta_j1",
        "caller_points_delta",
        "hand_winner",
        "caller_won_hand",
    ]

    totals = {
        "calls": {agent_0: 0, agent_1: 0},
        "bluff_calls": {agent_0: 0, agent_1: 0},
        "won_calls": {agent_0: 0, agent_1: 0},
        "envido_calls": {agent_0: 0, agent_1: 0},
        "envido_bluff_calls": {agent_0: 0, agent_1: 0},
        "accepted": 0,
        "declined": 0,
        "raised": 0,
    }

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for game_id in range(1, games + 1):
            events = _play_game(
                env,
                agent_0_inst,
                agent_1_inst,
                agent_0,
                agent_1,
                game_id,
            )
            for event in events:
                writer.writerow({k: event.get(k, "") for k in fieldnames})
                caller_agent = event["caller_agent"]
                if event.get("event_type") == "truco":
                    totals["calls"][caller_agent] += 1
                    if event["caller_weaker_hand"]:
                        totals["bluff_calls"][caller_agent] += 1
                    if event.get("caller_won_hand") == 1:
                        totals["won_calls"][caller_agent] += 1
                elif event.get("event_type") == "envido":
                    totals["envido_calls"][caller_agent] += 1
                    if event["caller_weaker_envido"]:
                        totals["envido_bluff_calls"][caller_agent] += 1
                if event["response_type"] == "accept":
                    totals["accepted"] += 1
                elif event["response_type"] == "decline":
                    totals["declined"] += 1
                elif event["response_type"] == "raise":
                    totals["raised"] += 1

    with open(summary_path, "w", encoding="utf-8") as summary:
        summary.write(f"Agente J0: {agent_0}\n")
        summary.write(f"Agente J1: {agent_1}\n")
        summary.write(f"Partidas: {games}\n")
        summary.write(f"Llamados Truco totales: {sum(totals['calls'].values())}\n")
        summary.write(f"Llamados Envido totales: {sum(totals['envido_calls'].values())}\n")
        summary.write(f"Aceptados: {totals['accepted']}\n")
        summary.write(f"Rechazados: {totals['declined']}\n")
        summary.write(f"Subidas: {totals['raised']}\n")
        for agent_name in [agent_0, agent_1]:
            calls = totals["calls"][agent_name]
            bluff_calls = totals["bluff_calls"][agent_name]
            won_calls = totals["won_calls"][agent_name]
            envido_calls = totals["envido_calls"][agent_name]
            envido_bluff_calls = totals["envido_bluff_calls"][agent_name]
            bluff_rate = (bluff_calls / calls) if calls else 0.0
            win_rate = (won_calls / calls) if calls else 0.0
            envido_bluff_rate = (
                envido_bluff_calls / envido_calls if envido_calls else 0.0
            )
            summary.write(f"\nAgente {agent_name}:\n")
            summary.write(f"Llamados Truco: {calls}\n")
            summary.write(f"Llamados con mano mas debil: {bluff_calls}\n")
            summary.write(f"Tasa de mentira: {bluff_rate:.2f}\n")
            summary.write(f"Manos ganadas tras cantar: {won_calls}\n")
            summary.write(f"Tasa de exito en cantos: {win_rate:.2f}\n")
            summary.write(f"Llamados Envido: {envido_calls}\n")
            summary.write(f"Llamados Envido con tanto menor: {envido_bluff_calls}\n")
            summary.write(f"Tasa de mentira Envido: {envido_bluff_rate:.2f}\n")

    print(f"Resultados guardados en {output_path}")
    print(f"Resumen guardado en {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simula partidas y registra datos de cantos de Truco para evaluar la mentira."
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
