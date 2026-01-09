import argparse
import numpy as np
from constantes import Acciones
from truco_env import TrucoEnv
from agents.registry import create_agent, get_agent_registry


def _calcular_tantos_envido(logic):
    estado = logic.estado
    puntos_j0 = logic.calcular_puntos_envido(
        estado.mano_jugador
        + [c[0] for c in estado.cartas_jugadas if c[1] == 0]
    )
    puntos_j1 = logic.calcular_puntos_envido(
        estado.mano_oponente
        + [c[0] for c in estado.cartas_jugadas if c[1] == 1]
    )
    return puntos_j0, puntos_j1


def _run_match(agent_0, agent_1, render_mode):
    env = TrucoEnv()
    agent_0 = create_agent(agent_0)
    agent_1 = create_agent(agent_1)

    env.reset()
    done = False

    while not done:
        player_id = env.get_current_player()
        if render_mode:
            env.render(mode=render_mode, player_id=player_id)

        action_mask = env.get_action_mask(player_id)
        if not np.any(action_mask):
            print("No hay acciones válidas. El juego ha terminado.")
            break

        if player_id == 0:
            chosen_action = agent_0.choose_action(action_mask, env, player_id)
        else:
            chosen_action = agent_1.choose_action(action_mask, env, player_id)

        envido_pendiente = env.logic.estado.turno_responder_envido
        envido_aceptado = envido_pendiente and chosen_action == Acciones.QUIERO.value
        puntos_envido = _calcular_tantos_envido(env.logic) if envido_aceptado else None
        _, reward, done, _, _ = env.step(chosen_action, player_id)
        if envido_aceptado and puntos_envido is not None:
            puntos_j0, puntos_j1 = puntos_envido
            print(f"Tantos envido: J0 {puntos_j0} | J1 {puntos_j1}")
        if render_mode:
            print(f"J{player_id} juega: {Acciones(chosen_action).name} | reward J0: {reward}")

    estado = env.logic.estado
    print("Partida terminada.")
    print(f"Puntos finales: J0 {estado.puntos_jugador} | J1 {estado.puntos_oponente}")
    if estado.puntos_jugador == estado.puntos_oponente:
        print("Resultado: Empate.")
    elif estado.puntos_jugador > estado.puntos_oponente:
        print("Resultado: Gana J0.")
    else:
        print("Resultado: Gana J1.")


def main(render_mode, agent_0, agent_1):
    _run_match(agent_0, agent_1, render_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simula una partida random vs random.")
    parser.add_argument(
        "--render",
        choices=["human", "debug"],
        default="human",
        help="Modo de render.",
    )
    parser.add_argument(
        "--agent-0",
        choices=sorted(get_agent_registry().keys()),
        default="rational",
        help="Agente para J0.",
    )
    parser.add_argument(
        "--agent-1",
        choices=sorted(get_agent_registry().keys()),
        default="random",
        help="Agente para J1.",
    )
    args = parser.parse_args()

    main(args.render, args.agent_0, args.agent_1)
