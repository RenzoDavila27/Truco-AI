import argparse
import numpy as np
from constantes import Acciones
from truco_env import TrucoEnv
from agents.registry import create_agent, get_agent_registry


def get_human_action(action_mask):
    valid_actions_indices = [i for i, valid in enumerate(action_mask) if valid]

    print("\nAcciones válidas:")
    for i, action_index in enumerate(valid_actions_indices):
        print(f"{i}: {Acciones(action_index).name}")

    while True:
        try:
            choice = int(input("Elige una acción: "))
            if 0 <= choice < len(valid_actions_indices):
                return valid_actions_indices[choice]
            print("Opción inválida.")
        except ValueError:
            print("Entrada inválida. Introduce un número.")


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


def _format_cartas_jugadas(cartas_jugadas):
    palos = {0: "Espada", 1: "Basto", 2: "Oro", 3: "Copa"}
    def format_card(carta):
        return f"{carta[0]}-{palos.get(carta[1], carta[1])}"
    return [f"({format_card(c)}, J{jugador_id})" for c, jugador_id in cartas_jugadas]


def main(human_player, mode, agent_name):
    env = TrucoEnv()
    agent = create_agent(agent_name)

    env.reset()
    done = False
    cartas_mano = []

    while not done:
        player_id = env.get_current_player()
        env.render(mode=mode, player_id=human_player)

        action_mask = env.get_action_mask(player_id)
        if not np.any(action_mask):
            print("No hay acciones válidas. El juego ha terminado.")
            break

        envido_pendiente = env.logic.estado.turno_responder_envido
        if player_id == human_player:
            chosen_action = get_human_action(action_mask)
        else:
            chosen_action = agent.choose_action(action_mask, env, player_id)
            print(f"\nEl agente elige: {Acciones(chosen_action).name}")
            input("Presiona Enter para continuar...")

        prev_cartas = list(env.logic.estado.cartas_jugadas)
        if chosen_action in [
            Acciones.JUGAR_CARTA_1.value,
            Acciones.JUGAR_CARTA_2.value,
            Acciones.JUGAR_CARTA_3.value,
        ]:
            mano = (
                env.logic.estado.mano_jugador
                if player_id == 0
                else env.logic.estado.mano_oponente
            )
            idx = chosen_action
            if idx < len(mano):
                cartas_mano.append((mano[idx], player_id))
        envido_aceptado = envido_pendiente and chosen_action == Acciones.QUIERO.value
        puntos_envido = _calcular_tantos_envido(env.logic) if envido_aceptado else None
        _, reward, done, _, _ = env.step(chosen_action, player_id)
        if envido_aceptado and puntos_envido is not None:
            puntos_j0, puntos_j1 = puntos_envido
            print(f"Tantos envido: J0 {puntos_j0} | J1 {puntos_j1}")
        mano_terminada = prev_cartas and not env.logic.estado.cartas_jugadas
        if (mano_terminada or done) and cartas_mano:
            print(f"Cartas jugadas en la mano: {_format_cartas_jugadas(cartas_mano)}")
            cartas_mano = []
        print(f"\nRecompensa (perspectiva jugador 0): {reward}")
        if player_id == human_player:
            input("Presiona Enter para continuar...")

    estado = env.logic.estado
    print("\n¡Juego terminado!")
    print(f"Puntos finales: J0 {estado.puntos_jugador} | J1 {estado.puntos_oponente}")
    if estado.puntos_jugador == estado.puntos_oponente:
        print("Resultado: Empate.")
    elif estado.puntos_jugador > estado.puntos_oponente:
        print("Resultado: Gana J0.")
    else:
        print("Resultado: Gana J1.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1v1 por consola contra un agente.")
    parser.add_argument(
        "--human-player",
        type=int,
        choices=[0, 1],
        default=0,
        help="ID del jugador humano (0 o 1).",
    )
    parser.add_argument(
        "--mode",
        choices=["human", "debug"],
        default="human",
        help="Modo de render: human oculta la mano rival, debug muestra todo.",
    )
    parser.add_argument(
        "--agent",
        choices=sorted(get_agent_registry().keys()),
        default="rational",
        help="Agente rival.",
    )
    args = parser.parse_args()

    main(args.human_player, args.mode, args.agent)
