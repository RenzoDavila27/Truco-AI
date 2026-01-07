import argparse
import numpy as np
from constantes import Acciones
from truco_env import TrucoEnv
from agents.random_agent import RandomAgent


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


def main(human_player, mode):
    env = TrucoEnv()
    agent = RandomAgent()

    env.reset()
    done = False

    while not done:
        player_id = env.logic.estado.turno_actual
        env.render(mode=mode, player_id=human_player)

        action_mask = env.get_action_mask(player_id)
        if not np.any(action_mask):
            print("No hay acciones válidas. El juego ha terminado.")
            break

        if player_id == human_player:
            chosen_action = get_human_action(action_mask)
        else:
            chosen_action = agent.choose_action(action_mask)
            print(f"\nEl agente elige: {Acciones(chosen_action).name}")
            input("Presiona Enter para continuar...")

        _, reward, done, _, _ = env.step(chosen_action, player_id)
        print(f"\nRecompensa (perspectiva jugador 0): {reward}")
        if player_id == human_player:
            input("Presiona Enter para continuar...")

    print("\n¡Juego terminado!")
    env.render(mode="debug", player_id=human_player)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1v1 por consola contra un agente random.")
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
    args = parser.parse_args()

    main(args.human_player, args.mode)
