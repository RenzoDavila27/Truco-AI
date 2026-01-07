import argparse
import numpy as np
from constantes import Acciones
from truco_env import TrucoEnv
from agents.random_agent import RandomAgent


def main(render_mode):
    env = TrucoEnv()
    agent_0 = RandomAgent()
    agent_1 = RandomAgent()

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
            chosen_action = agent_0.choose_action(action_mask)
        else:
            chosen_action = agent_1.choose_action(action_mask)

        _, reward, done, _, _ = env.step(chosen_action, player_id)
        if render_mode:
            print(f"J{player_id} juega: {Acciones(chosen_action).name} | reward J0: {reward}")

    print("Partida terminada.")
    if render_mode:
        env.render(mode=render_mode, player_id=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simula una partida random vs random.")
    parser.add_argument(
        "--render",
        choices=["human", "debug"],
        default=None,
        help="Modo de render (opcional).",
    )
    args = parser.parse_args()

    main(args.render)
