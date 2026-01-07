import argparse
import os
import numpy as np
from truco_env import TrucoEnv
from constantes import Acciones
from agents.random_agent import RandomAgent

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

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
            else:
                print("Opción inválida.")
        except ValueError:
            print("Entrada inválida. Introduce un número.")

def main(agent_type):
    env = TrucoEnv()
    
    if agent_type == 'random':
        agent = RandomAgent()
    elif agent_type == 'human':
        agent = None # Human input will be handled directly
    else:
        raise ValueError(f"Agente desconocido: {agent_type}")

    obs, info = env.reset()
    done = False

    while not done:
        #clear_console()
        env.render()

        player_id = env.logic.estado.turno_actual
        action_mask = env.get_action_mask(player_id)
        
        if not np.any(action_mask):
            print("No hay acciones válidas. El juego ha terminado.")
            break

        if agent_type == 'human':
            chosen_action = get_human_action(action_mask)
        else:
            chosen_action = agent.choose_action(action_mask)
            print(f"\nEl agente {agent_type} elige: {Acciones(chosen_action).name}")
            input("Presiona Enter para continuar...")


        obs, reward, done, truncated, info = env.step(chosen_action, player_id)

        print(f"\nRecompensa: {reward}")
        if agent_type == 'human':
            input("Presiona Enter para continuar...")

    print("\n¡Juego terminado!")
    env.render()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Juega contra un agente en el entorno de Truco.")
    parser.add_argument("agent", choices=["human", "random"], help="El tipo de agente contra el que jugar.")
    args = parser.parse_args()
    
    main(args.agent)
