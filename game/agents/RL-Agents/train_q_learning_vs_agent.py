import argparse
import math
import os
import pickle
import sys
import random

GAME_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if GAME_DIR not in sys.path:
    sys.path.insert(0, GAME_DIR)

from truco_env import TrucoEnv
from constantes import Acciones
from agent_q_learning import QLearningAgent
from agents.registry import create_agent, get_agent_registry


QTABLE_PATH = os.path.join(os.path.dirname(__file__), "q_tables", "q_table.pkl")


def _save_q_table(q_table):
    os.makedirs(os.path.dirname(QTABLE_PATH), exist_ok=True)
    with open(QTABLE_PATH, "wb") as f:
        pickle.dump(q_table, f)


def _get_q(q_table, state, action):
    return q_table.get((state, action), 0.0)


def _set_q(q_table, state, action, value):
    q_table[(state, action)] = value


def _epsilon_greedy(q_table, state, action_mask, epsilon):
    valid_actions = [i for i, valid in enumerate(action_mask) if valid]
    if not valid_actions:
        return None
    if random.random() < epsilon:
        return random.choice(valid_actions)
    best_action = None
    best_value = None
    for action in valid_actions:
        q_val = _get_q(q_table, state, action)
        if best_value is None or q_val > best_value:
            best_value = q_val
            best_action = action
    return best_action


def _update_hand(q_table, hand_steps, final_reward, alpha, gamma):
    if not hand_steps:
        return
    G = float(final_reward)
    for state, action in reversed(hand_steps):
        old_q = _get_q(q_table, state, action)
        new_q = old_q + alpha * (G - old_q)
        _set_q(q_table, state, action, new_q)
        G *= gamma


def train(
    episodes,
    alpha,
    gamma,
    epsilon,
    reset_q_table,
    opponent_name,
    q_player,
):
    env = TrucoEnv()
    agent = QLearningAgent(q_table_path=QTABLE_PATH)
    if reset_q_table:
        agent.q_table = {}

    opponent = create_agent(opponent_name)

    try:
        for episode_idx in range(episodes):
            env.reset()
            done = False
            hand_steps = []
            prev_es_mano = env.logic.estado.es_mano
            hand_start_points = (
                env.logic.estado.puntos_jugador,
                env.logic.estado.puntos_oponente,
            )
            t = episode_idx + 1
            current_epsilon = epsilon * math.cos((t * math.pi) / (2 * episodes))
            if current_epsilon < 0:
                current_epsilon = 0.0

            while not done:
                player_id = env.get_current_player()
                action_mask = env.get_action_mask(player_id)
                if not any(action_mask):
                    break

                if player_id == q_player:
                    state = agent.encode_state(env, player_id)
                    action = _epsilon_greedy(
                        agent.q_table, state, action_mask, current_epsilon
                    )
                    if action is None:
                        break
                else:
                    action = opponent.choose_action(action_mask, env, player_id)

                _, reward, done, _, _ = env.step(action, player_id)

                if player_id == q_player:
                    hand_steps.append((state, action))

                if player_id == q_player and reward <= -5:
                    _update_hand(agent.q_table, [hand_steps.pop()], -1.0, alpha, gamma)
                    continue

                current_es_mano = env.logic.estado.es_mano
                hand_end = done or (current_es_mano != prev_es_mano)
                if hand_end:
                    delta_j0 = env.logic.estado.puntos_jugador - hand_start_points[0]
                    delta_j1 = env.logic.estado.puntos_oponente - hand_start_points[1]
                    if q_player == 0:
                        points_diff = delta_j0 - delta_j1
                    else:
                        points_diff = delta_j1 - delta_j0
                    final_reward = points_diff / 30.0
                    if player_id == q_player and action == Acciones.IR_AL_MAZO.value:
                        final_reward -= 0.1
                    if final_reward > 1.0:
                        final_reward = 1.0
                    elif final_reward < -1.0:
                        final_reward = -1.0

                    _update_hand(agent.q_table, hand_steps, final_reward, alpha, gamma)
                    hand_steps = []
                    hand_start_points = (
                        env.logic.estado.puntos_jugador,
                        env.logic.estado.puntos_oponente,
                    )
                    prev_es_mano = current_es_mano

            _update_hand(agent.q_table, hand_steps, 0.0, alpha, gamma)
            if t % 1000 == 0 or t == episodes:
                print(
                    f"Episodio {t}/{episodes} | epsilon={current_epsilon:.4f} | Q-size={len(agent.q_table)}"
                )
    except KeyboardInterrupt:
        pass
    finally:
        _save_q_table(agent.q_table)


if __name__ == "__main__":
    registry = get_agent_registry()
    opponent_choices = sorted(name for name in registry.keys() if name != "q_learning")

    parser = argparse.ArgumentParser(
        description="Entrena Q-Learning contra un agente fijo (no self-play)."
    )
    parser.add_argument("--episodes", type=int, default=100, help="Cantidad de episodios.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=1, help="Discount factor.")
    parser.add_argument("--epsilon", type=float, default=0.5, help="Epsilon para exploracion.")
    parser.add_argument(
        "--reset-q-table",
        action="store_true",
        help="Reinicia la Q-table antes de entrenar.",
    )
    parser.add_argument(
        "--opponent",
        choices=opponent_choices,
        default="rational",
        help="Agente oponente.",
    )
    parser.add_argument(
        "--q-player",
        type=int,
        choices=[0, 1],
        default=0,
        help="Posicion del agente Q-Learning (0 o 1).",
    )
    args = parser.parse_args()

    train(
        args.episodes,
        args.alpha,
        args.gamma,
        args.epsilon,
        args.reset_q_table,
        args.opponent,
        args.q_player,
    )
