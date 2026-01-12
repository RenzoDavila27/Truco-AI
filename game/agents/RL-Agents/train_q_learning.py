import argparse
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


QTABLE_PATH = os.path.join(os.path.dirname(__file__), "q_tables", "q_table.pkl")


def _load_q_table():
    if not os.path.exists(QTABLE_PATH):
        return {}
    with open(QTABLE_PATH, "rb") as f:
        return pickle.load(f)


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


def _max_q_for_state(q_table, state, action_mask):
    valid_actions = [i for i, valid in enumerate(action_mask) if valid]
    if not valid_actions:
        return 0.0
    return max(_get_q(q_table, state, a) for a in valid_actions)


def train(episodes, alpha, gamma, epsilon):
    env = TrucoEnv()
    agent = QLearningAgent(q_table_path=QTABLE_PATH)

    try:
        for _ in range(episodes):
            env.reset()
            done = False
            episode = []

            while not done:
                player_id = env.get_current_player()
                action_mask = env.get_action_mask(player_id)
                if not any(action_mask):
                    break

                state = agent.encode_state(env, player_id)
                action = _epsilon_greedy(agent.q_table, state, action_mask, epsilon)
                if action is None:
                    break

                _, reward, done, _, _ = env.step(action, player_id)
                episode.append((state, action, player_id, reward))

            G = 0.0
            for state, action, player_id, reward in reversed(episode):
                step_reward = reward if player_id == 0 else -reward
                G = step_reward + gamma * G
                old_q = _get_q(agent.q_table, state, action)
                new_q = old_q + alpha * (G - old_q)
                _set_q(agent.q_table, state, action, new_q)
    except KeyboardInterrupt:
        pass
    finally:
        _save_q_table(agent.q_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena un agente Q-Learning en self-play.")
    parser.add_argument("--episodes", type=int, default=100, help="Cantidad de episodios.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon para exploracion.")
    args = parser.parse_args()

    train(args.episodes, args.alpha, args.gamma, args.epsilon)
