import argparse
import os
import pickle
import sys
import random

GAME_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if GAME_DIR not in sys.path:
    sys.path.insert(0, GAME_DIR)

from truco_env import TrucoEnv
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


def _linear_epsilon(episode_idx, total_episodes, start, end):
    if total_episodes <= 1:
        return end
    ratio = episode_idx / (total_episodes - 1)
    return start + (end - start) * ratio


def _update_q_for_episode(q_table, episode_steps, alpha, gamma):
    G = 0.0
    for state, action, reward in reversed(episode_steps):
        G = reward + gamma * G
        old_q = _get_q(q_table, state, action)
        new_q = old_q + alpha * (G - old_q)
        _set_q(q_table, state, action, new_q)


def train(
    episodes,
    alpha,
    gamma,
    epsilon_start,
    epsilon_end,
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
            episode_steps = []
            epsilon = _linear_epsilon(episode_idx, episodes, epsilon_start, epsilon_end)

            while not done:
                player_id = env.get_current_player()
                action_mask = env.get_action_mask(player_id)
                if not any(action_mask):
                    break

                if player_id == q_player:
                    state = agent.encode_state(env, player_id)
                    action = _epsilon_greedy(agent.q_table, state, action_mask, epsilon)
                    if action is None:
                        break
                else:
                    action = opponent.choose_action(action_mask, env, player_id)

                _, reward, done, _, _ = env.step(action, player_id)

                if player_id == q_player:
                    reward_q = reward if q_player == 0 else -reward
                    episode_steps.append((state, action, reward_q))

            if episode_steps:
                _update_q_for_episode(agent.q_table, episode_steps, alpha, gamma)
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
    parser.add_argument("--episodes", type=int, default=100, help="Cantidad de partidas.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor.")
    parser.add_argument("--epsilon-start", type=float, default=0.5, help="Epsilon inicial.")
    parser.add_argument("--epsilon-end", type=float, default=0.1, help="Epsilon final.")
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
        args.epsilon_start,
        args.epsilon_end,
        args.reset_q_table,
        args.opponent,
        args.q_player,
    )
