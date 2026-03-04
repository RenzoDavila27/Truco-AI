import argparse
import math
import os
import pickle
import sys
import random
import numpy as np

GAME_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if GAME_DIR not in sys.path:
    sys.path.insert(0, GAME_DIR)

from truco_env import TrucoEnv
from constantes import Acciones
from agent_q_learning import QLearningAgent


QTABLE_DIR = os.path.join(os.path.dirname(__file__), "q_tables")
DEFAULT_QTABLE_NAME = "q_table.pkl"


def _resolve_qtable_path(name):
    if not name:
        name = DEFAULT_QTABLE_NAME
    root, ext = os.path.splitext(name)
    if not ext:
        name = f"{name}.pkl"
    if os.path.isabs(name) or os.sep in name or "/" in name:
        return name
    return os.path.join(QTABLE_DIR, name)


def _load_q_table(path):
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_q_table(q_table, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
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

def _update_hand(q_table, hand_steps, final_reward, alpha, gamma):
    if not hand_steps:
        return
    G = float(final_reward)
    for state, action, player_id in reversed(hand_steps):
        step_return = G if player_id == 0 else -G
        old_q = _get_q(q_table, state, action)
        new_q = old_q + alpha * (step_return - old_q)
        _set_q(q_table, state, action, new_q)
        G *= gamma


def _checkpoint_episodes(episodes, checkpoints):
    if checkpoints <= 0 or episodes <= 0:
        return set()
    checkpoints = min(checkpoints, max(0, episodes - 1))
    return {max(1, (i * episodes) // (checkpoints + 1)) for i in range(1, checkpoints + 1)}


def _seed_everything(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def train(episodes, alpha, gamma, epsilon, q_table_name, checkpoints, seed):
    env = TrucoEnv()
    if q_table_name == DEFAULT_QTABLE_NAME and seed is not None:
        q_table_name = f"q_table_{seed}.pkl"
    q_table_path = _resolve_qtable_path(q_table_name)
    agent = QLearningAgent(q_table_path=q_table_path)
    checkpoint_set = _checkpoint_episodes(episodes, checkpoints)
    _seed_everything(seed)

    try:
        for episode_idx in range(episodes):
            if seed is not None:
                random.seed(seed + episode_idx)
                np.random.seed(seed + episode_idx)
            env.reset(seed=None if seed is None else seed + episode_idx)
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

                state = agent.encode_state(env, player_id)
                action = _epsilon_greedy(agent.q_table, state, action_mask, current_epsilon)
                if action is None:
                    break

                _, reward, done, _, _ = env.step(action, player_id)
                hand_steps.append((state, action, player_id))

                if reward <= -5:
                    _update_hand(agent.q_table, [hand_steps.pop()], -1.0, alpha, gamma)
                    continue

                current_es_mano = env.logic.estado.es_mano
                hand_end = done or (current_es_mano != prev_es_mano)
                if hand_end:
                    delta_j0 = env.logic.estado.puntos_jugador - hand_start_points[0]
                    delta_j1 = env.logic.estado.puntos_oponente - hand_start_points[1]
                    points_diff = delta_j0 - delta_j1
                    final_reward = points_diff / 30.0
                    if action == Acciones.IR_AL_MAZO.value:
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
            if t in checkpoint_set:
                checkpoint_path = _resolve_qtable_path(f"{os.path.splitext(os.path.basename(q_table_path))[0]}_ep{t}")
                _save_q_table(agent.q_table, checkpoint_path)
    except KeyboardInterrupt:
        pass
    finally:
        _save_q_table(agent.q_table, q_table_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena un agente Q-Learning en self-play.")
    parser.add_argument("--episodes", type=int, default=100, help="Cantidad de episodios.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=1, help="Discount factor.")
    parser.add_argument("--epsilon", type=float, default=0.5, help="Epsilon para exploracion.")
    parser.add_argument(
        "--q-table-name",
        type=str,
        default=DEFAULT_QTABLE_NAME,
        help="Nombre o ruta de la Q-table (por defecto q_table.pkl).",
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        default=0,
        help="Cantidad de Q-tables intermedias a guardar.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed para entrenamiento reproducible.",
    )
    args = parser.parse_args()

    train(
        args.episodes,
        args.alpha,
        args.gamma,
        args.epsilon,
        args.q_table_name,
        args.checkpoints,
        args.seed,
    )
