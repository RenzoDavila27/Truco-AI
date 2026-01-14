import argparse
import os
import sys
import numpy as np

GAME_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if GAME_DIR not in sys.path:
    sys.path.insert(0, GAME_DIR)

from constantes import Acciones
from truco_env import TrucoEnv
from agent_policy_gradient import PolicyGradientAgent


def _is_hand_end(prev_cartas, prev_turno_responder_truco, action, env):
    if prev_cartas and not env.logic.estado.cartas_jugadas:
        return True
    if action == Acciones.IR_AL_MAZO.value:
        return True
    if action == Acciones.NO_QUIERO.value and prev_turno_responder_truco:
        return True
    return False


def _compute_returns(rewards, gamma):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return np.array(returns, dtype=np.float32)


def _policy_grad_for_sample(obs, probs, action, grad_logp_scale):
    one_hot = np.zeros_like(probs)
    one_hot[action] = 1.0
    grad_logits = (one_hot - probs) * grad_logp_scale
    grad_Wp = np.outer(obs, grad_logits)
    grad_bp = grad_logits
    return grad_Wp, grad_bp


def _train_on_hand(agent, hand_steps, lr_policy, lr_value, gamma, clip_eps, epochs):
    obs_batch = np.array([s["obs"] for s in hand_steps], dtype=np.float32)
    mask_batch = np.array([s["mask"] for s in hand_steps], dtype=bool)
    actions = np.array([s["action"] for s in hand_steps], dtype=np.int64)
    rewards = np.array([s["reward"] for s in hand_steps], dtype=np.float32)
    old_logp = np.array([s["logp"] for s in hand_steps], dtype=np.float32)
    values = np.array([s["value"] for s in hand_steps], dtype=np.float32)

    returns = _compute_returns(rewards, gamma)
    advantages = returns - values
    if advantages.size > 1:
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

    for _ in range(epochs):
        grad_Wp = np.zeros_like(agent.Wp)
        grad_bp = np.zeros_like(agent.bp)
        grad_Wv = np.zeros_like(agent.Wv)
        grad_bv = 0.0

        for i in range(len(hand_steps)):
            obs = obs_batch[i]
            mask = mask_batch[i]
            action = actions[i]
            adv = advantages[i]
            old_lp = old_logp[i]

            logits, value, probs = agent.predict(obs, mask)
            prob_action = max(probs[action], 1e-12)
            new_logp = float(np.log(prob_action))
            ratio = float(np.exp(new_logp - old_lp))
            clipped_ratio = float(np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps))
            use_ratio = ratio if (ratio * adv) <= (clipped_ratio * adv) else clipped_ratio

            if use_ratio == ratio:
                grad_logp_scale = -ratio * adv
            else:
                grad_logp_scale = 0.0

            gWp, gBp = _policy_grad_for_sample(obs, probs, action, grad_logp_scale)
            grad_Wp += gWp
            grad_bp += gBp

            ret = returns[i]
            grad_v = (value - ret)
            grad_Wv += obs * grad_v
            grad_bv += grad_v

        n = max(1, len(hand_steps))
        grad_Wp /= n
        grad_bp /= n
        grad_Wv /= n
        grad_bv /= n

        agent.Wp -= lr_policy * grad_Wp
        agent.bp -= lr_policy * grad_bp
        agent.Wv -= lr_value * grad_Wv
        agent.bv -= lr_value * grad_bv


def train(
    hands,
    gamma,
    lr_policy,
    lr_value,
    clip_eps,
    epochs,
    reset_model,
):
    env = TrucoEnv()
    agent = PolicyGradientAgent()
    if reset_model:
        agent.Wp[:] = 0.0
        agent.bp[:] = 0.0
        agent.Wv[:] = 0.0
        agent.bv = 0.0

    hands_done = 0
    done = False
    env.reset()
    hand_steps = []

    while hands_done < hands:
        if done:
            env.reset()
            done = False

        player_id = env.get_current_player()
        action_mask = env.get_action_mask(player_id)
        if not any(action_mask):
            done = True
            if hand_steps:
                _train_on_hand(agent, hand_steps, lr_policy, lr_value, gamma, clip_eps, epochs)
                hand_steps = []
                hands_done += 1
            continue

        obs = env.get_observation(player_id)
        _, value, probs = agent.predict(obs, action_mask)
        action = int(np.random.choice(len(probs), p=probs))
        logp = float(np.log(max(probs[action], 1e-12)))

        prev_cartas = list(env.logic.estado.cartas_jugadas)
        prev_turno_responder_truco = env.logic.estado.turno_responder_truco

        _, reward, done, _, _ = env.step(action, player_id)
        step_reward = reward if player_id == 0 else -reward
        hand_steps.append(
            {
                "obs": obs,
                "mask": action_mask,
                "action": action,
                "logp": logp,
                "value": value,
                "reward": step_reward,
            }
        )

        if _is_hand_end(prev_cartas, prev_turno_responder_truco, action, env) or done:
            _train_on_hand(agent, hand_steps, lr_policy, lr_value, gamma, clip_eps, epochs)
            hand_steps = []
            hands_done += 1

    agent.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrena un agente policy gradient en self-play (por manos)."
    )
    parser.add_argument("--hands", type=int, default=1000, help="Cantidad de manos.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor.")
    parser.add_argument("--lr-policy", type=float, default=1e-3, help="LR politica.")
    parser.add_argument("--lr-value", type=float, default=1e-3, help="LR valor.")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="Clip PPO.")
    parser.add_argument("--epochs", type=int, default=4, help="Epochs por mano.")
    parser.add_argument(
        "--reset-model",
        action="store_true",
        help="Reinicia el modelo antes de entrenar.",
    )
    args = parser.parse_args()

    train(
        args.hands,
        args.gamma,
        args.lr_policy,
        args.lr_value,
        args.clip_eps,
        args.epochs,
        args.reset_model,
    )
