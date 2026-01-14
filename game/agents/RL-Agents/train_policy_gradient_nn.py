import argparse
import os
import sys
import torch

GAME_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if GAME_DIR not in sys.path:
    sys.path.insert(0, GAME_DIR)

from constantes import Acciones
from truco_env import TrucoEnv
from agent_policiy_gradient_nn import PolicyGradientNNAgent


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
    return torch.tensor(returns, dtype=torch.float32)


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
    agent = PolicyGradientNNAgent()
    model = agent.model
    device = agent.device

    if reset_model:
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    optimizer_policy = torch.optim.Adam(model.actor.parameters(), lr=lr_policy)
    optimizer_value = torch.optim.Adam(model.critic.parameters(), lr=lr_value)

    hands_done = 0
    done = False
    env.reset()
    hand_steps = []

    try:
        while hands_done < hands:
            if done:
                env.reset()
                done = False

            player_id = env.get_current_player()
            action_mask = env.get_action_mask(player_id)
            if not any(action_mask):
                done = True
                if hand_steps:
                    _train_on_hand(
                        hand_steps,
                        model,
                        optimizer_policy,
                        optimizer_value,
                        gamma,
                        clip_eps,
                        epochs,
                        device,
                    )
                    hand_steps = []
                    hands_done += 1
                continue

            obs = torch.tensor(env.get_observation(player_id), dtype=torch.float32, device=device)
            mask = torch.tensor(action_mask, dtype=torch.bool, device=device)
            with torch.no_grad():
                _, value, probs = agent.predict(obs, mask)
                action_dist = torch.distributions.Categorical(probs)
                action = int(action_dist.sample().item())
                logp = float(action_dist.log_prob(torch.tensor(action, device=device)).item())

            prev_cartas = list(env.logic.estado.cartas_jugadas)
            prev_turno_responder_truco = env.logic.estado.turno_responder_truco

            _, reward, done, _, _ = env.step(action, player_id)
            step_reward = reward if player_id == 0 else -reward
            hand_steps.append(
                {
                    "obs": obs.detach(),
                    "mask": mask.detach(),
                    "action": action,
                    "logp": logp,
                    "value": float(value.item()),
                    "reward": step_reward,
                }
            )

            if _is_hand_end(prev_cartas, prev_turno_responder_truco, action, env) or done:
                _train_on_hand(
                    hand_steps,
                    model,
                    optimizer_policy,
                    optimizer_value,
                    gamma,
                    clip_eps,
                    epochs,
                    device,
                )
                hand_steps = []
                hands_done += 1
    finally:
        agent.save()


def _train_on_hand(
    hand_steps,
    model,
    optimizer_policy,
    optimizer_value,
    gamma,
    clip_eps,
    epochs,
    device,
):
    obs_batch = torch.stack([s["obs"] for s in hand_steps]).to(device)
    mask_batch = torch.stack([s["mask"] for s in hand_steps]).to(device)
    actions = torch.tensor([s["action"] for s in hand_steps], dtype=torch.long, device=device)
    old_logp = torch.tensor([s["logp"] for s in hand_steps], dtype=torch.float32, device=device)
    values = torch.tensor([s["value"] for s in hand_steps], dtype=torch.float32, device=device)
    rewards = torch.tensor([s["reward"] for s in hand_steps], dtype=torch.float32, device=device)

    returns = _compute_returns(rewards, gamma).to(device)
    advantages = returns - values
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(epochs):
        logits, new_values = model(obs_batch)
        masked_logits = torch.where(mask_batch, logits, torch.tensor(-1e9, device=device))
        probs = torch.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_logp = dist.log_prob(actions)

        ratio = torch.exp(new_logp - old_logp)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        policy_loss = -torch.mean(torch.min(ratio * advantages, clipped_ratio * advantages))

        value_loss = torch.mean((new_values - returns) ** 2)

        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrena un policy gradient con red neuronal (self-play por manos)."
    )
    parser.add_argument("--hands", type=int, default=1000, help="Cantidad de manos.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor.")
    parser.add_argument("--lr-policy", type=float, default=3e-4, help="LR politica.")
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
