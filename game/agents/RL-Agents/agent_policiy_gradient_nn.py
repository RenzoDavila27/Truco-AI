import os
import torch
import torch.nn as nn


MODEL_PATH = os.path.join(os.path.dirname(__file__), "pg_models", "policy_nn.pt")


class PolicyGradientNN(nn.Module):
    def __init__(self, num_obs=13, num_actions=13, hidden_size=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_obs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(num_obs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs):
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return logits, value


class PolicyGradientNNAgent:
    def __init__(self, model_path=None, num_obs=13, num_actions=13, hidden_size=128):
        self.model_path = model_path or MODEL_PATH
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.device = torch.device("cpu")
        self.model = PolicyGradientNN(num_obs, num_actions, hidden_size).to(self.device)
        self._load()

    def choose_action(self, action_mask, env=None, player_id=0):
        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        if not valid_actions:
            return None
        if env is None:
            return valid_actions[0]

        obs = torch.tensor(env.get_observation(player_id), dtype=torch.float32, device=self.device)
        mask = torch.tensor(action_mask, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits, _ = self.model(obs)
            masked_logits = torch.where(mask, logits, torch.tensor(-1e9, device=self.device))
            probs = torch.softmax(masked_logits, dim=-1)
            action = int(torch.argmax(probs).item())
        return action

    def predict(self, obs, mask):
        logits, value = self.model(obs)
        masked_logits = torch.where(mask, logits, torch.tensor(-1e9, device=logits.device))
        probs = torch.softmax(masked_logits, dim=-1)
        return logits, value, probs

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)

    def _load(self):
        if not os.path.exists(self.model_path):
            return
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
