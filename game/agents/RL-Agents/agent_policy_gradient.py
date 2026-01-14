import os
import pickle
import numpy as np


MODEL_PATH = os.path.join(os.path.dirname(__file__), "pg_models", "policy.pkl")


def _masked_softmax(logits, mask):
    mask = np.asarray(mask, dtype=bool)
    masked_logits = np.where(mask, logits, -1e9)
    max_logits = np.max(masked_logits)
    exp_logits = np.exp(masked_logits - max_logits)
    exp_logits = exp_logits * mask
    denom = np.sum(exp_logits)
    if denom <= 0:
        probs = np.zeros_like(exp_logits)
        valid_idx = np.where(mask)[0]
        if valid_idx.size:
            probs[valid_idx] = 1.0 / valid_idx.size
        return probs
    return exp_logits / denom


class PolicyGradientAgent:
    def __init__(self, model_path=None, num_obs=13, num_actions=13):
        self.model_path = model_path or MODEL_PATH
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.Wp = np.zeros((num_obs, num_actions), dtype=np.float32)
        self.bp = np.zeros((num_actions,), dtype=np.float32)
        self.Wv = np.zeros((num_obs,), dtype=np.float32)
        self.bv = 0.0
        self._load()

    def choose_action(self, action_mask, env=None, player_id=0):
        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        if not valid_actions:
            return None
        if env is None:
            return valid_actions[0]

        obs = env.get_observation(player_id)
        logits = obs @ self.Wp + self.bp
        probs = _masked_softmax(logits, action_mask)
        return int(np.argmax(probs))

    def predict(self, obs, mask):
        logits = obs @ self.Wp + self.bp
        value = float(obs @ self.Wv + self.bv)
        probs = _masked_softmax(logits, mask)
        return logits, value, probs

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        payload = {
            "Wp": self.Wp,
            "bp": self.bp,
            "Wv": self.Wv,
            "bv": self.bv,
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(payload, f)

    def _load(self):
        if not os.path.exists(self.model_path):
            return
        with open(self.model_path, "rb") as f:
            payload = pickle.load(f)
        self.Wp = payload.get("Wp", self.Wp)
        self.bp = payload.get("bp", self.bp)
        self.Wv = payload.get("Wv", self.Wv)
        self.bv = payload.get("bv", self.bv)
