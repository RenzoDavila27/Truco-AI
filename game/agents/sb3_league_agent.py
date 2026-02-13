import os
import sys
from typing import Optional

GAME_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if GAME_DIR not in sys.path:
    sys.path.insert(0, GAME_DIR)

from sb3.sb3_agent import SB3Agent


def _find_latest_model(models_dir: str) -> Optional[str]:
    if not os.path.isdir(models_dir):
        return None
    candidates = []
    for root, _, files in os.walk(models_dir):
        for name in files:
            if name.endswith(".zip"):
                path = os.path.join(root, name)
                candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _find_preferred_model(models_dir: str) -> Optional[str]:
    preferred = os.path.join(models_dir, "ppo_truco_league.zip")
    if os.path.isfile(preferred):
        return preferred
    return _find_latest_model(models_dir)


class SB3LeagueAgent:
    """
    Agent that loads the latest SB3 league policy from game/sb3/models.
    Override with SB3_TRUCO_LEAGUE_MODEL env var to force a specific model.
    """

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = os.getenv("SB3_TRUCO_LEAGUE_MODEL")
        if model_path is None:
            models_dir = os.path.abspath(os.path.join(GAME_DIR, "sb3", "models"))
            model_path = _find_preferred_model(models_dir)
        if model_path is None:
            raise FileNotFoundError("No se encontro ningun modelo en game/sb3/models.")
        self._agent = SB3Agent(model_path)

    def choose_action(self, action_mask, env=None, player_id=0):
        return self._agent.choose_action(action_mask, env, player_id)
