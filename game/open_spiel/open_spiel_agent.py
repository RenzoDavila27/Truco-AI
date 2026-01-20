import os
import pickle
import random
import sys

GAME_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OPEN_SPIEL_DIR = os.path.join(GAME_DIR, "open_spiel")
if GAME_DIR not in sys.path:
    sys.path.insert(0, GAME_DIR)
if OPEN_SPIEL_DIR not in sys.path:
    sys.path.insert(0, OPEN_SPIEL_DIR)

import pyspiel

from agents.rational_agent import RationalAgent
from constantes import Acciones

import truco_argentino  # noqa: F401  registers game in pyspiel

RANKS = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
RANK_INDEX = {rank: idx for idx, rank in enumerate(RANKS)}


class OpenSpielPolicyAgent:
    """Agente que usa una policy de OpenSpiel para decidir jugadas de cartas."""

    def __init__(self, policy_path, sample=False, seed=None, fallback_agent=None):
        self.policy_path = policy_path
        self.sample = sample
        self._rng = random.Random(seed)
        self.policy = self._load_policy()
        self.game = pyspiel.load_game("truco_argentino")
        self.fallback_agent = fallback_agent or RationalAgent()

    def _load_policy(self):
        if not os.path.exists(self.policy_path):
            raise FileNotFoundError(f"No se encontró la policy en {self.policy_path}")
        with open(self.policy_path, "rb") as handle:
            return pickle.load(handle)

    def choose_action(self, action_mask, env=None, player_id=0):
        if env is None:
            raise ValueError("env is required to build the OpenSpiel state")

        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        if not valid_actions:
            return None

        card_actions = [
            Acciones.JUGAR_CARTA_1.value,
            Acciones.JUGAR_CARTA_2.value,
            Acciones.JUGAR_CARTA_3.value,
        ]
        if any(action_mask[a] for a in card_actions):
            chosen = self._choose_card_action(action_mask, env, player_id)
            if chosen is not None:
                return chosen

        return self.fallback_agent.choose_action(action_mask, env, player_id)

    def _choose_card_action(self, action_mask, env, player_id):
        state = self._build_open_spiel_state(env, player_id)
        action_probs = self.policy.action_probabilities(state, player_id)

        mano = (
            env.logic.estado.mano_jugador
            if player_id == 0
            else env.logic.estado.mano_oponente
        )
        card_id_to_env_action = {}
        for idx, card in enumerate(mano):
            env_action = Acciones.JUGAR_CARTA_1.value + idx
            if env_action < len(action_mask) and action_mask[env_action]:
                card_id_to_env_action[self._env_card_to_os_id(card)] = env_action

        weighted_actions = {}
        for os_action, prob in action_probs.items():
            if os_action in card_id_to_env_action:
                env_action = card_id_to_env_action[os_action]
                weighted_actions[env_action] = weighted_actions.get(env_action, 0.0) + float(prob)

        if not weighted_actions:
            return None

        if self.sample:
            actions = list(weighted_actions.keys())
            weights = [weighted_actions[a] for a in actions]
            return self._rng.choices(actions, weights=weights, k=1)[0]

        max_prob = max(weighted_actions.values())
        best_actions = [a for a, p in weighted_actions.items() if p == max_prob]
        return self._rng.choice(best_actions)

    def _build_open_spiel_state(self, env, player_id):
        state = self.game.new_initial_state()
        estado = env.logic.estado

        hand_p0 = [self._env_card_to_os_id(card) for card in estado.mano_jugador]
        hand_p1 = [self._env_card_to_os_id(card) for card in estado.mano_oponente]
        played = [self._env_card_to_os_id(card) for card, _ in estado.cartas_jugadas]

        unique_cards = []
        for card_id in hand_p0 + hand_p1 + played:
            if card_id not in unique_cards:
                unique_cards.append(card_id)

        if len(unique_cards) < 6:
            for card_id in range(40):
                if card_id not in unique_cards:
                    unique_cards.append(card_id)
                    if len(unique_cards) == 6:
                        break

        state.cards_dealt = unique_cards[:6]
        state.player_hands = [hand_p0, hand_p1]
        state.current_history = list(played)
        state._next_player = player_id
        state._game_over = False
        state._mano_player = 0 if estado.es_mano else 1
        return state

    def _env_card_to_os_id(self, card):
        rank, suit = card
        if rank not in RANK_INDEX:
            raise ValueError(f"Rank inválido para Truco: {rank}")
        return suit * 10 + RANK_INDEX[rank]
