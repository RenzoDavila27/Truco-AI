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

        state = self._build_open_spiel_state(env, player_id)
        action_probs = self.policy.action_probabilities(state, player_id)

        weighted_actions = self._project_policy_to_env(
            action_probs, action_mask, env, player_id
        )
        if not weighted_actions:
            return self.fallback_agent.choose_action(action_mask, env, player_id)

        return self._select_weighted_action(weighted_actions)

    def _project_policy_to_env(self, action_probs, action_mask, env, player_id):
        weighted_actions = {}
        card_mapping = self._card_id_to_env_action(action_mask, env, player_id)

        for os_action, prob in action_probs.items():
            env_action = None

            if os_action < truco_argentino.NUM_CARDS:
                env_action = card_mapping.get(os_action)
            else:
                env_action = self._map_call_action(os_action, action_mask, env)

            if env_action is None:
                continue

            weighted_actions[env_action] = weighted_actions.get(env_action, 0.0) + float(prob)

        return weighted_actions

    def _card_id_to_env_action(self, action_mask, env, player_id):
        mano = (
            env.logic.estado.mano_jugador
            if player_id == 0
            else env.logic.estado.mano_oponente
        )
        card_id_to_env_action = {}
        for idx, card in enumerate(mano):
            env_action = Acciones.JUGAR_CARTA_1.value + idx
            if env_action < len(action_mask) and action_mask[env_action]:
                card_id = self._env_card_to_os_id(card)
                card_id_to_env_action[card_id] = env_action

        return card_id_to_env_action

    def _map_call_action(self, os_action, action_mask, env):
        estado = env.logic.estado
        candidate = None

        if os_action == truco_argentino.FOLD:
            if estado.turno_responder_envido or estado.turno_responder_truco:
                candidate = Acciones.NO_QUIERO.value
            else:
                candidate = Acciones.IR_AL_MAZO.value
        elif os_action == truco_argentino.QUIERO:
            candidate = Acciones.QUIERO.value
        elif os_action == truco_argentino.ENVIDO:
            candidate = Acciones.ENVIDO.value
        elif os_action == truco_argentino.REAL_ENVIDO:
            candidate = Acciones.REAL_ENVIDO.value
        elif os_action == truco_argentino.FALTA_ENVIDO:
            candidate = Acciones.FALTA_ENVIDO.value
        elif os_action == truco_argentino.TRUCO:
            candidate = Acciones.TRUCO.value
        elif os_action == truco_argentino.RE_TRUCO:
            candidate = Acciones.RETRUCO.value
        elif os_action == truco_argentino.VALE_CUATRO:
            candidate = Acciones.VALE_CUATRO.value

        if candidate is None:
            return None
        if candidate >= len(action_mask) or not action_mask[candidate]:
            return None
        return candidate

    def _select_weighted_action(self, weighted_actions):
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
