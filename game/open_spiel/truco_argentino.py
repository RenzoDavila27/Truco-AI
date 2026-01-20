import random

import pyspiel

NUM_PLAYERS = 2

RANKS = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
SUITS = ["E", "B", "O", "C"]  # Espadas, Bastos, Oros, Copas
NUM_CARDS = 40

FOLD = 40
QUIERO = 41
ENVIDO = 42
REAL_ENVIDO = 43
FALTA_ENVIDO = 44
TRUCO = 45
RE_TRUCO = 46
VALE_CUATRO = 47

CALL_ACTIONS = [
    FOLD,
    QUIERO,
    ENVIDO,
    REAL_ENVIDO,
    FALTA_ENVIDO,
    TRUCO,
    RE_TRUCO,
    VALE_CUATRO,
]

CALL_ACTION_NAMES = {
    FOLD: "Fold",
    QUIERO: "Quiero",
    ENVIDO: "Envido",
    REAL_ENVIDO: "RealEnvido",
    FALTA_ENVIDO: "FaltaEnvido",
    TRUCO: "Truco",
    RE_TRUCO: "ReTruco",
    VALE_CUATRO: "ValeCuatro",
}

ENVIDO_STATE_NONE = 0
ENVIDO_STATE_ENVIDO = 1
ENVIDO_STATE_REAL = 2
ENVIDO_STATE_FALTA = 3
ENVIDO_STATE_CLOSED = 4

TRUCO_LEVEL_NONE = 0
TRUCO_LEVEL_TRUCO = 1
TRUCO_LEVEL_RETRUCO = 2
TRUCO_LEVEL_VALE_CUATRO = 3

NUM_ACTIONS = VALE_CUATRO + 1
MAX_GAME_LENGTH = 30

CARD_STRENGTH = {
    0: 14,   # 1 Espada
    10: 13,  # 1 Basto 
    6: 12,   # 7 Espada
    26: 11,  # 7 Oro
    # Los 3
    2: 10, 12: 10, 22: 10, 32: 10,
    # Los 2
    1: 9, 11: 9, 21: 9, 31: 9,
    # Los "1s Falsos"
    20: 8, 30: 8,
    # 12s
    9: 7, 19: 7, 29: 7, 39: 7,
    # 11s
    8: 6, 18: 6, 28: 6, 38: 6,
    # 10s
    7: 5, 17: 5, 27: 5, 37: 5,
    # 7s Falsos
    16: 4, 36: 4,
    # 6s
    5: 3, 15: 3, 25: 3, 35: 3,
    # 5s
    4: 2, 14: 2, 24: 2, 34: 2,
    # 4s
    3: 1, 13: 1, 23: 1, 33: 1
}


def _card_id_to_string(card_id):
    suit_index = card_id // 10
    rank_index = card_id % 10
    return f"{RANKS[rank_index]}{SUITS[suit_index]}"


def _card_id_to_rank_suit(card_id):
    suit_index = card_id // 10
    rank_index = card_id % 10
    return RANKS[rank_index], suit_index


def _envido_card_value(rank):
    return rank if rank <= 7 else 0


class TrucoState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self.cards_dealt = []
        self.player_hands = [[] for _ in range(NUM_PLAYERS)]
        self.current_history = []
        
        # Estado del juego
        self._next_player = 0 # Quién tiene el turno
        self._mano_player = 0 # Quién es "mano" (empezó la ronda)
        self._game_over = False
        self._winner = -1 # ID del ganador de la mano
        self._hand_points_override = None
        self._hand_points_awarded = False
        self._score = [0.0, 0.0]
        self._match_points = self._random_match_points()

        # Estado de Truco/Envido
        self._envido_state = ENVIDO_STATE_NONE
        self._pending_envido = False
        self._envido_points = 0
        self._envido_previous_points = 0
        self._envido_caller = None
        self._envido_return_player = None

        self._truco_level = TRUCO_LEVEL_NONE
        self._pending_truco = False
        self._pending_truco_level = TRUCO_LEVEL_NONE
        self._truco_caller = None
        self._truco_return_player = None
        self._truco_last_acceptor = None

        # Lógica de Bazas (Tricks)
        self.current_trick_cards = [] # Tuplas (player_id, card_id)
        self.trick_results = [] # Lista de ganadores de baza: [0, 1, -1] (-1 es parda)
        self.cards_played_count = 0 

    def current_player(self):
        if len(self.cards_dealt) < 6:
            return pyspiel.PlayerId.CHANCE
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        return self._next_player

    def _legal_actions(self, player):
        if player == pyspiel.PlayerId.CHANCE:
            return [c for c in range(NUM_CARDS) if c not in self.cards_dealt]
        
        if self._game_over or player != self.current_player():
            return []

        if self._pending_envido:
            actions = [QUIERO, FOLD]
            if self._envido_state == ENVIDO_STATE_ENVIDO:
                actions.extend([REAL_ENVIDO, FALTA_ENVIDO])
            elif self._envido_state == ENVIDO_STATE_REAL:
                actions.append(FALTA_ENVIDO)
            return actions

        if self._pending_truco:
            actions = [QUIERO, FOLD]
            if self._pending_truco_level == TRUCO_LEVEL_TRUCO:
                actions.append(RE_TRUCO)
            elif self._pending_truco_level == TRUCO_LEVEL_RETRUCO:
                actions.append(VALE_CUATRO)
            if self._can_call_envido(player):
                actions.extend([ENVIDO, REAL_ENVIDO, FALTA_ENVIDO])
            return actions

        # Acciones posibles: Cartas en mano + Gritos
        actions = list(self.player_hands[player])
        actions.append(FOLD)
        
        if self._can_call_envido(player):
            actions.extend([ENVIDO, REAL_ENVIDO, FALTA_ENVIDO])

        if self._can_call_truco(player, TRUCO):
            actions.append(TRUCO)
        if self._can_call_truco(player, RE_TRUCO):
            actions.append(RE_TRUCO)
        if self._can_call_truco(player, VALE_CUATRO):
            actions.append(VALE_CUATRO)
            
        return actions

    def _apply_action(self, action):
        # 1. Fase de Reparto (Chance)
        if self.current_player() == pyspiel.PlayerId.CHANCE:
            self.cards_dealt.append(action)
            target_player = (len(self.cards_dealt) - 1) % NUM_PLAYERS
            self.player_hands[target_player].append(action)
            return

        # 2. Registro de Historia
        self.current_history.append(action)

        # 3. Acciones de Juego
        if action < NUM_CARDS:
            # Jugar Carta
            if not self._pending_envido and not self._pending_truco:
                self._play_card(action)
        else:
            # Gritos (Simplificado para esta etapa)
            if action == FOLD:
                if self._pending_envido:
                    self._resolve_envido(accepted=False, responder=self._next_player)
                elif self._pending_truco:
                    self._resolve_truco(accepted=False, responder=self._next_player)
                else:
                    winner = 1 - self._next_player
                    points = self._hand_points()
                    if (
                        self._next_player == self._mano_player
                        and len(self.current_history) == 1
                    ):
                        points = max(points, 2)
                    self._hand_points_override = points
                    self._award_hand_points(winner, points)
                    self._winner = winner
                    self._game_over = True
            elif action == QUIERO:
                if self._pending_envido:
                    self._resolve_envido(accepted=True, responder=self._next_player)
                elif self._pending_truco:
                    self._resolve_truco(accepted=True, responder=self._next_player)
            elif action in [ENVIDO, REAL_ENVIDO, FALTA_ENVIDO]:
                if self._pending_envido:
                    self._raise_envido(action, self._next_player)
                elif self._can_call_envido(self._next_player):
                    if self._pending_truco:
                        truco_caller = self._truco_caller
                        self._pending_truco = False
                        self._pending_truco_level = TRUCO_LEVEL_NONE
                        self._truco_caller = None
                        self._truco_return_player = None
                        self._start_envido(action, self._next_player, return_player=truco_caller)
                    else:
                        self._start_envido(action, self._next_player)
            elif action in [TRUCO, RE_TRUCO, VALE_CUATRO]:
                if self._pending_truco:
                    self._raise_truco(action, self._next_player)
                elif self._can_call_truco(self._next_player, action):
                    self._start_truco(action, self._next_player)

    def _play_card(self, card):
        player = self._next_player
        
        # Quitar de mano y poner en mesa
        if card in self.player_hands[player]:
            self.player_hands[player].remove(card)
        
        self.current_trick_cards.append((player, card))
        self.cards_played_count += 1

        # Verificar si terminó la baza (2 cartas en mesa)
        if len(self.current_trick_cards) == 2:
            self._resolve_trick()
        else:
            # Turno del otro jugador para completar la baza
            self._next_player = 1 - player

    def _resolve_trick(self):
        """ Determina quién ganó la baza y actualiza el estado """
        p0_play = self.current_trick_cards[0] # (player_id, card)
        p1_play = self.current_trick_cards[1]
        
        # Recuperar fuerza
        s0 = CARD_STRENGTH[p0_play[1]]
        s1 = CARD_STRENGTH[p1_play[1]]

        winner = -1 # Parda por defecto
        
        if s0 > s1:
            winner = p0_play[0]
        elif s1 > s0:
            winner = p1_play[0]
        
        # Guardar resultado
        self.trick_results.append(winner)
        self.current_trick_cards = [] # Limpiar mesa
        
        # Verificar si alguien ganó la MANO completa
        hand_winner = self._check_hand_winner()
        
        if hand_winner is not None:
            self._winner = hand_winner
            self._award_hand_points(hand_winner)
            self._game_over = True
        else:
            # El ganador de la baza empieza la siguiente
            # Si fue parda (-1), empieza el que fue mano en esta baza
            if winner != -1:
                self._next_player = winner
            else:
                # Regla compleja de Truco: Si hay parda, "la mano manda" para el turno
                # Simplificación: Mantiene el turno el que tiró primero en esta baza
                self._next_player = p0_play[0]

    def _check_hand_winner(self):
        """ Reglas oficiales de victoria de Truco """
        results = self.trick_results
        
        # Si aun no jugamos ni 2 bazas, no hay ganador
        if len(results) < 2:
            return None

        # 1. Alguien ganó 2 bazas?
        p0_wins = results.count(0)
        p1_wins = results.count(1)
        
        if p0_wins >= 2: return 0
        if p1_wins >= 2: return 1

        # 2. Definiciones en 3ra baza (si estamos en la 3ra)
        if len(results) == 3:
            # Caso normal: 1ra para uno, 2da para otro, 3ra define
            if results[2] != -1: return results[2]
            # Caso Parda en 3ra: Gana el que ganó la 1ra
            if results[0] != -1: return results[0]
            # Si todo parda (raro), gana la mano
            return self._mano_player

        # 3. Definiciones tempranas (en 2da baza)
        # Si hubo parda en la primera
        if results[0] == -1:
            if results[1] != -1: return results[1] # Ganador de 2da gana todo
            # Si parda en 1ra y 2da -> Definimos en 3ra (continúa juego)
            return None
            
        # Si hubo parda en la segunda (y alguien ganó la primera)
        if results[1] == -1:
            return results[0] # El que ganó primera gana todo

        return None

    def returns(self):
        if not self._game_over:
            return [0.0, 0.0]
        
        # Utilidad Zero-Sum
        base_score = list(self._score)
        if self._hand_points_awarded or self._winner == -1:
            return base_score

        score = float(
            self._hand_points_override
            if self._hand_points_override is not None
            else self._hand_points()
        )
        if self._winner == 0:
            base_score[0] += score
            base_score[1] -= score
        elif self._winner == 1:
            base_score[0] -= score
            base_score[1] += score
        return base_score

    def is_terminal(self):
        return self._game_over

    def chance_outcomes(self):
        # Probabilidad explícita para nodos de azar
        if len(self.cards_dealt) >= 6:
            return []
        remaining_cards = [c for c in range(NUM_CARDS) if c not in self.cards_dealt]
        p = 1.0 / len(remaining_cards)
        return [(c, p) for c in remaining_cards]

    def information_state_string(self, player):
        # Información Privada: Mano
        hand = sorted(self.player_hands[player])
        hand_str = ",".join([_card_id_to_string(c) for c in hand])
        
        # Información Pública: Historia de acciones
        hist_str = []
        for action in self.current_history:
            if action < NUM_CARDS:
                hist_str.append(_card_id_to_string(action))
            else:
                hist_str.append(CALL_ACTION_NAMES.get(action, str(action)))
        
        score_str = f"{self._match_points[0]}-{self._match_points[1]}"
        return f"Score:{score_str}|Hand:{hand_str}|Hist:{'-'.join(hist_str)}"

    def __str__(self):
        return f"Turn:P{self._next_player} Wins:{self.trick_results}"

    def _can_call_envido(self, player):
        if self._pending_envido:
            return False
        if self._envido_state == ENVIDO_STATE_CLOSED:
            return False
        if len(self.trick_results) > 0:
            return False
        if self._truco_level != TRUCO_LEVEL_NONE:
            return False
        if self._pending_truco:
            return (
                self._pending_truco_level == TRUCO_LEVEL_TRUCO
                and player == self._next_player
            )
        return True

    def _can_call_truco(self, player, action):
        if self._pending_truco or self._pending_envido:
            return False
        if action == TRUCO:
            return self._truco_level == TRUCO_LEVEL_NONE
        if action == RE_TRUCO:
            return (
                self._truco_level == TRUCO_LEVEL_TRUCO
                and self._truco_last_acceptor == player
            )
        if action == VALE_CUATRO:
            return (
                self._truco_level == TRUCO_LEVEL_RETRUCO
                and self._truco_last_acceptor == player
            )
        return False

    def _hand_points(self):
        if self._truco_level == TRUCO_LEVEL_NONE:
            return 1
        return self._truco_level + 1

    def _start_envido(self, action, caller, return_player=None):
        self._envido_previous_points = max(1, self._envido_points)
        if action == ENVIDO:
            self._envido_points = 2
            self._envido_state = ENVIDO_STATE_ENVIDO
        elif action == REAL_ENVIDO:
            self._envido_points = 3
            self._envido_state = ENVIDO_STATE_REAL
        elif action == FALTA_ENVIDO:
            self._envido_points = self._falta_envido_points()
            self._envido_state = ENVIDO_STATE_FALTA
        self._pending_envido = True
        self._envido_caller = caller
        if return_player is None:
            return_player = caller
        self._envido_return_player = return_player
        self._next_player = 1 - caller

    def _raise_envido(self, action, caller):
        if action == REAL_ENVIDO and self._envido_state == ENVIDO_STATE_ENVIDO:
            self._envido_previous_points = self._envido_points
            self._envido_points = 5
            self._envido_state = ENVIDO_STATE_REAL
        elif action == FALTA_ENVIDO:
            self._envido_previous_points = self._envido_points
            self._envido_points = self._falta_envido_points()
            self._envido_state = ENVIDO_STATE_FALTA
        else:
            return
        self._envido_caller = caller
        self._next_player = 1 - caller

    def _resolve_envido(self, accepted, responder):
        if not self._pending_envido:
            return
        if accepted:
            puntos_p0 = self._compute_envido_points(self.player_hands[0])
            puntos_p1 = self._compute_envido_points(self.player_hands[1])
            if puntos_p0 > puntos_p1 or (puntos_p0 == puntos_p1 and self._mano_player == 0):
                self._award_points(0, self._envido_points)
            else:
                self._award_points(1, self._envido_points)
        else:
            puntos_rechazo = max(1, self._envido_previous_points)
            self._award_points(self._envido_caller, puntos_rechazo)
        self._pending_envido = False
        self._envido_state = ENVIDO_STATE_CLOSED
        self._next_player = self._envido_return_player
        self._envido_return_player = None

    def _start_truco(self, action, caller):
        if action == TRUCO:
            level = TRUCO_LEVEL_TRUCO
        elif action == RE_TRUCO:
            level = TRUCO_LEVEL_RETRUCO
        else:
            level = TRUCO_LEVEL_VALE_CUATRO
        self._pending_truco = True
        self._pending_truco_level = level
        self._truco_caller = caller
        self._truco_return_player = caller
        self._next_player = 1 - caller

    def _raise_truco(self, action, caller):
        if action == RE_TRUCO and self._pending_truco_level == TRUCO_LEVEL_TRUCO:
            level = TRUCO_LEVEL_RETRUCO
        elif action == VALE_CUATRO and self._pending_truco_level == TRUCO_LEVEL_RETRUCO:
            level = TRUCO_LEVEL_VALE_CUATRO
        else:
            return
        self._pending_truco_level = level
        self._truco_caller = caller
        self._truco_return_player = caller
        self._next_player = 1 - caller

    def _resolve_truco(self, accepted, responder):
        if not self._pending_truco:
            return
        if accepted:
            self._truco_level = self._pending_truco_level
            self._truco_last_acceptor = responder
            self._pending_truco = False
            self._pending_truco_level = TRUCO_LEVEL_NONE
            self._next_player = self._truco_return_player
            self._truco_return_player = None
        else:
            puntos_rechazo = self._truco_reject_points(self._pending_truco_level)
            self._pending_truco = False
            self._pending_truco_level = TRUCO_LEVEL_NONE
            self._hand_points_override = puntos_rechazo
            self._award_hand_points(self._truco_caller, puntos_rechazo)
            self._winner = self._truco_caller
            self._game_over = True

    def _truco_reject_points(self, level):
        if level == TRUCO_LEVEL_VALE_CUATRO:
            return 3
        if level == TRUCO_LEVEL_RETRUCO:
            return 2
        return 1

    def _award_points(self, player, points):
        if player == 0:
            self._score[0] += points
            self._score[1] -= points
            self._match_points[0] = min(30, self._match_points[0] + points)
        else:
            self._score[0] -= points
            self._score[1] += points
            self._match_points[1] = min(30, self._match_points[1] + points)
        if max(self._match_points) >= 30:
            self._game_over = True

    def _award_hand_points(self, winner, points_override=None):
        if self._hand_points_awarded:
            return
        if winner not in (0, 1):
            return
        points = points_override if points_override is not None else self._hand_points()
        self._hand_points_override = points
        self._award_points(winner, points)
        self._hand_points_awarded = True

    def _falta_envido_points(self):
        return max(1, 30 - max(self._match_points))

    def _random_match_points(self):
        return [random.randrange(0, 30), random.randrange(0, 30)]

    def _compute_envido_points(self, hand):
        by_suit = {0: [], 1: [], 2: [], 3: []}
        for card_id in hand:
            rank, suit = _card_id_to_rank_suit(card_id)
            by_suit[suit].append(_envido_card_value(rank))

        max_points = 0
        for values in by_suit.values():
            if len(values) >= 2:
                values.sort(reverse=True)
                points = 20 + values[0] + values[1]
            elif len(values) == 1:
                points = values[0]
            else:
                points = 0
            if points > max_points:
                max_points = points
        return max_points


GAME_TYPE = pyspiel.GameType(
    short_name="truco_argentino",
    long_name="Truco Argentino",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=NUM_PLAYERS,
    min_num_players=NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=False,
    provides_observation_tensor=False,
    parameter_specification={}
)

GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=NUM_ACTIONS,
    max_chance_outcomes=NUM_CARDS,
    num_players=NUM_PLAYERS,
    min_utility=-30.0,
    max_utility=30.0,
    utility_sum=0.0,
    max_game_length=MAX_GAME_LENGTH,
)

class TrucoGame(pyspiel.Game):
    def __init__(self, params=None):
        super().__init__(GAME_TYPE, GAME_INFO, params or {})
    def new_initial_state(self):
        return TrucoState(self)

pyspiel.register_game(GAME_TYPE, TrucoGame)

# --- Main de prueba ---
if __name__ == "__main__":
    game = pyspiel.load_game("truco_argentino")
    state = game.new_initial_state()
    rng = random.Random()
    
    print("Repartiendo cartas...")
    while state.current_player() == pyspiel.PlayerId.CHANCE:
        actions, probs = zip(*state.chance_outcomes())
        action = rng.choices(actions, weights=probs)[0]
        state.apply_action(action)
    
    print(f"Mano P0: {[ _card_id_to_string(c) for c in state.player_hands[0] ]}")
    print(f"Mano P1: {[ _card_id_to_string(c) for c in state.player_hands[1] ]}")
    print(f"Puntos partido: J0 {state._match_points[0]} | J1 {state._match_points[1]}")

    # Simular partida random hasta el final
    while not state.is_terminal():
        current = state.current_player()
        legal = state.legal_actions()
        action = rng.choice(legal)
        
        act_str = _card_id_to_string(action) if action < NUM_CARDS else CALL_ACTION_NAMES[action]
        print(f"Jugador {current} juega: {act_str}")
        
        state.apply_action(action)
    
    print(f"Juego terminado. Ganador: {state._winner}. Returns: {state.returns()}")
    print(f"Puntos finales partido: J0 {state._match_points[0]} | J1 {state._match_points[1]}")
