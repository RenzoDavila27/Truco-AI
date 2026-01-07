import gymnasium as gym
import numpy as np
from gymnasium import spaces
from constantes import (
    Acciones,
    ESTADO_ENVIDO,
    ESTADO_ENVIDO_ENVIDO,
    ESTADO_REAL_ENVIDO,
    ESTADO_FALTA_ENVIDO,
)
from truco_logic import TrucoGameLogic


class TrucoEnv(gym.Env):
    """
    Entorno Gymnasium para el Truco Argentino.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(self):
        super(TrucoEnv, self).__init__()

        # ---------------------------------------------------------------------
        # 1. ACTION SPACE (Espacio de Accion)
        # ---------------------------------------------------------------------
        # Definimos 13 acciones posibles segun constantes.py
        self.action_space = spaces.Discrete(len(Acciones))

        # ---------------------------------------------------------------------
        # 2. OBSERVATION SPACE (Espacio de Observacion)
        # ---------------------------------------------------------------------
        # Estructura del Vector (Total: 13 elementos):
        # [0-2]:  Mis Cartas (Ranking 1-14, 0 si no hay carta)
        # [3-5]:  Cartas Oponente en Mesa (Ranking 1-14, 0 si no jugo)
        # [6]:    Mis Puntos (0-30)
        # [7]:    Puntos Oponente (0-30)
        # [8]:    Ronda Actual (1, 2, 3)
        # [9]:    Turno (1=Jugador, 0=Oponente)
        # [10]:   Nivel Truco (0=Nada, 1=Truco, 2=Retruco, 3=Vale4)
        # [11]:   Nivel Envido (0=Nada, 1=Envido, 2=Real, 3=Falta)
        # [12]:   Es Mano (1=Si, 0=No)

        low_bounds = np.array(
            [
                0, 0, 0,  # Mis cartas
                0, 0, 0,  # Cartas rival
                0,        # Mis puntos
                0,        # Puntos rival
                1,        # Ronda
                0,        # Turno
                0,        # Nivel Truco
                0,        # Nivel Envido
                0,        # Es mano
            ],
            dtype=np.float32,
        )

        high_bounds = np.array(
            [
                14, 14, 14,  # Ranking max
                14, 14, 14,
                30,          # Puntos max
                30,
                3,           # Ronda max
                1,           # Turno bool
                4,           # Nivel Truco (Vale 4)
                4,           # Nivel Envido (aprox)
                1,           # Es mano bool
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=low_bounds,
            high=high_bounds,
            dtype=np.float32,
        )

        # Estado interno y motor de reglas
        self.state = None
        self.logic = TrucoGameLogic()

    def reset(self, seed=None, options=None, player_id=None):
        super().reset(seed=seed)
        self.logic.reset_partida()
        if player_id is None:
            player_id = self.get_current_player()
        self.state = self._estado_a_observacion(player_id)

        info = {}
        return self.state, info

    def step(self, action, player_id=None):
        """
        Ejecuta una accion.
        action: int (indice del Enum Acciones)
        """
        # 1. Validar accion (usando Action Masking en el futuro)
        # 2. Comunicar accion al Motor de Reglas
        # 3. Obtener nuevo estado, recompensa y flags del Motor
        if player_id is None:
            player_id = self.get_current_player()
        reward, terminated, _ = self.logic.aplicar_accion(action, player_id)
        self.state = self._estado_a_observacion(player_id)
        truncated = False

        info = {}
        return self.state, reward, terminated, truncated, info

    def render(self, mode="human", player_id=0):
        """
        Renderiza el estado en texto.
        mode: "human" oculta la mano del rival, "debug" muestra todo.
        """
        estado = self.logic.estado
        palos = {0: "Espada", 1: "Basto", 2: "Oro", 3: "Copa"}

        def format_card(carta):
            return f"{carta[0]}-{palos.get(carta[1], carta[1])}"

        def format_hand(mano):
            return [format_card(c) for c in mano]

        def format_played(cartas_jugadas):
            return [f"({format_card(c)}, J{jugador_id})" for c, jugador_id in cartas_jugadas]

        mano_jugador = format_hand(estado.mano_jugador)
        mano_oponente = format_hand(estado.mano_oponente)

        print(f"--- Ronda {estado.numero_ronda} ---")
        if mode == "debug":
            print(f"Mano Jugador 0: {mano_jugador}")
            print(f"Mano Jugador 1: {mano_oponente}")
        else:
            if player_id == 0:
                print(f"Tu mano: {mano_jugador}")
                rival_id = 1
            else:
                print(f"Tu mano: {mano_oponente}")
                rival_id = 0

            mano_rival = ["?"] * 3
            cartas_rival = [c for c, j in estado.cartas_jugadas if j == rival_id]
            for i, carta in enumerate(cartas_rival[:3]):
                mano_rival[i] = format_card(carta)

            print(f"Mano rival: {mano_rival}")

        print(f"Cartas jugadas: {format_played(estado.cartas_jugadas)}")
        print(f"Puntos: J0 {estado.puntos_jugador} pts | J1 {estado.puntos_oponente} pts")
        print(f"Turno actual: {estado.turno_actual}")
        print(f"Truco Lvl: {estado.nivel_truco} | Envido Lvl: {self._nivel_envido_desde_estado(estado)}")

    def close(self):
        pass
    
    def get_current_player(self):
        estado = self.logic.estado
        if estado.turno_responder_envido:
            return 1 - estado.jugador_que_canto_envido
        if estado.turno_responder_truco:
            return 1 - estado.jugador_que_canto_truco
        return estado.turno_actual

    def get_observation(self, player_id=0):
        return self._estado_a_observacion(player_id)

    def _estado_a_observacion(self, player_id=0):
        estado = self.logic.estado
        obs = np.zeros(13, dtype=np.float32)

        # Mis cartas
        mano = estado.mano_jugador if player_id == 0 else estado.mano_oponente
        for i, carta in enumerate(mano[:3]):
            obs[i] = self.logic.obtener_ranking(carta)

        # Cartas del oponente en mesa
        rival_id = 1 - player_id
        cartas_op = [c for c, jugador_id in estado.cartas_jugadas if jugador_id == rival_id]
        for i, carta in enumerate(cartas_op[:3]):
            obs[3 + i] = self.logic.obtener_ranking(carta)

        if player_id == 0:
            obs[6] = float(estado.puntos_jugador)
            obs[7] = float(estado.puntos_oponente)
        else:
            obs[6] = float(estado.puntos_oponente)
            obs[7] = float(estado.puntos_jugador)
        obs[8] = float(estado.numero_ronda)
        obs[9] = 1.0 if self.get_current_player() == player_id else 0.0
        obs[10] = float(estado.nivel_truco)
        obs[11] = float(self._nivel_envido_desde_estado(estado))
        if player_id == 0:
            obs[12] = 1.0 if estado.es_mano else 0.0
        else:
            obs[12] = 1.0 if not estado.es_mano else 0.0

        return obs

    def _nivel_envido_desde_estado(self, estado):
        if estado.estado_canto_envido == ESTADO_FALTA_ENVIDO:
            return 3
        if estado.estado_canto_envido == ESTADO_REAL_ENVIDO:
            return 2
        if estado.estado_canto_envido in [ESTADO_ENVIDO, ESTADO_ENVIDO_ENVIDO]:
            return 1
        return 0

    def get_action_mask(self, player_id=None):
        if player_id is None:
            player_id = self.logic.estado.turno_actual
        return self.logic.get_action_mask(player_id)
