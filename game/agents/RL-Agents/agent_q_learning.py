import os
import pickle

from constantes import Acciones


class QLearningAgent:
    """
    Agente Q-Learning con encoder de estado discreto.
    """

    def __init__(self, q_table_path=None):
        self.q_table_path = q_table_path or os.path.join(
            os.path.dirname(__file__), "q_tables", "q_table.pkl"
        )
        self.q_table = self._load_q_table()

    def choose_action(self, action_mask, env=None, player_id=0):
        """
        Seleccion greedy usando la Q-table cargada.
        """
        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        if not valid_actions:
            return None
        if env is None:
            return valid_actions[0]

        state = self.encode_state(env, player_id)
        best_action = None
        best_value = None
        for action in valid_actions:
            q_val = self.q_table.get((state, action), 0.0)
            print(f"Action: {action}, Q-value: {q_val}")
            if best_value is None or q_val > best_value:
                best_value = q_val
                best_action = action
        return best_action if best_action is not None else valid_actions[0]

    def encode_state(self, env, player_id=0):
        """
        Convierte el estado actual del entorno a una tupla discreta para Q-table.
        """
        obs = env.get_observation(player_id)
        estado = env.logic.estado

        mano_ranks = self._sorted_hand_ranks(obs[0:3])
        rival_ranks = self._sorted_hand_ranks(obs[3:6])
        puntos_mios = self._bin_points(int(obs[6]))
        puntos_rival = self._bin_points(int(obs[7]))
        ronda = int(obs[8])
        mi_turno = int(obs[9])
        nivel_truco = int(obs[10])
        nivel_envido = int(obs[11])
        es_mano = int(obs[12])

        estado_envido = int(estado.estado_canto_envido)
        estado_truco = int(estado.estado_canto_truco)
        responder_envido = int(estado.turno_responder_envido)
        responder_truco = int(estado.turno_responder_truco)

        rondas_ganadas_mias = (
            estado.rondas_ganadas_jugador if player_id == 0 else estado.rondas_ganadas_oponente
        )
        rondas_ganadas_rival = (
            estado.rondas_ganadas_oponente if player_id == 0 else estado.rondas_ganadas_jugador
        )
        rondas_empatadas = estado.rondas_empatadas

        return (
            mano_ranks,
            rival_ranks,
            puntos_mios,
            puntos_rival,
            ronda,
            mi_turno,
            nivel_truco,
            nivel_envido,
            es_mano,
            estado_envido,
            estado_truco,
            responder_envido,
            responder_truco,
            rondas_ganadas_mias,
            rondas_ganadas_rival,
            rondas_empatadas,
        )

    def _sorted_hand_ranks(self, ranks):
        valid = [int(r) for r in ranks if int(r) > 0]
        valid.sort()
        while len(valid) < 3:
            valid.append(0)
        return tuple(valid[:3])

    def _bin_points(self, puntos):
        if puntos <= 10:
            return 0
        if puntos <= 20:
            return 1
        if puntos <= 29:
            return 2
        return 3

    def _load_q_table(self):
        if not os.path.exists(self.q_table_path):
            return {}
        with open(self.q_table_path, "rb") as f:
            return pickle.load(f)
