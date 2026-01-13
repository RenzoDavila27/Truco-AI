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
            #print(f"Action: {action}, Q-value: {q_val}")
            if best_value is None or q_val > best_value:
                best_value = q_val
                best_action = action
        return best_action if best_action is not None else valid_actions[0]

    def encode_state(self, env, player_id=0):
        """
        Versión optimizada para reducir la Explosión de Estados.
        Se enfoca en información relativa y 'Zonas' de juego.
        """
        obs = env.get_observation(player_id)
        estado = env.logic.estado

        # 1. MIS CARTAS: Mantenemos el ranking ordenado (Fundamental)
        mano_ranks = self._sorted_hand_ranks(obs[0:3])

        # 2. RIVAL: Solo nos importa la carta más fuerte que tenga en mesa para matarla
        # obs[3:6] son las cartas del rival en mesa
        rival_ranks_list = [int(r) for r in obs[3:6] if int(r) > 0]
        max_rival_mesa = max(rival_ranks_list) if rival_ranks_list else 0

        # 3. PUNTOS: Usamos ZONAS en lugar de bins lineales
        if player_id == 0:
            mis_puntos = int(obs[6])
            rival_puntos = int(obs[7])
        else:
            mis_puntos = int(obs[7])
            rival_puntos = int(obs[6])

        # Zona 0: Malas (0-15), Zona 1: Buenas (16-25), Zona 2: Definición (26-30)
        def get_zona(p):
            if p <= 15: return 0
            if p <= 25: return 1
            return 2

        mi_zona = get_zona(mis_puntos)
        rival_zona = get_zona(rival_puntos)
        voy_ganando = 1 if mis_puntos >= rival_puntos else 0

        # 4. CONTEXTO DE RONDA
        # Saber si soy mano es vital para el Envido y para definir en 3ra ronda
        soy_mano = int(obs[12]) 
        # Nivel de presión del truco (0, 1, 2, 3)
        nivel_truco = int(obs[10])
        # Estado del envido (No cantado, Envido, Real, Falta, Cerrado)
        estado_envido = int(estado.estado_canto_envido)
        
        # Ronda actual (1, 2, 3)
        ronda = int(obs[8])

        # Retornamos una tupla compacta de 8 elementos
        return (
            mano_ranks,         # Tupla (R1, R2, R3)
            max_rival_mesa,     # Int (0-14)
            mi_zona,            # Int (0-2)
            rival_zona,         # Int (0-2)
            voy_ganando,        # Bool (0-1)
            nivel_truco,        # Int (0-4)
            estado_envido,      # Int
            soy_mano,           # Bool (0-1)
            # Opcional: ronda. A veces es útil, a veces redundante con las cartas jugadas.
            ronda               # Int (1-3)
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
