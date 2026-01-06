import gymnasium as gym
import numpy as np
from gymnasium import spaces
from constantes import Acciones, MAZO_DATOS

class TrucoEnv(gym.Env):
    """
    Entorno Gymnasium para el Truco Argentino.
    """
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(self):
        super(TrucoEnv, self).__init__()
        
        # ---------------------------------------------------------------------
        # 1. ACTION SPACE (Espacio de Acción)
        # ---------------------------------------------------------------------
        # Definimos 13 acciones posibles según constantes.py
        self.action_space = spaces.Discrete(len(Acciones))

        # ---------------------------------------------------------------------
        # 2. OBSERVATION SPACE (Espacio de Observación)
        # ---------------------------------------------------------------------
        # Basado en la función 'estado_a_numerico' del readme.md
        # Se agregan 3 slots para las cartas jugadas por el oponente para 
        # mantener la propiedad de Markov.
        
        # Estructura del Vector (Total: 13 elementos):
        # [0-2]:  Mis Cartas (Ranking 1-14, 0 si no hay carta)
        # [3-5]:  Cartas Oponente en Mesa (Ranking 1-14, 0 si no jugó)
        # [6]:    Mis Puntos (0-30)
        # [7]:    Puntos Oponente (0-30)
        # [8]:    Ronda Actual (1, 2, 3)
        # [9]:    Turno (1=Mío, 0=Oponente)
        # [10]:   Nivel Truco (0=Nada, 1=Truco, 2=Retruco, 3=Vale4)
        # [11]:   Nivel Envido (0=Nada, 1=Envido, 2=Real, 3=Falta)
        # [12]:   Es Mano (1=Sí, 0=No)

        LOW_BOUNDS = np.array([
            0, 0, 0,    # Mis cartas
            0, 0, 0,    # Cartas rival
            0,          # Mis puntos
            0,          # Puntos rival
            1,          # Ronda
            0,          # Turno
            0,          # Nivel Truco
            0,          # Nivel Envido
            0           # Es mano
        ], dtype=np.float32)

        HIGH_BOUNDS = np.array([
            14, 14, 14, # Ranking máx (4 de copas)
            14, 14, 14,
            30,         # Puntos máx
            30,
            3,          # Ronda máx
            1,          # Turno bool
            4,          # Nivel Truco (Vale 4)
            4,          # Nivel Envido (aprox)
            1           # Es mano bool
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=LOW_BOUNDS, 
            high=HIGH_BOUNDS, 
            dtype=np.float32
        )

        # Estado interno (Placeholder hasta conectar con el Motor de Juego)
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # AQUI IRÍA LA INICIALIZACIÓN DEL MOTOR DE JUEGO (REPARTIR CARTAS)
        # Por ahora devolvemos un estado vacío válido para probar el shape
        
        self.state = np.zeros(13, dtype=np.float32)
        
        # Ejemplo: Soy mano (idx 12), ronda 1 (idx 8)
        self.state[12] = 1.0 
        self.state[8] = 1.0 
        
        # Ejemplo: Me tocaron el 1 de Espada (Ranking 1) y dos 4s (Ranking 14)
        self.state[0] = 1.0
        self.state[1] = 14.0
        self.state[2] = 14.0

        info = {}
        return self.state, info

    def step(self, action):
        """
        Ejecuta una acción.
        action: int (índice del Enum Acciones)
        """
        # 1. Validar acción (usando Action Masking en el futuro)
        # 2. Comunicar acción al Motor de Reglas
        # 3. Obtener nuevo estado, recompensa y flags del Motor
        
        # MOCK DE RESPUESTA (Solo para verificar estructura)
        reward = 0.0
        terminated = False
        truncated = False
        
        # Simular cambio de turno
        self.state[9] = 0.0 # Ahora es turno del oponente

        info = {}
        return self.state, reward, terminated, truncated, info

    def render(self):
        # Muestra simple del estado en texto
        print(f"--- Ronda {self.state[8]} ---")
        print(f"Mis Cartas (Rank): {self.state[0:3]}")
        print(f"Puntos: {self.state[6]} vs {self.state[7]}")
        print(f"Truco Lvl: {self.state[10]} | Envido Lvl: {self.state[11]}")
    
    def close(self):
        pass