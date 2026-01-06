from enum import Enum

# =============================================================================
# DEFINICIÓN DE PALOS Y VALORES
# =============================================================================
ESPADA = 0
BASTO = 1
ORO = 2
COPA = 3

# =============================================================================
# RANKING DE CARTAS
# Ranking 1 (Más fuerte) a 14 (Más débil).
# Valor Envido: El valor literal para sumar en el envido (ej: el 7 vale 7, el 10 vale 0).
# =============================================================================

MAZO_DATOS = {
    # Los 1
    (1, ESPADA): {"ranking": 1, "valor_envido": 1}, 
    (1, BASTO):  {"ranking": 2, "valor_envido": 1},  
    (1, ORO):    {"ranking": 7, "valor_envido": 1},
    (1, COPA):   {"ranking": 7, "valor_envido": 1},
    
    # Los 2
    (2, ESPADA): {"ranking": 6, "valor_envido": 2},
    (2, BASTO):  {"ranking": 6, "valor_envido": 2},
    (2, ORO):    {"ranking": 6, "valor_envido": 2},
    (2, COPA):   {"ranking": 6, "valor_envido": 2},

    # Los 3
    (3, ESPADA): {"ranking": 5, "valor_envido": 3},
    (3, BASTO):  {"ranking": 5, "valor_envido": 3},
    (3, ORO):    {"ranking": 5, "valor_envido": 3},
    (3, COPA):   {"ranking": 5, "valor_envido": 3},

    # Los 4
    (4, ESPADA): {"ranking": 14, "valor_envido": 4},
    (4, BASTO):  {"ranking": 14, "valor_envido": 4},
    (4, ORO):    {"ranking": 14, "valor_envido": 4},
    (4, COPA):   {"ranking": 14, "valor_envido": 4},

    # Los 5
    (5, ESPADA): {"ranking": 13, "valor_envido": 5},
    (5, BASTO):  {"ranking": 13, "valor_envido": 5},
    (5, ORO):    {"ranking": 13, "valor_envido": 5},
    (5, COPA):   {"ranking": 13, "valor_envido": 5},

    # Los 6
    (6, ESPADA): {"ranking": 12, "valor_envido": 6},
    (6, BASTO):  {"ranking": 12, "valor_envido": 6},
    (6, ORO):    {"ranking": 12, "valor_envido": 6},
    (6, COPA):   {"ranking": 12, "valor_envido": 6},

    # Los 7
    (7, ESPADA): {"ranking": 3, "valor_envido": 7},
    (7, ORO):    {"ranking": 4, "valor_envido": 7},
    (7, BASTO):  {"ranking": 11, "valor_envido": 7},
    (7, COPA):   {"ranking": 11, "valor_envido": 7},

    # Las Figuras (10, 11, 12)
    (10, ESPADA): {"ranking": 10, "valor_envido": 0},
    (10, BASTO):  {"ranking": 10, "valor_envido": 0},
    (10, ORO):    {"ranking": 10, "valor_envido": 0},
    (10, COPA):   {"ranking": 10, "valor_envido": 0},

    (11, ESPADA): {"ranking": 9, "valor_envido": 0},
    (11, BASTO):  {"ranking": 9, "valor_envido": 0},
    (11, ORO):    {"ranking": 9, "valor_envido": 0},
    (11, COPA):   {"ranking": 9, "valor_envido": 0},

    (12, ESPADA): {"ranking": 8, "valor_envido": 0},
    (12, BASTO):  {"ranking": 8, "valor_envido": 0},
    (12, ORO):    {"ranking": 8, "valor_envido": 0},
    (12, COPA):   {"ranking": 8, "valor_envido": 0},
}

# =============================================================================
# ESTADOS DEL ENVIDO
# =============================================================================
ESTADO_NO_CANTADO = 0
ESTADO_ENVIDO = 1
ESTADO_ENVIDO_ENVIDO = 2
ESTADO_REAL_ENVIDO = 3
ESTADO_FALTA_ENVIDO = 4
ESTADO_CERRADO = 5

# =============================================================================
# ACCIONES (ACTION SPACE)
# =============================================================================
class Acciones(Enum):
    # Nivel Operativo: check / jugar carta
    JUGAR_CARTA_1 = 0
    JUGAR_CARTA_2 = 1
    JUGAR_CARTA_3 = 2
    
    # Nivel Operativo: bet / raise (Envido)
    ENVIDO = 3
    ENVIDO_ENVIDO = 4
    REAL_ENVIDO = 5
    FALTA_ENVIDO = 6
     
    
    # Nivel Operativo: bet / raise (Truco)
    TRUCO = 7
    RETRUCO = 8
    VALE_CUATRO = 9
    
    # Nivel Operativo: call / pass / fold
    QUIERO = 10
    NO_QUIERO = 11      # Incluye fold (truco) y pass (envido)
    IR_AL_MAZO = 12     # Fold fuera de turno de respuesta