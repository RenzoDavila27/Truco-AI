import random
import numpy as np
from constantes import MAZO_DATOS, ESPADA, BASTO, ORO, COPA, Acciones

class EstadoTruco:
    """
    Clase que define el estado del juego de Truco.
    """
    def __init__(self):
        self.mano_jugador = []      # Lista de tuplas (num, palo)
        self.mano_oponente = []     # Lista de tuplas
        self.cartas_jugadas = []    # Lista de (carta, jugador_id)
        
        self.puntos_jugador = 0
        self.puntos_oponente = 0
        
        self.numero_ronda = 1       # 1, 2, 3 (Primera, Segunda, Tercera)
        self.turno_actual = 0       # 0: Jugador, 1: Oponente
        
        # 0: No cantado, 1: Truco, 2: Retruco, 3: Vale 4
        self.nivel_truco = 0        
        
        # 0: No, 1: Envido, 2: Real, 3: Falta
        self.nivel_envido = 0       
        
        self.es_mano = True         # Bool
        self.envido_cantado_fase = False # Flag auxiliar para saber si ya pasó el momento

class TrucoGameLogic:

    def __init__(self):
        self.estado = EstadoTruco()
        # Generar el mazo base desde constantes (keys del dict)
        self.mazo_base = list(MAZO_DATOS.keys())

    def reset_partida(self):
        """Reinicia los puntos a 0."""
        self.estado.puntos_jugador = 0
        self.estado.puntos_oponente = 0
        self.nueva_mano()

    def nueva_mano(self):
        """
        Reparte cartas y reinicia variables de la ronda.
        Simula la distribución hipergeométrica
        """
        random.shuffle(self.mazo_base)
        
        # Repartir 3 a cada uno
        self.estado.mano_jugador = self.mazo_base[:3]
        self.estado.mano_oponente = self.mazo_base[3:6]
        
        # Resetear flags de ronda
        self.estado.cartas_jugadas = [] # Limpiar mesa
        self.estado.numero_ronda = 1
        self.estado.nivel_truco = 0
        self.estado.nivel_envido = 0
        self.estado.envido_cantado_fase = False
        
        # Alternar mano (simple toggle para este ejemplo)
        self.estado.es_mano = not self.estado.es_mano
        # Si soy mano, turno = 0 (mío), sino 1 (oponente)
        self.estado.turno_actual = 0 if self.estado.es_mano else 1

    def calcular_puntos_envido(self, mano):
        """
        Calcula el tanto de una mano.
        Regla: 2 cartas del mismo palo suman 20 + sus valores.
        """
        # Agrupar por palo
        cartas_por_palo = {ESPADA: [], BASTO: [], ORO: [], COPA: []}
        
        for carta in mano:
            palo = carta[1]
            valor = MAZO_DATOS[carta]["valor_envido"]
            cartas_por_palo[palo].append(valor)

        max_puntos = 0
        
        # Buscar la mejor combinación
        for palo, valores in cartas_por_palo.items():
            if len(valores) >= 2:
                # Tomar los 2 más altos si hubiera 3 del mismo palo
                valores.sort(reverse=True)
                puntos = 20 + valores[0] + valores[1]
            elif len(valores) == 1:
                puntos = valores[0]
            else:
                puntos = 0
            
            if puntos > max_puntos:
                max_puntos = puntos
                
        return max_puntos

    def obtener_ranking(self, carta):
        """Retorna el ranking (1-14) de la carta."""
        return MAZO_DATOS[carta]["ranking"]

    def determinar_ganador_mano(self, carta_j, carta_op):
        """
        Compara dos cartas. 
        Retorna: 0 si gana Jugador, 1 si gana Oponente, 2 si es Empate (Parda).
        Nota: En el readme, menor ranking es mejor carta (1 gana a 14).
        """
        rank_j = self.obtener_ranking(carta_j)
        rank_op = self.obtener_ranking(carta_op)

        if rank_j < rank_op:
            return 0 # Gana Jugador
        elif rank_op < rank_j:
            return 1 # Gana Oponente
        else:
            return 2 # Empate

    def aplicar_accion(self, accion_idx):
        """
        Procesa la acción recibida desde el entorno.
        Retorna: (reward, terminado, info)
        """
        reward = 0
        terminado = False
        info = {}

        # Mapeo inverso para legibilidad
        accion = Acciones(accion_idx)

        # ---------------------------------------------------------
        # LÓGICA DE JUGAR CARTAS (Nivel Operativo: check)
        # ---------------------------------------------------------
        if accion in [Acciones.JUGAR_CARTA_1, Acciones.JUGAR_CARTA_2, Acciones.JUGAR_CARTA_3]:
            idx = accion.value # 0, 1 o 2
            
            # Verificar si la carta existe (no se jugó ya)
            if idx < len(self.estado.mano_jugador):
                carta_jugada = self.estado.mano_jugador.pop(idx)
                
                # Agregar a mesa
                self.estado.cartas_jugadas.append((carta_jugada, 0)) # 0 es ID jugador
                
                # CAMBIO DE TURNO (Simplificado)
                self.estado.turno_actual = 1 
                
                # AQUÍ DEBERÍA JUGAR EL OPONENTE (Simulado)
                # En un entorno real, el step() llamaría al agente oponente.
                # Por ahora, hacemos que el oponente juegue random inmediatamente
                if self.estado.mano_oponente:
                    carta_op = self.estado.mano_oponente.pop(0)
                    self.estado.cartas_jugadas.append((carta_op, 1))
                    
                    # Evaluar quién ganó la vuelta
                    ganador = self.determinar_ganador_mano(carta_jugada, carta_op)
                    if ganador == 0:
                        reward = 0.5 # Pequeña recompensa por ganar mano
                    
                    self.estado.turno_actual = 0 # Vuelve a mí
                    self.estado.numero_ronda += 1
            else:
                # Castigo por intentar jugar carta inexistente
                reward = -5 

        # ---------------------------------------------------------
        # LÓGICA DE CANTOS (Envido / Truco)
        # ---------------------------------------------------------
        elif accion == Acciones.ENVIDO:
            if not self.estado.envido_cantado_fase and self.estado.numero_ronda == 1:
                self.estado.nivel_envido = 1
                self.estado.envido_cantado_fase = True
                # Aquí iría la lógica de respuesta del oponente
                # Simulación: Oponente "Quiero"
                puntos_mios = self.calcular_puntos_envido(self.estado.mano_jugador + [c[0] for c in self.estado.cartas_jugadas if c[1]==0])
                puntos_op = self.calcular_puntos_envido(self.estado.mano_oponente + [c[0] for c in self.estado.cartas_jugadas if c[1]==1])
                
                if puntos_mios > puntos_op:
                    self.estado.puntos_jugador += 2
                    reward = 2
                else:
                    self.estado.puntos_oponente += 2
                    reward = -1

        elif accion == Acciones.TRUCO:
            if self.estado.nivel_truco == 0:
                self.estado.nivel_truco = 1
                # Simulación: Oponente NO QUIERE
                self.estado.puntos_jugador += 1
                reward = 1
                terminado = True # Se termina la mano si no quiere

        elif accion == Acciones.IR_AL_MAZO:
            self.estado.puntos_oponente += 1
            terminado = True
            reward = -1

        # Verificar fin de partida (15 o 30 puntos)
        if self.estado.puntos_jugador >= 30:
            reward += 100
            terminado = True
        elif self.estado.puntos_oponente >= 30:
            reward -= 100
            terminado = True

        return reward, terminado, info

    def get_action_mask(self):
        """
        Devuelve una lista de booleanos indicando qué acciones son válidas
        en el estado actual.
        Orden del array corresponde a los índices de Acciones(Enum).
        """
        # Inicializamos todo en False (nada permitido por defecto)
        mask = [False] * len(Acciones)
        
        # ----------------------------------------------------
        # 1. MÁSCARA PARA JUGAR CARTAS
        # ----------------------------------------------------
        # Solo se pueden jugar cartas si NO hay un desafío pendiente de responder
        # (Si me cantaron Truco, primero debo responder, no puedo tirar carta)
        hay_desafio_pendiente = (self.estado.turno_responder_envido or 
                                self.estado.turno_responder_truco)
        
        if not hay_desafio_pendiente:
            if 0 < len(self.estado.mano_jugador): mask[Acciones.JUGAR_CARTA_1.value] = True
            if 1 < len(self.estado.mano_jugador): mask[Acciones.JUGAR_CARTA_2.value] = True
            if 2 < len(self.estado.mano_jugador): mask[Acciones.JUGAR_CARTA_3.value] = True
            
            # También se puede cantar TRUCO si no hay desafío de envido activo
            # Y si no se cantó ya truco o si tengo el quiero para retrucar
            if self.validar_canto_truco():
                mask[Acciones.TRUCO.value] = True
                # (Agregar lógica similar para Retruco/Vale4 según corresponda)

        # ----------------------------------------------------
        # 2. MÁSCARA PARA EL ENVIDO (Tu pregunta específica)
        # ----------------------------------------------------
        # El envido solo se canta en la primera ronda y antes de jugar cartas (generalmente)
        if self.estado.numero_ronda == 1 and not self.estado.envido_finalizado:
            
            estado_actual = self.estado.estado_canto_envido

            # CASO A: Nadie cantó nada aún (Piso 0)
            # "se puede cantar envido, real o falta, sin haber cantado el anterior"
            if estado_actual == ESTADO_NO_CANTADO:
                mask[Acciones.ENVIDO.value] = True
                mask[Acciones.REAL_ENVIDO.value] = True
                mask[Acciones.FALTA_ENVIDO.value] = True
                # Envido-Envido PROHIBIDO aquí
            
            # CASO B: Me cantaron "Envido" (Escalón 1)
            # "envido-envido, que unicamente puede cantarse luego del envido simple"
            elif estado_actual == ESTADO_ENVIDO:
                mask[Acciones.ENVIDO_ENVIDO.value] = True # Permitido
                mask[Acciones.REAL_ENVIDO.value] = True   # Permitido elevar
                mask[Acciones.FALTA_ENVIDO.value] = True  # Permitido elevar
                mask[Acciones.QUIERO.value] = True
                mask[Acciones.NO_QUIERO.value] = True
                # Envido simple PROHIBIDO (sería repetir)

            # CASO C: Me cantaron "Envido Envido" (Escalón 2)
            elif estado_actual == ESTADO_ENVIDO_ENVIDO:
                mask[Acciones.REAL_ENVIDO.value] = True   # Permitido
                mask[Acciones.FALTA_ENVIDO.value] = True  # Permitido
                mask[Acciones.QUIERO.value] = True
                mask[Acciones.NO_QUIERO.value] = True
                # Envido y Envido-Envido PROHIBIDOS

            # CASO D: Me cantaron "Real Envido" (Directo o escalado)
            elif estado_actual == ESTADO_REAL_ENVIDO:
                mask[Acciones.FALTA_ENVIDO.value] = True  # Única subida posible
                mask[Acciones.QUIERO.value] = True
                mask[Acciones.NO_QUIERO.value] = True
                # Prohibido volver a Envido o Real Envido
                
            # CASO E: Falta Envido
            elif estado_actual == ESTADO_FALTA_ENVIDO:
                mask[Acciones.QUIERO.value] = True
                mask[Acciones.NO_QUIERO.value] = True
                # No se puede subir más

        return mask