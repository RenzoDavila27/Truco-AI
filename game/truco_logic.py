import random
from constantes import (
    MAZO_DATOS,
    ESPADA,
    BASTO,
    ORO,
    COPA,
    Acciones,
    ESTADO_NO_CANTADO,
    ESTADO_ENVIDO,
    ESTADO_ENVIDO_ENVIDO,
    ESTADO_REAL_ENVIDO,
    ESTADO_FALTA_ENVIDO,
    ESTADO_CERRADO,
    ESTADO_TRUCO_NO_CANTADO,
    ESTADO_TRUCO,
    ESTADO_RETRUCO,
    ESTADO_VALE_CUATRO,
    ESTADO_TRUCO_CERRADO,
)

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
        
        self.es_mano = True         # Bool
        self.envido_finalizado = False
        self.estado_canto_envido = ESTADO_NO_CANTADO
        self.envido_total = 0
        self.envido_total_anterior = 0
        self.turno_responder_envido = False
        self.turno_responder_truco = False
        self.estado_canto_truco = ESTADO_TRUCO_NO_CANTADO
        self.jugador_que_canto_envido = None
        self.jugador_que_canto_truco = None
        self.jugador_que_acepto_truco = None
        self.turno_post_envido = None

        self.rondas_ganadas_jugador = 0
        self.rondas_ganadas_oponente = 0
        self.rondas_empatadas = 0
        self.resultados_ronda = []
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
        self.estado.envido_finalizado = False
        self.estado.estado_canto_envido = ESTADO_NO_CANTADO
        self.estado.envido_total = 0
        self.estado.envido_total_anterior = 0
        self.estado.turno_responder_envido = False
        self.estado.turno_responder_truco = False
        self.estado.estado_canto_truco = ESTADO_TRUCO_NO_CANTADO
        self.estado.jugador_que_canto_envido = None
        self.estado.jugador_que_canto_truco = None
        self.estado.jugador_que_acepto_truco = None
        self.estado.turno_post_envido = None
        
        self.estado.rondas_ganadas_jugador = 0
        self.estado.rondas_ganadas_oponente = 0
        self.estado.rondas_empatadas = 0
        self.estado.resultados_ronda = []

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

    def _registrar_resultado_ronda(self, ganador):
        self.estado.resultados_ronda.append(ganador)
        if ganador == 0:
            self.estado.rondas_ganadas_jugador += 1
        elif ganador == 1:
            self.estado.rondas_ganadas_oponente += 1
        else:
            self.estado.rondas_empatadas += 1

    def _ganador_mano_completa(self):
        ganadas_j = self.estado.rondas_ganadas_jugador
        ganadas_o = self.estado.rondas_ganadas_oponente
        empatadas = self.estado.rondas_empatadas
        rondas_jugadas = len(self.estado.resultados_ronda)
        ganador_ultima_no_empate = None
        for resultado in reversed(self.estado.resultados_ronda):
            if resultado in [0, 1]:
                ganador_ultima_no_empate = resultado
                break

        if ganadas_j >= 2:
            return 0
        if ganadas_o >= 2:
            return 1

        if rondas_jugadas >= 2:
            if ganadas_j == 1 and empatadas == 1:
                return ganador_ultima_no_empate
            if ganadas_o == 1 and empatadas == 1:
                return ganador_ultima_no_empate
            if empatadas == 2:
                return None

        if rondas_jugadas >= 3:
            if ganadas_j > ganadas_o:
                return 0
            if ganadas_o > ganadas_j:
                return 1
            if empatadas == 3:
                return 0 if self.estado.es_mano else 1
            if ganador_ultima_no_empate is not None:
                return ganador_ultima_no_empate
            return 0 if self.estado.es_mano else 1

        return None

    def _valor_truco_puntaje(self):
        if self.estado.nivel_truco <= 0:
            return 1
        return self.estado.nivel_truco + 1

    def aplicar_accion(self, accion_idx, player_id):
        """
        Procesa la acción recibida desde el entorno.
        Retorna: (reward, terminado, info)
        """
        reward = 0
        terminado = False
        info = {}

        # Mapeo inverso para legibilidad
        accion = Acciones(accion_idx)
        if player_id != self.estado.turno_actual:
            return -5, False, {"error": "No es el turno del jugador."}

        # ---------------------------------------------------------
        # LÓGICA DE JUGAR CARTAS (Nivel Operativo: check)
        # ---------------------------------------------------------
        if accion in [Acciones.JUGAR_CARTA_1, Acciones.JUGAR_CARTA_2, Acciones.JUGAR_CARTA_3]:
            idx = accion.value # 0, 1 o 2
            if self.estado.turno_responder_envido or self.estado.turno_responder_truco:
                return -5, False, {"error": "Hay un canto pendiente de respuesta."}

            # Verificar si la carta existe (no se jugó ya)
            mano = self.estado.mano_jugador if player_id == 0 else self.estado.mano_oponente
            if idx < len(mano):
                carta_jugada = mano.pop(idx)
                
                # Agregar a mesa
                self.estado.cartas_jugadas.append((carta_jugada, player_id))
                
                # Cambia turno al otro jugador
                self.estado.turno_actual = 1 - player_id

                # Resolver la vuelta cuando ambos jugaron una carta
                if len(self.estado.cartas_jugadas) % 2 == 0:
                    carta_j, jugador_j = self.estado.cartas_jugadas[-2]
                    carta_op, jugador_op = self.estado.cartas_jugadas[-1]
                    resultado = self.determinar_ganador_mano(carta_j, carta_op)
                    if resultado == 0:
                        ganador = jugador_j
                    elif resultado == 1:
                        ganador = jugador_op
                    else:
                        ganador = 2

                    self._registrar_resultado_ronda(ganador)
                    if ganador == 0:
                        reward = 0.5
                    elif ganador == 1:
                        reward = -0.5

                    if ganador == 2:
                        self.estado.turno_actual = 0 if self.estado.es_mano else 1
                    else:
                        self.estado.turno_actual = ganador
                    self.estado.numero_ronda += 1

                    ganador_mano = self._ganador_mano_completa()
                    if ganador_mano is not None:
                        puntos_truco = self._valor_truco_puntaje()
                        if ganador_mano == 0:
                            self.estado.puntos_jugador += puntos_truco
                            reward += puntos_truco
                        else:
                            self.estado.puntos_oponente += puntos_truco
                            reward -= puntos_truco
                        terminado = True
            else:
                # Castigo por intentar jugar carta inexistente
                reward = -5 

        # ---------------------------------------------------------
        # LÓGICA DE CANTOS (Envido / Truco)
        # ---------------------------------------------------------
        elif accion in [Acciones.ENVIDO, Acciones.ENVIDO_ENVIDO, Acciones.REAL_ENVIDO, Acciones.FALTA_ENVIDO]:
            if not self._puede_cantar_envido():
                reward = -5
            elif not self._validar_canto_envido(accion):
                reward = -5
            else:
                if (
                    self.estado.numero_ronda == 1
                    and self.estado.estado_canto_envido == ESTADO_NO_CANTADO
                    and self.estado.turno_responder_truco
                ):
                    self.estado.turno_post_envido = self.estado.jugador_que_canto_truco
                    self.estado.turno_responder_truco = False
                    self.estado.estado_canto_truco = ESTADO_TRUCO_NO_CANTADO
                    self.estado.jugador_que_canto_truco = None
                    self.estado.jugador_que_acepto_truco = None
                        
                self._aplicar_canto_envido(accion, player_id)
                reward = 0

        elif accion in [Acciones.TRUCO, Acciones.RETRUCO, Acciones.VALE_CUATRO]:
            if not self._puede_cantar_truco():
                reward = -5
            elif not self._validar_canto_truco(accion, player_id):
                reward = -5
            else:
                self._aplicar_canto_truco(accion, player_id)
                reward = 0

        elif accion == Acciones.QUIERO:
            if self.estado.turno_responder_envido:
                reward += self._resolver_envido(acepta=True, player_id=player_id)
            elif self.estado.turno_responder_truco:
                reward += self._resolver_truco(acepta=True, player_id=player_id)

        elif accion == Acciones.NO_QUIERO:
            if self.estado.turno_responder_envido:
                reward += self._resolver_envido(acepta=False, player_id=player_id)
            elif self.estado.turno_responder_truco:
                reward += self._resolver_truco(acepta=False, player_id=player_id)
                terminado = True

        elif accion == Acciones.IR_AL_MAZO:
            if player_id == 0:
                self.estado.puntos_oponente += 1
                reward = -1
            else:
                self.estado.puntos_jugador += 1
                reward = 1
            terminado = True

        # Verificar fin de partida (15 o 30 puntos)
        if self.estado.puntos_jugador >= 30:
            reward += 100
            terminado = True
        elif self.estado.puntos_oponente >= 30:
            reward -= 100
            terminado = True

        return reward, terminado, info

    def _puede_cantar_envido(self):
        return (
            self.estado.numero_ronda == 1
            and not self.estado.envido_finalizado
            and self.estado.nivel_truco == 0
        )

    def _validar_canto_envido(self, accion):
        estado_actual = self.estado.estado_canto_envido
        if accion == Acciones.ENVIDO:
            return estado_actual == ESTADO_NO_CANTADO
        if accion == Acciones.ENVIDO_ENVIDO:
            return estado_actual == ESTADO_ENVIDO
        if accion == Acciones.REAL_ENVIDO:
            return estado_actual not in [ESTADO_REAL_ENVIDO, ESTADO_FALTA_ENVIDO, ESTADO_CERRADO]
        if accion == Acciones.FALTA_ENVIDO:
            return estado_actual != ESTADO_CERRADO
        return False

    def _aplicar_canto_envido(self, accion, player_id):
        self.estado.envido_total_anterior = self.estado.envido_total
        estado_actual = self.estado.estado_canto_envido

        if accion == Acciones.ENVIDO:
            self.estado.envido_total = 2
            self.estado.estado_canto_envido = ESTADO_ENVIDO
        elif accion == Acciones.ENVIDO_ENVIDO:
            self.estado.envido_total = 4
            self.estado.estado_canto_envido = ESTADO_ENVIDO_ENVIDO
        elif accion == Acciones.REAL_ENVIDO:
            if estado_actual == ESTADO_NO_CANTADO:
                self.estado.envido_total = 3
            elif estado_actual == ESTADO_ENVIDO:
                self.estado.envido_total = 5
            elif estado_actual == ESTADO_ENVIDO_ENVIDO:
                self.estado.envido_total = 7
            self.estado.estado_canto_envido = ESTADO_REAL_ENVIDO
        elif accion == Acciones.FALTA_ENVIDO:
            self.estado.envido_total = self._calcular_falta_envido()
            self.estado.estado_canto_envido = ESTADO_FALTA_ENVIDO

        self.estado.turno_responder_envido = True
        self.estado.jugador_que_canto_envido = player_id
        self.estado.turno_actual = 1 - player_id

    def _calcular_falta_envido(self):
        objetivo = 30
        puntos_falta = objetivo - max(self.estado.puntos_jugador, self.estado.puntos_oponente)
        return max(1, puntos_falta)

    def _resolver_envido(self, acepta, player_id):
        puntos_mios = self.calcular_puntos_envido(
            self.estado.mano_jugador
            + [c[0] for c in self.estado.cartas_jugadas if c[1] == 0]
        )
        puntos_op = self.calcular_puntos_envido(
            self.estado.mano_oponente
            + [c[0] for c in self.estado.cartas_jugadas if c[1] == 1]
        )

        reward = 0
        if acepta:
            if puntos_mios > puntos_op or (puntos_mios == puntos_op and self.estado.es_mano):
                self.estado.puntos_jugador += self.estado.envido_total
                reward = self.estado.envido_total
            else:
                self.estado.puntos_oponente += self.estado.envido_total
                reward = -self.estado.envido_total
        else:
            puntos_rechazo = max(1, self.estado.envido_total_anterior)
            if player_id == 0:
                self.estado.puntos_oponente += puntos_rechazo
                reward = -puntos_rechazo
            else:
                self.estado.puntos_jugador += puntos_rechazo
                reward = puntos_rechazo

        self.estado.envido_finalizado = True
        self.estado.turno_responder_envido = False
        self.estado.estado_canto_envido = ESTADO_CERRADO
        if self.estado.turno_post_envido is not None:
            self.estado.turno_actual = self.estado.turno_post_envido
            self.estado.turno_post_envido = None
        elif self.estado.jugador_que_canto_envido is not None:
            self.estado.turno_actual = self.estado.jugador_que_canto_envido
        return reward

    def _puede_cantar_truco(self):
        return not self.estado.turno_responder_envido

    def _validar_canto_truco(self, accion, player_id):
        estado_actual = self.estado.estado_canto_truco
        if accion == Acciones.TRUCO:
            return (
                estado_actual == ESTADO_TRUCO_NO_CANTADO
                and not self.estado.turno_responder_truco
                and self.estado.nivel_truco == 0
            )
        if accion == Acciones.RETRUCO:
            if estado_actual == ESTADO_TRUCO and self.estado.turno_responder_truco:
                return True
            return (
                not self.estado.turno_responder_truco
                and self.estado.nivel_truco == 1
                and self.estado.jugador_que_acepto_truco == player_id
            )
        if accion == Acciones.VALE_CUATRO:
            if estado_actual == ESTADO_RETRUCO and self.estado.turno_responder_truco:
                return True
            return (
                not self.estado.turno_responder_truco
                and self.estado.nivel_truco == 2
                and self.estado.jugador_que_acepto_truco == player_id
            )
        return False

    def _aplicar_canto_truco(self, accion, player_id):
        if accion == Acciones.TRUCO:
            self.estado.estado_canto_truco = ESTADO_TRUCO
        elif accion == Acciones.RETRUCO:
            self.estado.estado_canto_truco = ESTADO_RETRUCO
        elif accion == Acciones.VALE_CUATRO:
            self.estado.estado_canto_truco = ESTADO_VALE_CUATRO

        self.estado.turno_responder_truco = True
        self.estado.jugador_que_canto_truco = player_id
        self.estado.turno_actual = 1 - player_id

    def _resolver_truco(self, acepta, player_id):
        reward = 0
        nivel_truco = self._nivel_truco_desde_estado()
        if acepta:
            self.estado.nivel_truco = nivel_truco
            self.estado.jugador_que_acepto_truco = player_id
        else:
            puntos_rechazo = self._puntos_rechazo_truco()
            if player_id == 0:
                self.estado.puntos_oponente += puntos_rechazo
                reward = -puntos_rechazo
            else:
                self.estado.puntos_jugador += puntos_rechazo
                reward = puntos_rechazo

        self.estado.turno_responder_truco = False
        self.estado.estado_canto_truco = ESTADO_TRUCO_CERRADO
        if self.estado.jugador_que_canto_truco is not None:
            self.estado.turno_actual = self.estado.jugador_que_canto_truco
        return reward

    def _nivel_truco_desde_estado(self):
        if self.estado.estado_canto_truco == ESTADO_VALE_CUATRO:
            return 3
        if self.estado.estado_canto_truco == ESTADO_RETRUCO:
            return 2
        if self.estado.estado_canto_truco == ESTADO_TRUCO:
            return 1
        return 0

    def _puntos_rechazo_truco(self):
        if self.estado.estado_canto_truco == ESTADO_VALE_CUATRO:
            return 3
        if self.estado.estado_canto_truco == ESTADO_RETRUCO:
            return 2
        return 1

    def validar_canto_truco(self, player_id):
        return self._validar_canto_truco(Acciones.TRUCO, player_id)

    def get_action_mask(self, player_id):
        """
        Devuelve una lista de booleanos indicando que acciones son validas
        en el estado actual.
        Orden del array corresponde a los indices de Acciones(Enum).
        """
        # Inicializamos todo en False (nada permitido por defecto)
        mask = [False] * len(Acciones)
        if player_id != self.estado.turno_actual:
            return mask
        
        # ----------------------------------------------------
        # 1. MASCARA PARA JUGAR CARTAS
        # ----------------------------------------------------
        # Solo se pueden jugar cartas si NO hay un desafio pendiente de responder
        # (Si me cantaron Truco, primero debo responder, no puedo tirar carta)
        hay_desafio_pendiente = (self.estado.turno_responder_envido or 
                                self.estado.turno_responder_truco)
        
        if not hay_desafio_pendiente:
            mano = self.estado.mano_jugador if player_id == 0 else self.estado.mano_oponente
            if 0 < len(mano): mask[Acciones.JUGAR_CARTA_1.value] = True
            if 1 < len(mano): mask[Acciones.JUGAR_CARTA_2.value] = True
            if 2 < len(mano): mask[Acciones.JUGAR_CARTA_3.value] = True
        
        # ----------------------------------------------------
        # 1b. MASCARA PARA TRUCO
        # ----------------------------------------------------
        if self.estado.turno_responder_truco:
            estado_truco = self.estado.estado_canto_truco
            if estado_truco == ESTADO_TRUCO:
                mask[Acciones.RETRUCO.value] = True
            elif estado_truco == ESTADO_RETRUCO:
                mask[Acciones.VALE_CUATRO.value] = True
            mask[Acciones.QUIERO.value] = True
            mask[Acciones.NO_QUIERO.value] = True
        elif self.estado.turno_responder_envido:
            mask[Acciones.QUIERO.value] = True
            mask[Acciones.NO_QUIERO.value] = True
        else:
            if self.validar_canto_truco(player_id):
                mask[Acciones.TRUCO.value] = True
            if self._validar_canto_truco(Acciones.RETRUCO, player_id):
                mask[Acciones.RETRUCO.value] = True
            if self._validar_canto_truco(Acciones.VALE_CUATRO, player_id):
                mask[Acciones.VALE_CUATRO.value] = True

        # ----------------------------------------------------
        # 2. MASCARA PARA EL ENVIDO (Tu pregunta especifica)
        # ----------------------------------------------------
        # El envido solo se canta en la primera ronda y antes de jugar cartas (generalmente)
        if self._puede_cantar_envido():
            
            estado_actual = self.estado.estado_canto_envido
            responder_envido = self.estado.turno_responder_envido
            puede_interrumpir_truco = (
                self.estado.turno_responder_truco
                and self.estado.nivel_truco == 0
                and not self.estado.envido_finalizado
            )
            if self.estado.turno_responder_truco and not puede_interrumpir_truco:
                return mask

            # CASO A: Nadie canto nada aun (Piso 0)
            # "se puede cantar envido, real o falta, sin haber cantado el anterior"
            if estado_actual == ESTADO_NO_CANTADO and not responder_envido:
                mask[Acciones.ENVIDO.value] = True
                mask[Acciones.REAL_ENVIDO.value] = True
                mask[Acciones.FALTA_ENVIDO.value] = True
                # Envido-Envido PROHIBIDO aqui
            
            # CASO B: Me cantaron "Envido" (Escalon 1)
            # "envido-envido, que unicamente puede cantarse luego del envido simple"
            elif estado_actual == ESTADO_ENVIDO and responder_envido:
                mask[Acciones.ENVIDO_ENVIDO.value] = True # Permitido
                mask[Acciones.REAL_ENVIDO.value] = True   # Permitido elevar
                mask[Acciones.FALTA_ENVIDO.value] = True  # Permitido elevar
                mask[Acciones.QUIERO.value] = True
                mask[Acciones.NO_QUIERO.value] = True
                # Envido simple PROHIBIDO (seria repetir)

            # CASO C: Me cantaron "Envido Envido" (Escalon 2)
            elif estado_actual == ESTADO_ENVIDO_ENVIDO and responder_envido:
                mask[Acciones.REAL_ENVIDO.value] = True   # Permitido
                mask[Acciones.FALTA_ENVIDO.value] = True  # Permitido
                mask[Acciones.QUIERO.value] = True
                mask[Acciones.NO_QUIERO.value] = True
                # Envido y Envido-Envido PROHIBIDOS

            # CASO D: Me cantaron "Real Envido" (Directo o escalado)
            elif estado_actual == ESTADO_REAL_ENVIDO and responder_envido:
                mask[Acciones.FALTA_ENVIDO.value] = True  # Unica subida posible
                mask[Acciones.QUIERO.value] = True
                mask[Acciones.NO_QUIERO.value] = True
                # Prohibido volver a Envido o Real Envido
                
            # CASO E: Falta Envido
            elif estado_actual == ESTADO_FALTA_ENVIDO and responder_envido:
                mask[Acciones.QUIERO.value] = True
                mask[Acciones.NO_QUIERO.value] = True
                # No se puede subir mas

        return mask
