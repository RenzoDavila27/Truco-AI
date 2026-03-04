from constantes import Acciones, ESTADO_RETRUCO, ESTADO_VALE_CUATRO


class RationalAgent:
    def choose_action(self, action_mask, env=None, player_id=0):
        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        if not valid_actions:
            return None
        if env is None:
            return valid_actions[0]

        estado = env.logic.estado
        envido_points = self._calcular_tantos_envido(env, player_id)
        is_mano = estado.es_mano if player_id == 0 else not estado.es_mano

        if estado.turno_responder_truco:
            cartas_fuertes = self._contar_cartas_fuertes(env, player_id)
            gano_una = self._ya_gano_una_ronda(estado, player_id)
            condicion_retruco = cartas_fuertes >= 1 and gano_una
            condicion_vale_cuatro = cartas_fuertes >= 2 and gano_una

            if action_mask[Acciones.VALE_CUATRO.value] and condicion_vale_cuatro:
                return Acciones.VALE_CUATRO.value
            if action_mask[Acciones.RETRUCO.value] and condicion_retruco:
                return Acciones.RETRUCO.value

            if self._todas_cartas_debiles(env, player_id):
                if action_mask[Acciones.NO_QUIERO.value]:
                    return Acciones.NO_QUIERO.value

            if action_mask[Acciones.QUIERO.value]:
                aceptar_retruco = condicion_retruco or cartas_fuertes >= 2
                aceptar_vale_cuatro = condicion_vale_cuatro or (
                    self._promedio_ranking_mano(env, player_id) <= 6
                )
                if estado.estado_canto_truco == ESTADO_RETRUCO and not aceptar_retruco:
                    if action_mask[Acciones.NO_QUIERO.value]:
                        return Acciones.NO_QUIERO.value
                if estado.estado_canto_truco == ESTADO_VALE_CUATRO and not aceptar_vale_cuatro:
                    if action_mask[Acciones.NO_QUIERO.value]:
                        return Acciones.NO_QUIERO.value
                return Acciones.QUIERO.value

        if estado.turno_responder_envido:
            raise_action = self._mejor_subida_envido(action_mask)
            if envido_points == 33 and is_mano and raise_action is not None:
                return raise_action
            if envido_points > 30 and raise_action is not None:
                return raise_action
            if envido_points > 25 and action_mask[Acciones.QUIERO.value]:
                return Acciones.QUIERO.value
            if action_mask[Acciones.NO_QUIERO.value]:
                return Acciones.NO_QUIERO.value

        if not estado.turno_responder_envido:
            if envido_points > 30 and action_mask[Acciones.REAL_ENVIDO.value]:
                return Acciones.REAL_ENVIDO.value
            if envido_points > 27 and action_mask[Acciones.ENVIDO.value]:
                return Acciones.ENVIDO.value

        cartas_fuertes = self._contar_cartas_fuertes(env, player_id)
        gano_una = self._ya_gano_una_ronda(estado, player_id)
        condicion_retruco = cartas_fuertes >= 1 and gano_una
        condicion_vale_cuatro = cartas_fuertes >= 2 and gano_una

        if action_mask[Acciones.VALE_CUATRO.value] and condicion_vale_cuatro:
            return Acciones.VALE_CUATRO.value
        if action_mask[Acciones.RETRUCO.value] and condicion_retruco:
            return Acciones.RETRUCO.value
        if action_mask[Acciones.TRUCO.value] and condicion_retruco:
            return Acciones.TRUCO.value

        card_action = self._elegir_carta(action_mask, env, player_id)
        if card_action is not None:
            return card_action

        return valid_actions[0]

    def _calcular_tantos_envido(self, env, player_id):
        estado = env.logic.estado
        if player_id == 0:
            mano = estado.mano_jugador
        else:
            mano = estado.mano_oponente
        jugadas = [c[0] for c in estado.cartas_jugadas if c[1] == player_id]
        return env.logic.calcular_puntos_envido(mano + jugadas)

    def _mejor_subida_envido(self, action_mask):
        if action_mask[Acciones.FALTA_ENVIDO.value]:
            return Acciones.FALTA_ENVIDO.value
        if action_mask[Acciones.REAL_ENVIDO.value]:
            return Acciones.REAL_ENVIDO.value
        if action_mask[Acciones.ENVIDO_ENVIDO.value]:
            return Acciones.ENVIDO_ENVIDO.value
        return None

    def _elegir_carta(self, action_mask, env, player_id):
        indices = []
        if action_mask[Acciones.JUGAR_CARTA_1.value]:
            indices.append(0)
        if action_mask[Acciones.JUGAR_CARTA_2.value]:
            indices.append(1)
        if action_mask[Acciones.JUGAR_CARTA_3.value]:
            indices.append(2)
        if not indices:
            return None

        estado = env.logic.estado
        mano = estado.mano_jugador if player_id == 0 else estado.mano_oponente

        opponent_card = None
        if estado.cartas_jugadas and len(estado.cartas_jugadas) % 2 == 1:
            last_card, last_player = estado.cartas_jugadas[-1]
            if last_player != player_id:
                opponent_card = last_card

        if opponent_card is not None:
            opponent_rank = env.logic.obtener_ranking(opponent_card)
            stronger = []
            for idx in indices:
                rank = env.logic.obtener_ranking(mano[idx])
                if rank < opponent_rank:
                    stronger.append((idx, rank))
            if stronger:
                best_idx = max(stronger, key=lambda item: item[1])[0]
                return Acciones(best_idx).value

        if estado.numero_ronda == 1 and opponent_card is None:
            strongest_idx = min(
                indices,
                key=lambda idx: env.logic.obtener_ranking(mano[idx]),
            )
            return Acciones(strongest_idx).value

        weakest_idx = max(
            indices,
            key=lambda idx: env.logic.obtener_ranking(mano[idx]),
        )
        return Acciones(weakest_idx).value

    def _ya_gano_una_ronda(self, estado, player_id):
        if player_id == 0:
            return estado.rondas_ganadas_jugador > 0
        return estado.rondas_ganadas_oponente > 0

    def _tiene_cartas_poderosas(self, env, player_id):
        estado = env.logic.estado
        mano = estado.mano_jugador if player_id == 0 else estado.mano_oponente
        if not mano:
            return False
        return any(env.logic.obtener_ranking(carta) <= 2 for carta in mano)

    def _contar_cartas_fuertes(self, env, player_id):
        estado = env.logic.estado
        mano = estado.mano_jugador if player_id == 0 else estado.mano_oponente
        return sum(1 for carta in mano if env.logic.obtener_ranking(carta) <= 5)

    def _promedio_ranking_mano(self, env, player_id):
        estado = env.logic.estado
        mano = estado.mano_jugador if player_id == 0 else estado.mano_oponente
        if not mano:
            return 99
        return sum(env.logic.obtener_ranking(carta) for carta in mano) / len(mano)

    def _todas_cartas_debiles(self, env, player_id):
        estado = env.logic.estado
        mano = estado.mano_jugador if player_id == 0 else estado.mano_oponente
        if not mano:
            return True
        return all(env.logic.obtener_ranking(carta) >= 10 for carta in mano)
