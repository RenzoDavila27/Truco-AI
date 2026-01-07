# Truco-AI

Proyecto de Truco Argentino (1v1, sin flor) con un motor de reglas, un entorno tipo Gymnasium y un modo de juego por consola contra agentes simples.

## Funcionalidades principales

- Motor de reglas de Truco (rondas, puntos, envido, truco y niveles de apuesta).
- Entorno estilo Gymnasium para entrenamiento o self-play.
- Agente aleatorio de ejemplo.
- Juego por consola humano vs agente.

## Requisitos

Python 3.11+.

El entorno virtual ya existe en el repo. Para activarlo:

Linux/Mac:
```bash
source truco_ent/bin/activate
```

Windows (PowerShell):
```powershell
truco_ent\\Scripts\\Activate.ps1
```

## Estructura del proyecto

- `game/constantes.py`: Definiciones de palos, mazo, rankings y enumeracion de acciones.
- `game/truco_logic.py`: Motor de reglas y estado del juego (turnos, rondas, cantos, puntajes).
- `game/truco_env.py`: Wrapper tipo Gymnasium que expone `reset`, `step` y `get_action_mask`.
- `game/agents/random_agent.py`: Agente aleatorio que elige acciones validas.
- `game/console_game.py`: Juego 1v1 por consola (humano vs agente).
- `game/jugadas_prueba.txt`: Secuencias de jugadas para validar reglas y turnos.
- `readmeDesafio.md`: Documento original del desafio y contexto teorico.

## Jugar contra un agente (consola)

Ejecuta el juego 1v1 desde la raiz del repo:

```bash
python3 game/console_game.py
```

Opciones utiles:

```bash
# Elegir jugador humano (0 o 1)
python3 game/console_game.py --human-player 1

# Ver todo el estado (modo debug)
python3 game/console_game.py --mode debug
```

Durante la partida, el juego imprime las acciones validas y el estado actual. El rival (agente) juega automaticamente cuando le corresponde.

## Notas

- El entorno esta pensado para extenderse a entrenamiento RL (self-play).
- Las reglas actuales implementan un set 1v1 sin flor.
