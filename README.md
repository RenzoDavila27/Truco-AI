# Truco-AI

Proyecto de Truco Argentino (1v1, sin flor) con un motor de reglas, un entorno tipo Gymnasium y un modo de juego por consola contra agentes simples.

## Funcionalidades principales

- Motor de reglas de Truco (rondas, puntos, envido, truco y niveles de apuesta).
- Entorno estilo Gymnasium para entrenamiento o self-play.
- Agentes random y racional.
- Juego por consola humano vs agente configurable.

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
- `game/agents/rational_agent.py`: Agente con reglas deterministicas (envido/truco/cartas).
- `game/agents/registry.py`: Registro de agentes disponibles.
- `game/console_game.py`: Juego 1v1 por consola (humano vs agente configurable).
- `game/agent_vs_agent.py`: Partida completa entre agentes configurables.
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

# Elegir el agente rival
python3 game/console_game.py --agent random
```

Durante la partida, el juego imprime las acciones validas y el estado actual. El rival (agente) juega automaticamente cuando le corresponde.

Agentes disponibles (ver `game/agents/registry.py`):
- `random`: elige acciones validas al azar.
- `rational`: reglas deterministicas para envido, truco y eleccion de cartas.
- `q_learning`: toma la decision segun su q_table (si es vacia o no existe el archivo, tomara decisiones greedy)
## Agente vs agente

Simula una partida completa entre dos agentes:

```bash
python3 game/agent_vs_agent.py
```

Opciones utiles:

```bash
# Elegir agentes para J0 y J1
python3 game/agent_vs_agent.py --agent-0 rational --agent-1 random

# Ver todo el estado (modo debug)
python3 game/agent_vs_agent.py --render debug
```

## Matchup de agentes (estadisticas)

Simula multiples partidas y guarda estadisticas en CSV y un resumen en TXT:

```bash
python3 game/agent_matchup.py --agent-0 random --agent-1 rational --games 200
```

Opciones utiles:

```bash
# Personalizar archivos de salida
python3 game/agent_matchup.py --agent-0 q_learning --agent-1 rational --games 500 --output-csv q_learningvsrationalresults.csv --output-summary q_learningvsrationalsummary.txt
```

## Agente Q-Learning (RL)

El agente Q-Learning esta en `game/agents/RL-Agents/agent_q_learning.py`. Usa una Q-Table persistida en `game/agents/RL-Agents/q_tables/q_table.pkl` para elegir acciones de forma greedy (explotacion).

### Entrenamiento (self-play)

Entrena al agente jugando contra si mismo:

```bash
python3 game/agents/RL-Agents/train_q_learning.py --episodes 1000
```

Opciones utiles:

```bash
python3 game/agents/RL-Agents/train_q_learning.py --episodes 5000 --alpha 0.1 --gamma 0.95 --epsilon 0.1
```

Si el entrenamiento se cancela con Ctrl+C, la Q-Table se guarda automaticamente.

### Uso en consola

Para jugar contra el agente entrenado:

```bash
python3 game/console_game.py --agent q_learning
```

La tabla se carga automaticamente al iniciar el agente. Si no existe, juega con valores Q en cero (comportamiento casi aleatorio).

## Notas

- El entorno esta pensado para extenderse a entrenamiento RL (self-play).
- Las reglas actuales implementan un set 1v1 sin flor.
