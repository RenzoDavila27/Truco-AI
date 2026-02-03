# Truco-AI

Proyecto de Truco Argentino (1v1, sin flor) con un motor de reglas, un entorno tipo Gymnasium y un modo de juego por consola contra agentes simples.

## Funcionalidades principales

- Motor de reglas de Truco (rondas, puntos, envido, truco y niveles de apuesta).
- Entorno estilo Gymnasium para entrenamiento o self-play.
- Agentes random y racional.
- Juego por consola humano vs agente configurable.

## Requisitos

Python 3.11+.

Crear y activar un entorno virtual:

Linux/Mac:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
```

Instalar dependencias:

```bash
pip install -r requirements.txt
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
- `game/agent_matchup.py`: Simulador de multiples partidas con estadisticas.
- `game/agent_bluff_matchup.py`: Simulador de partidas con recoleccion de tasas de mentira.
- `game/agents/RL-Agents/agent_q_learning.py`: Agente Q-Learning con Q-Table persistida.
- `game/agents/RL-Agents/train_q_learning.py`: Entrenamiento Q-Learning en self-play.
- `game/agents/RL-Agents/train_q_learning_vs_agent.py`: Entrenamiento Q-Learning contra un agente fijo.
- `game/agents/RL-Agents/analyze_q_table.py`: Analisis de Q-Table con resumen por acciones y top valores.
- `game/sb3/sb3_env.py`: Wrapper compatible con Stable-Baselines3.
- `game/sb3/sb3_train.py`: Entrenamiento PPO con action masking (MaskablePPO).
- `game/sb3/sb3_agent.py`: Wrapper para cargar modelos SB3 como agente.
- `game/sb3/models/`: Modelos entrenados con SB3.
- `game/plots/points_violin.py`: Generador de graficos de violin para distribucion de puntos.
- `game/plots/points_boxplot.py`: Generador de graficos boxplot para puntos.
- `game/plots/bluff_rate_bars.py`: Generador de graficos de barras para tasas de mentira.
- `game/plots/matchup_heatmap.py`: Generador de heatmaps para resultados de matchups.
- `game/plots/points_sources_area.py`: Generador de graficos de area para fuentes de puntos.
- `resultados/`: Directorio con CSVs y resumenes de simulaciones.
- `referencias/`: Directorio con papers y documentacion de referencia.
- `proyecto_final.md`: Informe final del proyecto.
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
- `q_learning`: toma la decision segun su q_table (si es vacia o no existe el archivo, tomara decisiones greedy).
- `sb3`: modelo PPO de SB3 (usa `SB3_TRUCO_MODEL` si se define).

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

Parametros principales:

- `--agent-0`: agente para J0 (ver registry).
- `--agent-1`: agente para J1 (ver registry).
- `--games`: cantidad de partidas a simular.
- `--output-csv`: nombre del CSV de salida (se guarda en `resultados/`).
- `--output-summary`: nombre del TXT de resumen (se guarda en `resultados/`).

## Agente Q-Learning (RL)

El agente Q-Learning esta en `game/agents/RL-Agents/agent_q_learning.py`. Usa una Q-Table persistida en `game/agents/RL-Agents/q_tables/q_table.pkl` para elegir acciones de forma greedy (explotacion).

### Metodo de entrenamiento (resumen)

- Entrenamiento por manos (episodios cortos), con actualizacion tipo Monte Carlo sobre el retorno final de la mano.
- Rewards principales: diferencia de puntos de la mano normalizada, penalizacion por jugadas invalidas y pequena penalizacion por irse al mazo.
- Episodios usados como referencia: ~20M episodios de self-play y ~700k episodios contra el agente racional.
- Q-Table entrenada: **[[link de descarga](https://drive.google.com/file/d/1_nmCoLXxLDg0361OvnOxTrOd8zFxHb0Y/view?usp=sharing)]**.

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

### Entrenamiento vs agente fijo

Entrena el agente Q-Learning contra un agente fijo (no self-play):

```bash
python3 game/agents/RL-Agents/train_q_learning_vs_agent.py --episodes 1000 --opponent rational
```

Parametros principales:

- `--episodes`: cantidad de episodios.
- `--alpha`: learning rate.
- `--gamma`: discount factor.
- `--epsilon`: epsilon inicial para exploracion (decae con coseno).
- `--reset-q-table`: reinicia la Q-Table antes de entrenar.
- `--opponent`: agente oponente (ver registry).
- `--q-player`: posicion del Q-Learning (0 o 1).

### Analisis de Q-Table

Genera un reporte con estadisticas de la Q-Table:

```bash
python3 game/agents/RL-Agents/analyze_q_table.py --output q_table_report.txt
```

Parametros principales:

- `--input`: ruta de la Q-Table (por defecto `game/agents/RL-Agents/q_tables/q_table.pkl`).
- `--output`: ruta del reporte de salida.

### Uso en consola

Para jugar contra el agente entrenado:

```bash
python3 game/console_game.py --agent q_learning
```

La tabla se carga automaticamente al iniciar el agente. Si no existe, juega con valores Q en cero (comportamiento casi aleatorio).

## Entrenamiento SB3 (MaskablePPO)

Entrena un agente PPO con action masking:

```bash
python3 game/sb3/sb3_train.py --timesteps 200000 --opponent random
```

Opciones utiles:

```bash
python3 game/sb3/sb3_train.py --timesteps 500000 --opponent selfplay --selfplay-winrate 0.8
```

Parametros principales:

- `--timesteps`: pasos de entrenamiento.
- `--opponent`: `random`, `rational` o `selfplay`.
- `--output`: ruta del modelo (default `game/sb3/models/ppo_truco`).
- `--selfplay-snapshot`: snapshot del oponente en self-play.

Para usar el modelo entrenado en consola:

Linux/Mac:

```bash
SB3_TRUCO_MODEL=game/sb3/models/ppo_truco python3 game/console_game.py --agent sb3
```

Windows (PowerShell):

```powershell
$env:SB3_TRUCO_MODEL="game\\sb3\\models\\ppo_truco"
python game\\console_game.py --agent sb3
```

## Notas

- Las reglas actuales implementan un set 1v1 sin flor.
