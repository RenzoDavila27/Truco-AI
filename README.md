# Truco-AI

Proyecto de Inteligencia Artificial para el Truco Argentino (1v1, sin flor). Incluye un motor de reglas completo, un entorno tipo Gymnasium, agentes heurísticos y dos agentes de aprendizaje por refuerzo (Q-Learning y PPO con MaskablePPO). El agente PPO, entrenado mediante una liga de oponentes, logra dominar todos los enfrentamientos con win rates del 76–93%.

## Funcionalidades principales

- Motor de reglas completo de Truco Argentino (rondas, puntos, envido, truco y niveles de apuesta).
- Entorno estilo Gymnasium para entrenamiento y evaluación.
- 5 agentes: Random, Racional, Q-Learning, SB3 (PPO simple) y SB3 League (PPO con liga de oponentes).
- Juego por consola humano vs agente configurable.
- Simulador de enfrentamientos masivos con estadísticas (matchup y bluff).
- Generación de gráficos: violín, heatmaps, barras de mentira, áreas de fuentes de puntos, convergencia y losses.

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

### Motor y entorno

- `game/constantes.py`: Definiciones de palos, mazo, rankings y enumeración de acciones.
- `game/truco_logic.py`: Motor de reglas y estado del juego (turnos, rondas, cantos, puntajes).
- `game/truco_env.py`: Wrapper tipo Gymnasium que expone `reset`, `step` y `get_action_mask`.

### Agentes

- `game/agents/random_agent.py`: Agente aleatorio que elige acciones válidas al azar.
- `game/agents/rational_agent.py`: Agente con reglas determinísticas (envido/truco/cartas).
- `game/agents/registry.py`: Registro central de agentes disponibles.
- `game/agents/RL-Agents/agent_q_learning.py`: Agente Q-Learning con Q-Table persistida.
- `game/sb3/sb3_agent.py`: Wrapper para cargar modelos SB3 como agente.
- `game/sb3/sb3_league_agent.py`: Agente que carga la política PPO de la liga de oponentes.

### Entrenamiento

- `game/agents/RL-Agents/train_q_learning.py`: Entrenamiento Q-Learning en self-play.
- `game/agents/RL-Agents/train_q_learning_vs_agent.py`: Entrenamiento Q-Learning contra un agente fijo.
- `game/sb3/sb3_train.py`: Entrenamiento PPO con action masking (MaskablePPO).
- `game/sb3/sb3_league_train.py`: Entrenamiento PPO mediante liga de oponentes (self-play + heurísticos + snapshots).
- `game/sb3/sb3_league.py`: Implementación de la liga de oponentes (selección probabilística).

### Simulación y evaluación

- `game/console_game.py`: Juego 1v1 por consola (humano vs agente configurable).
- `game/agent_vs_agent.py`: Partida completa entre agentes configurables.
- `game/agent_matchup.py`: Simulador de múltiples partidas con estadísticas (win rate, puntos, manos jugadas/ganadas).
- `game/agent_bluff_matchup.py`: Simulador con recolección de estadísticas de mentira (truco y envido).
- `game/agents/RL-Agents/analyze_q_table.py`: Análisis de Q-Table con resumen por acciones y top valores.

### Generación de gráficos

- `game/plots/points_violin.py`: Gráficos de violín para distribución de puntos.
- `game/plots/points_boxplot.py`: Gráficos boxplot para puntos.
- `game/plots/bluff_rate_bars.py`: Barras de tasas de mentira por agente.
- `game/plots/matchup_heatmap.py`: Heatmaps de win rate y diferencia de puntos.
- `game/plots/points_sources_area.py`: Gráficos de área para fuentes de puntos (envido/truco/cartas/abandono).
- `game/plots/bluff_rate_q_experiments.py`: Comparación de tasas de mentira entre experimentos Q-Learning.
- `game/plots/accept_rate_q_experiments.py`: Tasas de aceptación por experimento Q-Learning.
- `game/agents/RL-Agents/plot_q_learning_convergence.py`: Curvas de convergencia de Q-Learning.
- `game/sb3/plot_sb3_snapshots.py`: Curvas de evaluación de snapshots PPO durante entrenamiento.
- `game/sb3/plot_ppo_loss.py`: Gráficos de losses de entrenamiento PPO.

### Otros

- `game/sb3/models/`: Modelos entrenados (PPO league y snapshots).
- `game/agents/RL-Agents/q_tables/`: Q-Tables entrenadas.
- `resultados/`: CSVs y resúmenes de simulaciones.
- `referencias/`: Papers y documentación de referencia.
- `proyecto_final.md`: Informe final del proyecto.
- `readmeDesafio.md`: Documento original del desafío y contexto teórico.

## Jugar contra un agente (consola)

```bash
python3 game/console_game.py
```

Opciones útiles:

```bash
# Elegir jugador humano (0 o 1)
python3 game/console_game.py --human-player 1

# Ver todo el estado (modo debug)
python3 game/console_game.py --mode debug

# Elegir el agente rival
python3 game/console_game.py --agent sb3_league
```

Agentes disponibles (ver `game/agents/registry.py`):

| Agente       | Descripción                                                            |
| ------------ | ---------------------------------------------------------------------- |
| `random`     | Elige acciones válidas al azar.                                        |
| `rational`   | Reglas determinísticas para envido, truco y elección de cartas.        |
| `q_learning` | Decisión según Q-Table (si está vacía, comportamiento casi aleatorio). |
| `sb3`        | Modelo PPO de SB3 (usa `SB3_TRUCO_MODEL` si se define).                |
| `sb3_league` | Modelo PPO entrenado con liga de oponentes (el más fuerte).            |

## Matchup de agentes

Simula múltiples partidas y guarda estadísticas:

```bash
python3 game/agent_matchup.py --agent-0 sb3_league --agent-1 rational --games 1000
```

Parámetros principales:

- `--agent-0` / `--agent-1`: agentes para J0 y J1.
- `--games`: cantidad de partidas.
- `--output-csv` / `--output-summary`: nombres de los archivos de salida (en `resultados/`).
- `--q-table-j0` / `--q-table-j1`: Q-Table específica para J0/J1 (si es `q_learning`).
- `--model-j0` / `--model-j1`: modelo `.zip` específico para J0/J1 (si es `sb3` o `sb3_league`).
- `--seed`: seed para resultados reproducibles.

### Matchup con estadísticas de mentira

```bash
python3 game/agent_bluff_matchup.py --agent-0 q_learning --agent-1 sb3_league --games 1000 --seed 2026
```

Acepta los mismos parámetros que `agent_matchup.py`. Genera CSVs con datos detallados de cada canto (truco/envido), incluyendo fuerza de mano, puntos de envido, y si el canto fue una mentira.

## Agente Q-Learning

El agente Q-Learning usa una Q-Table persistida en `game/agents/RL-Agents/q_tables/`. Se seleccionó el Experimento 2 (entrenado contra racional) como política definitiva.

### Entrenamiento

```bash
# Self-play
python3 game/agents/RL-Agents/train_q_learning.py --episodes 1000000

# Contra agente fijo
python3 game/agents/RL-Agents/train_q_learning_vs_agent.py --episodes 1000 --opponent rational
```

Parámetros principales:

- `--episodes`: cantidad de episodios.
- `--alpha`: learning rate.
- `--gamma`: discount factor.
- `--epsilon`: epsilon inicial para exploración (decae con coseno).
- `--reset-q-table`: reinicia la Q-Table antes de entrenar.
- `--opponent`: agente oponente (solo para `train_q_learning_vs_agent.py`).
- `--q-table-name`: nombre del archivo de la Q-Table.

Si el entrenamiento se cancela con Ctrl+C, la Q-Table se guarda automáticamente.

### Gráficos de convergencia

```bash
python3 game/agents/RL-Agents/plot_q_learning_convergence.py \
    --q-tables-dir game/agents/RL-Agents/q_tables \
    --opponent rational --games 200
```

## Agente PPO (MaskablePPO)

### Entrenamiento simple

Entrena un agente PPO con action masking contra un oponente fijo o en self-play:

```bash
python3 game/sb3/sb3_train.py --timesteps 200000 --opponent random
```

### Entrenamiento con liga de oponentes (recomendado)

El entrenamiento por liga combina self-play, agentes heurísticos y snapshots de versiones anteriores del agente. Se realizaron tres experimentos variando la distribución de probabilidades:

| Experimento | Self-play | Heurísticos | Snapshots |
| ----------- | --------- | ----------- | --------- |
| Exp. 1      | 50%       | 0%          | 50%       |
| Exp. 2      | 30%       | 40%         | 30%       |
| Exp. 3      | 20%       | 60%         | 20%       |

Se seleccionó el Experimento 3 como política definitiva (mejor desempeño global).

```bash
python3 game/sb3/sb3_league_train.py \
    --timesteps 20000000 \
    --check-freq 200000 \
    --history-dir game/sb3/models/history_league_20_60_20 \
    --output game/sb3/models/ppo_truco_league_20_60_20 \
    --loss-csv resultados/ppo_training_losses.csv
```

> **Nota:** Las probabilidades de la liga (self-play / heurísticos / snapshots) se configuran editando los defaults del `LeaguePool` en `game/sb3/sb3_league.py` (atributos `self_play_weight`, `heuristic_weight`, `snapshot_weight`).

### Gráficos de entrenamiento PPO

```bash
# Curvas de evaluación durante entrenamiento
python3 game/sb3/plot_sb3_snapshots.py \
    --history-dir game/sb3/models/history_league_20_60_20 \
    --opponent rational --games 200

# Losses de entrenamiento
python3 game/sb3/plot_ppo_loss.py --csv resultados/ppo_training_losses.csv --output ppo_loss.png
```

### Uso en consola

```bash
python3 game/console_game.py --agent sb3_league
```

El agente `sb3_league` carga automáticamente `game/sb3/models/ppo_truco_league.zip`. Se puede forzar otro modelo con la variable de entorno `SB3_TRUCO_LEAGUE_MODEL`.

## Generación de gráficos de resultados

```bash
# Violin plot de distribución de puntos
python3 game/plots/points_violin.py --agent-0 random --agent-1 sb3_league

# Heatmap de win rate
python3 game/plots/matchup_heatmap.py --metric win_rate --agents "random,rational,q_learning,sb3_league"

# Heatmap de diferencia de puntos
python3 game/plots/matchup_heatmap.py --metric avg_diff --agents "random,rational,q_learning,sb3_league"

# Tasas de mentira comparativas
python3 game/plots/bluff_rate_bars.py --agents "random,rational,q_learning,sb3_league"

# Fuentes de puntos (área apilada)
python3 game/plots/points_sources_area.py --focal-agent sb3_league --opponents "random,rational,q_learning"
```

Los gráficos se guardan en `game/plots/images/`.

## Resultados principales

Resultados de 1000 partidas por enfrentamiento (seed 2026):

| Agente         | vs Random | vs Rational | vs Q-Learning | vs PPO |
| -------------- | --------- | ----------- | ------------- | ------ |
| **Random**     | —         | 5.1%        | 33.9%         | 6.6%   |
| **Rational**   | 94.9%     | —           | 30.6%         | 23.8%  |
| **Q-Learning** | 66.1%     | 69.4%       | —             | 21.8%  |
| **PPO**        | 93.4%     | 76.2%       | 78.2%         | —      |

El agente PPO (Exp. 3, liga 20/60/20) es el claro ganador, dominando todos los enfrentamientos.

## Notas

- Las reglas actuales implementan un set 1v1 sin flor.
- Q-Table de referencia: `exp2_vs_rational_789987.pkl` → copiar como `q_table.pkl` para uso por defecto.
- Modelo PPO de referencia: `ppo_truco_league_20_60_20.zip` → copiar como `ppo_truco_league.zip` para uso por defecto.
