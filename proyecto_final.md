# Proyecto Final — Inteligencia Artificial 1

**Título:** Truco-AI: Agentes inteligentes para Truco Argentino (1v1, sin flor)  
**Autores:** Ramiro Martinez, Renzo Dávila  
**Fecha:** 12/02/2026
**Cátedra:** Inteligencia Artificial 1 - UNCUYO

---

## Índice

1. [Introducción](#1-introducción)
2. [Marco teórico](#2-marco-teórico)
   - [2.1 El Truco Argentino](#21-el-truco-argentino)
   - [2.2 Aprendizaje por refuerzo (RL)](#22-aprendizaje-por-refuerzo-rl)
   - [2.3 Políticas y evaluación de decisiones](#23-políticas-y-evaluación-de-decisiones)
   - [2.4 Q-Learning y Monte Carlo Q-Learning](#24-q-learning-y-monte-carlo-q-learning)
   - [2.5 PPO (Proximal Policy Optimization)](#25-ppo-proximal-policy-optimization)
3. [Diseño experimental](#3-diseño-experimental)
   - [3.1 Métricas](#31-métricas)
   - [3.2 Herramientas](#32-herramientas)
   - [3.3 Agentes utilizados](#33-agentes-utilizados)
   - [3.4 Creación del entorno personalizado](#34-creación-del-entorno-personalizado)
   - [3.5 Agentes y sus entrenamientos](#35-agentes-y-sus-entrenamientos)
   - [3.6 Experimentos y resultados](#36-experimentos-y-resultados)
   - [3.7 Visualización de resultados](#37-visualización-de-resultados)
4. [Análisis y discusión de resultados](#4-análisis-y-discusión-de-resultados)
   - [4.1 Agente Random](#41-agente-random)
   - [4.2 Agente Racional](#42-agente-racional)
   - [4.3 Agente Q-Learning](#43-agente-q-learning)
   - [4.4 Agente PPO](#44-agente-ppo)
5. [Conclusiones finales](#5-conclusiones-finales)
6. [Referencias](#6-referencias)

---

## 1. Introducción

El Truco Argentino es un juego de cartas tradicional de dos jugadores, por turnos y de información imperfecta. Cada jugador recibe un conjunto reducido de cartas y, a lo largo de la mano, debe decidir cómo y cuándo jugar, cuándo aceptar o rechazar desafíos y cómo interpretar las señales del rival. La partida combina mecánicas de enfrentamiento directo con un componente fuerte de toma de decisiones bajo incertidumbre, ya que las cartas del oponente son desconocidas y la información disponible se construye a partir de las jugadas, los cantos y el contexto de la mano. En términos generales, se trata de un juego competitivo donde la estrategia depende tanto de la evaluación de la mano propia como de la lectura del oponente.

En este tipo de juegos, los **agentes tradicionales** suelen basarse en reglas fijas o heurísticas diseñadas a mano, y a menudo incorporan **parámetros de estilo** (agresividad, frecuencia de mentira, tolerancia al riesgo, etc.). En lugar de aprender, estos agentes aplican reglas predefinidas para elegir acciones “razonables” con la información parcial disponible. Este enfoque produce comportamientos coherentes y explicables, pero tiene limitaciones: al no aprender de la experiencia, no se adapta de forma dinámica a oponentes o contextos nuevos. Por eso, estos agentes son una buena línea base para comparar con enfoques de **Reinforcement Learning**, que aprenden políticas de acción a partir de la interacción con el entorno.

El Truco es un candidato adecuado para aplicar RL porque presenta los elementos clásicos de un entorno de aprendizaje: decisiones secuenciales, recompensas claras (puntos ganados o perdidos) y un espacio de estados rico en incertidumbre. Además, su naturaleza de información imperfecta permite estudiar comportamientos de exploración, adaptación y modelado implícito del oponente. Sin embargo, también plantea desafíos: el estado observado es parcial, las recompensas pueden ser retrasadas y la dinámica de apuestas introduce componentes estratégicos difíciles de capturar con reglas simples. En este sentido, el Truco resulta interesante para RL, pero exige un diseño cuidadoso de la representación del estado, la función de recompensa y la evaluación experimental.

Este informe se organiza de la siguiente manera: primero se presenta el marco teórico necesario para entender el problema y las técnicas empleadas; luego se describe el diseño experimental, las métricas y herramientas utilizadas; más adelante se reportan los resultados obtenidos y su análisis; y finalmente se exponen las conclusiones y el trabajo futuro, junto con la bibliografía consultada.

---

## 2. Marco teórico

### 2.1 El Truco Argentino

El Truco Argentino [12] es un juego de cartas tradicional para dos jugadores (en la modalidad utilizada en este proyecto), de naturaleza competitiva y de información imperfecta. Se juega con un mazo español de 40 cartas (sin ochos ni nueves), y cada mano comienza con el reparto de tres cartas a cada jugador. La partida se juega a 30 puntos.

#### Estructura de una mano

Cada mano se compone de hasta tres **rondas**. En cada ronda, ambos jugadores juegan una carta y se comparan según una jerarquía de fuerza específica del juego. El jugador que gana dos de las tres rondas gana la mano. Si se producen empates (llamados **pardas**), se aplican reglas especiales: si la primera ronda empata, gana quien gane cualquiera de las siguientes; si la primera se gana y la segunda empata, gana quien ganó la primera; si las tres rondas empatan, gana el jugador que es **mano** (rol que alterna entre manos y determina quién juega primero).

#### Jerarquía de cartas

A diferencia de otros juegos de cartas, el Truco posee una jerarquía de fuerza no lineal que no sigue el orden numérico convencional. De mayor a menor fuerza, las cartas se ordenan así:

1 de Espadas → 1 de Bastos → 7 de Espadas → 7 de Oros → 3 → 2 → 1 (copas/oros) → 12 → 11 → 10 → 7 (copas/bastos) → 6 → 5 → 4

Esta jerarquía implica que un 3 supera a un 7 de copas, y que los ases de espadas y bastos son las cartas más poderosas del juego. Cada carta también posee un **valor de envido** independiente: los naipes del 1 al 7 contribuyen con su valor numérico, mientras que las figuras (10, 11, 12) contribuyen con cero.

#### El Truco (apuesta por la mano)

El **truco** es una apuesta sobre el resultado de la mano. En cualquier momento de su turno, un jugador puede **cantar truco**, desafiando al oponente a aceptar una apuesta de mayor valor. Las variantes escalan de la siguiente forma:

| Canto       | Puntos si gana | Puntos si rechaza |
| ----------- | -------------- | ----------------- |
| Sin canto   | 1              | —                 |
| Truco       | 2              | 1                 |
| Retruco     | 3              | 2                 |
| Vale cuatro | 4              | 3                 |

Cuando se canta truco, el oponente puede **aceptar** (quiero), **rechazar** (no quiero, entregando los puntos correspondientes) o **re-cantar** (elevar al siguiente nivel). Solo el jugador que aceptó el nivel anterior puede proponer la escalada siguiente. Si un jugador rechaza, la mano termina inmediatamente. Un jugador también puede **ir al mazo** (retirarse de la mano) en cualquier momento, cediendo los puntos en juego.

#### El Envido (apuesta por el tanto)

El **envido** es una apuesta que se resuelve comparando el **tanto** de cada jugador, calculado a partir de sus cartas en mano. Solo puede cantarse durante la primera ronda, antes de que se jueguen cartas o inmediatamente después de que se juegue la primera carta. El tanto se calcula así: si el jugador tiene dos o más cartas del mismo palo, suma 20 puntos de base más los valores de envido de las dos cartas de mayor valor en ese palo; en caso contrario, el tanto es igual al valor de envido de su carta más alta.

Las variantes del envido y su puntuación son:

| Canto         | Puntos si gana                  | Puntos si rechaza              |
| ------------- | ------------------------------- | ------------------------------ |
| Envido        | 2                               | 1                              |
| Envido-Envido | 4                               | 2                              |
| Real Envido   | 3,5 o 7 (segun envido previo)   | 1, 2 o 4 (segun envido previo) |
| Falta Envido  | 30 − puntos del ganador parcial | Puntos del nivel anterior      |

El envido-envido solo puede cantarse como respuesta a un envido simple. El real envido puede cantarse directamente o como escalada. La falta envido puede proponerse desde cualquier estado del envido. El envido tiene prioridad sobre el truco e incluso puede interrumpir un canto de truco si se realiza en el momento oportuno.

#### El rol de la mentira

Un aspecto estratégico central del Truco es que los cantos (tanto de truco como de envido) **no requieren que el jugador tenga cartas que justifiquen la apuesta**. Un jugador puede cantar truco con una mano débil o envido con un tanto bajo, apostando a que el rival rechazará y cederá puntos. Esta mecánica de **mentira** introduce un componente de engaño que diferencia al Truco de juegos puramente basados en la fuerza de las cartas.

### 2.2 Aprendizaje por refuerzo (RL)

El **aprendizaje por refuerzo** [9] es un paradigma de aprendizaje automático donde un **agente** interactúa con un **entorno** tomando **acciones** y recibiendo **recompensas**. El objetivo del agente es aprender una **política** que maximice el beneficio esperado en el tiempo. En el Truco, el entorno es el juego, las acciones corresponden a las decisiones posibles (jugar una carta, cantar, aceptar o rechazar), y la recompensa se asocia al resultado en puntos.

Un concepto central es la formulación como **Proceso de Decisión de Markov Parcialmente Observable (POMDP)** [9]. Formalmente, un POMDP se describe por:

- **S:** conjunto de estados reales.
- **A:** conjunto de acciones.
- **T(s'|s,a):** función de transición.
- **R(s,a):** recompensa.
- **O:** conjunto de observaciones.
- **Z(o|s',a):** modelo de observación.
- **γ:** factor de descuento.

En Truco, el estado completo incluye las cartas de ambos jugadores, pero el agente solo accede a una observación parcial (sus cartas e historial), lo que obliga a decidir bajo incertidumbre, integrando señales del rival y el contexto de la mano.

El Truco pertenece a la categoría de **juegos de suma cero** [10]: todo punto que un jugador gana es un punto que el otro pierde. Formalmente, la suma de las utilidades de ambos jugadores es constante en cada resultado posible (**u₁ + u₂ = 0**). Esta propiedad implica que los intereses de los jugadores son estrictamente opuestos, lo que convierte cada decisión en un problema de optimización competitiva donde mejorar el propio desempeño necesariamente empeora el del rival.

Además, al tratarse de un juego competitivo de dos jugadores, el análisis se vincula con nociones de **teoría de juegos**, como el **equilibrio de Nash** [10]. En términos generales, un perfil de estrategias **(π₁, π₂)** es un equilibrio de Nash si ninguna parte puede mejorar su resultado cambiando unilateralmente su estrategia, es decir:

- **u₁(π₁, π₂) ≥ u₁(π₁', π₂)**
- **u₂(π₁, π₂) ≥ u₂(π₁, π₂')**

para cualquier estrategia alternativa **π₁'** o **π₂'**. En juegos de suma cero, esta noción se relaciona con estrategias minimax, donde cada jugador busca minimizar la máxima ganancia posible del rival, lo que garantiza la protección contra la mejor respuesta del oponente.

En esta clase de problemas también es habitual restringir las **acciones válidas** en cada turno para mantener la coherencia con las reglas del juego. Estas restricciones permiten que el agente concentre su aprendizaje en decisiones legítimas, evitando estados imposibles. En conjunto, estas características hacen del Truco un dominio natural para RL: decisiones secuenciales, información parcial y un fuerte componente estratégico que depende tanto de la mano propia como de la interpretación del oponente.

Una técnica de entrenamiento particularmente relevante en juegos de suma cero es el **self-play** (auto-juego), donde el agente entrena jugando contra copias de sí mismo. En lugar de requerir un oponente externo predefinido, el agente actúa simultáneamente como jugador y como rival, lo que genera un proceso de co-evolución: a medida que el agente mejora, su oponente (que es él mismo) también mejora, forzando una adaptación continua. Esta dinámica puede aproximar estrategias de equilibrio de Nash, ya que el agente debe aprender a responder a un oponente que utiliza exactamente su misma política. El self-play ayuda a los agentes a descubrir estrategias robustas donde no existe un oponente optimo conocido.

### 2.3 Políticas y evaluación de decisiones

Una **política** define cómo decide el agente en función de lo que observa. Para comparar alternativas, se utilizan **criterios de evaluación** que estiman qué tan conveniente es una acción según las recompensas esperadas. Estas estimaciones permiten ordenar las decisiones disponibles y elegir la opción que, en promedio, conduce a mejores resultados. En un juego como el Truco, esta evaluación debe considerar que el efecto de una decisión puede manifestarse varios turnos después y que el rival influye activamente en el resultado.

---

### 2.4 Q-Learning y Monte Carlo Q-Learning

Q-Learning [9] es un algoritmo model-free de aprendizaje por refuerzo que aprende una función de acción-valor **Q(s,a)**, la cual estima la recompensa esperada al ejecutar una acción **a** en un estado **s** y continuar con la política aprendida. La actualización clásica, basada en diferencias temporales (TD), es:

**Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]**

En Q-Learning clásico (basado en diferencias temporales), los valores se actualizan después de cada transición utilizando la estimación del siguiente estado. Sin embargo, en dominios episódicos donde el resultado de una jugada depende fuertemente del desenlace final, las recompensas intermedias no reflejan completamente el valor estratégico de cada decisión. Por esta razón, la variante **Monte Carlo Q-Learning** [1] resulta más adecuada: en lugar de actualizar tras cada paso, acumula todas las transiciones de un episodio y actualiza los valores una vez conocido el resultado final. La regla de actualización, aplicada en orden inverso desde el estado terminal, es:

**Q(s,a) ← Q(s,a) + α × (G − Q(s,a))**

Los componentes de esta fórmula son:

- **Q(s,a):** el valor actual almacenado en la tabla para el par estado-acción.
- **α (alpha):** la tasa de aprendizaje, que controla cuánto peso se da a la nueva información frente al valor previo.
- **G:** el retorno acumulado desde el paso _t_ hasta el final del episodio, definido como:

  **Gₜ = rₜ + γ rₜ₊₁ + γ² rₜ₊₂ + … = Σₖ₌₀ᵀ⁻ᵗ γᵏ rₜ₊ₖ**

  donde **rₜ₊ₖ** es la recompensa recibida en el paso _t+k_ y _T_ es el paso final del episodio.

- **γ (gamma):** el factor de descuento (γ ∈ [0, 1]), que controla la importancia relativa de las recompensas futuras en el cálculo de G. Con γ < 1, las recompensas más lejanas se atenúan exponencialmente (γᵏ decrece a medida que _k_ crece), dando mayor peso a las consecuencias inmediatas. Con γ = 1, todas las recompensas del episodio contribuyen por igual al retorno, lo cual es apropiado en dominios donde el resultado final es lo que importa.

#### Exploración y convergencia

Un aspecto fundamental del aprendizaje por refuerzo es el balance entre **exploración** (probar acciones desconocidas) y **explotación** (aprovechar el conocimiento adquirido). La estrategia más utilizada es **epsilon-greedy**: con probabilidad ε el agente elige una acción aleatoria, y con probabilidad (1-ε) elige la acción con mayor valor Q estimado.

Para garantizar la convergencia, es habitual aplicar un **decaimiento progresivo** de ε a lo largo del entrenamiento. Inicialmente, el agente explora con alta probabilidad; a medida que avanzan los episodios, esta probabilidad decrece, favoreciendo gradualmente la explotación de los valores aprendidos. Una variante particularmente efectiva es el **decaimiento cosenoidal**[1], cuya fórmula es:

**ε(t) = ε₀ × cos(t × π / 2T)**

donde **ε₀** es el valor inicial de exploración, **t** es el episodio actual y **T** es el número total de episodios de entrenamiento. Esta función produce una curva suave que mantiene alta exploración en las primeras etapas (cuando el coseno está cerca de 1) y la reduce gradualmente hasta cero al final del entrenamiento (cuando el coseno alcanza 0). A diferencia del decaimiento lineal o exponencial, el decaimiento cosenoidal ofrece una transición más gradual en las fases intermedias, evitando reducciones bruscas de la exploración.

### 2.5 PPO (Proximal Policy Optimization)

**PPO (Proximal Policy Optimization)** [11] es un método de optimización de políticas que busca mejorar la estabilidad del entrenamiento evitando actualizaciones demasiado grandes. En lugar de cambiar la política libremente, impone un límite al cambio permitido en cada paso.

En términos simples, PPO compara la política nueva con la anterior y solo permite cambios si el beneficio esperado mejora sin desviarse demasiado. Si el cambio propuesto es excesivo, se recorta para mantener la actualización dentro de un rango seguro. Esto evita saltos bruscos que podrían degradar el desempeño y ayuda a que el aprendizaje sea más estable.

#### Aproximación de función con redes neuronales

A diferencia de métodos tabulares como Q-Learning, PPO emplea una **red neuronal** como aproximador de función para representar la política. Esta representación continua permite que el agente generalice a estados no visitados durante el entrenamiento, interpolando entre situaciones conocidas. Mientras que Q-Learning requiere visitar explícitamente cada par estado-acción para asignarle un valor, una red neuronal puede inferir valores para estados similares basándose en los patrones aprendidos. Arquitecturas comunes incluyen redes multicapa perceptrón (MLP), que procesan directamente vectores de observación numéricos.

#### Action Masking

En dominios donde las acciones disponibles varían dinámicamente según el contexto, resulta fundamental incorporar un mecanismo de **enmascaramiento de acciones** (action masking). Variantes como **MaskablePPO** [2] incorporan nativamente esta funcionalidad: en cada paso de decisión, el entorno provee un vector booleano que indica qué acciones son legalmente válidas. El algoritmo utiliza esta información para:

- Asignar probabilidad cero a las acciones inválidas durante el muestreo de la política.
- Excluir las acciones ilegales del cálculo del gradiente, evitando que el modelo desperdicie capacidad aprendiendo a evitar movimientos imposibles.

Este mecanismo es especialmente relevante en juegos de cartas y otros dominios con reglas complejas, donde las acciones disponibles cambian drásticamente según el estado del juego.

---

## 3. Diseño experimental

### 3.1 Métricas

- **Win rate:** proporción de partidas ganadas sobre el total jugado. Se calcula como  
  **win rate = partidas ganadas / partidas totales**.  
  **Interpretación:** valores cercanos a 0.5 indican paridad con el rival; valores mayores a 0.5 reflejan ventaja sostenida.

- **Puntos obtenidos:** promedio de puntos ganados por partida (o por mano, si se normaliza). Se calcula como  
  **puntos promedio = puntos totales / partidas totales**.  
  **Interpretación:** permite comparar rendimiento cuando el win rate es similar; un mayor promedio indica que el agente no solo gana, sino que lo hace con mayor margen.

- **Porcentaje de mentiras:** proporción de cantos realizados con mano desfavorable según el siguiente criterio simplificado: para **Truco/Retruco/Vale Cuatro**, se considera **mentira** si el promedio de fuerza de la mano del que canta es menor que el promedio de fuerza de la mano del oponente (se calcula a partir del ranking de cartas y se compara el promedio). Para **Envido**, se considera **mentira** si el jugador que canta tiene **menos de 25 puntos de envido**. Se calcula como  
  **% mentiras = cantos con mano desfavorable / cantos totales**.  
  **Interpretación:** valores altos indican un estilo más bluff/agresivo; valores bajos reflejan un juego conservador. Se analiza junto al win rate para evaluar si la mentira es efectiva o simplemente riesgosa.

### 3.2 Herramientas

- **Lenguaje de programación:** Python 3.11+.  
  **Motivo:** ecosistema maduro para IA/RL y rápida iteración experimental.

- **Entorno propio:** implementación completa del juego y wrapper tipo Gymnasium en `game/truco_env.py`.  
  **Motivo:** se requiere modelar reglas y dinámicas específicas del Truco (1v1, sin flor), que no existen en entornos estándar.

- **Librerías (versions mínimas):**
  - **gymnasium** (>=1.0,<2): interfaz de entornos y compatibilidad con agentes RL.
  - **numpy** (>=1.24): operaciones numéricas y manejo de vectores/estadísticas.
  - **stable-baselines3** (>=2.0) y **sb3-contrib** (>=2.0): implementación de algoritmos modernos y utilidades adicionales.
  - **matplotlib** (>=3.7): generación de gráficos para análisis de resultados.
    **Motivo:** conjunto estándar en proyectos RL, con buen soporte y documentación.

- **Control de versiones:** Git (historial de commits).  
  **Motivo:** trazabilidad del desarrollo, colaboración y evaluación del proceso.

- **Graficadores:** Matplotlib (gráficos de métricas y resultados).  
  **Motivo:** visualización clara y reproducible de la evolución del rendimiento.

### 3.3 Agentes utilizados

Para evaluar el desempeño de los algoritmos de aprendizaje por refuerzo descritos en el marco teórico (secciones 2.3 y 2.4), se implementaron cuatro agentes con niveles crecientes de sofisticación:

- **Agente Random (baseline):** selecciona una acción válida al azar en cada turno, sin usar información histórica ni estrategia. Sirve como línea base mínima: si un agente entrenado no supera al azar, la señal de aprendizaje es débil o el modelado del entorno no está capturando adecuadamente la tarea.

- **Agente Rational (baseline basado en reglas):** aplica reglas determinísticas para decidir según la fuerza percibida de la mano (criterios para cantar, aceptar o jugar cartas). Ofrece una política coherente y explicable, útil como referencia más exigente que el azar. Permite evaluar si el aprendizaje automático capta patrones estratégicos que superen heurísticas diseñadas manualmente.

- **Agente Q-Learning (Monte Carlo):** implementación de Monte Carlo Q-Learning (sección 2.4) con discretización del espacio de estados y tabla de valores. Su naturaleza model-free permite aprender decisiones a partir de la interacción directa con el entorno, sin requerir un modelo explícito del rival ni de las transiciones.

- **Agente PPO (MaskablePPO):** implementación de PPO (sección 2.5) mediante la librería Stable-Baselines3 [3], con una red neuronal como aproximador de función y enmascaramiento nativo de acciones [2]. Su capacidad de generalización a estados no visitados y la estabilidad del entrenamiento lo hacen especialmente adecuado para un dominio con alto componente de incertidumbre.

### 3.4 Creación del entorno personalizado

El desarrollo del agente requirió la construcción de un entorno de simulación completo que captura las reglas, dinámicas y restricciones del Truco Argentino en modalidad uno contra uno, sin la mecánica de flor. Dado que no existe un entorno estándar disponible para este juego, se diseñó una implementación propia siguiendo la interfaz de Gymnasium [8], lo que garantiza compatibilidad con los algoritmos de aprendizaje por refuerzo más utilizados en la comunidad.

#### Representación del estado del juego

El estado interno completo del juego se modela mediante una estructura que encapsula toda la información necesaria para gestionar una partida. Sus componentes se organizan en los siguientes grupos:

**Cartas:**

- **Cartas en mano** de cada jugador.
- **Historial de cartas jugadas** durante la mano, con identificación de quién jugó cada una.

**Puntuación:**

- **Puntos acumulados** de cada jugador en la partida (0-30).
- **Contadores de rondas** ganadas por cada jugador y rondas empatadas dentro de la mano actual.
- **Historial de resultados** de cada ronda de la mano.

**Control de turno y flujo:**

- **Número de ronda** actual dentro de la mano (primera, segunda o tercera).
- **Turno actual**, que indica qué jugador debe actuar.
- **Condición de mano**, que señala si el jugador es mano, relevante para desempates y orden de juego.

**Estado del truco:**

- **Nivel de apuesta activo** (no cantado, truco, retruco o vale cuatro).
- **Estado del canto** de truco en curso y si se espera una respuesta.
- **Identificación** de quién cantó y quién aceptó el truco.

**Estado del envido:**

- **Estado del canto** de envido en curso y si se espera una respuesta.
- **Puntos acumulados** en la escalada de envido (actual y anterior).
- **Indicador de finalización** de la fase de envido.
- **Identificación** de quién inició el envido.

Este estado completo contiene información privilegiada (como las cartas del oponente) que no debe ser accesible para los agentes, lo que motiva la construcción de un espacio de observación parcial.

#### Espacio de observación

A partir del estado completo, se construye un **vector de observación de trece elementos** que codifica únicamente la información que un jugador legítimamente conocería en su turno, respetando el principio de información imperfecta:

| Posición | Contenido                   | Rango | Descripción                                                                       |
| -------- | --------------------------- | ----- | --------------------------------------------------------------------------------- |
| 1-3      | Cartas propias              | 0-14  | Ranking de fuerza de cada carta en mano (1=más fuerte, 14=más débil, 0=ya jugada) |
| 4-6      | Cartas del oponente en mesa | 0-14  | Ranking de las cartas que el oponente ha jugado (0=no jugada aún)                 |
| 7        | Puntos propios              | 0-30  | Puntos acumulados del agente                                                      |
| 8        | Puntos del oponente         | 0-30  | Puntos acumulados del rival                                                       |
| 9        | Número de ronda             | 1-3   | Ronda actual dentro de la mano                                                    |
| 10       | Turno                       | 0-1   | Indicador binario de si es turno del agente                                       |
| 11       | Nivel de truco              | 0-3   | Nivel de apuesta activo (0=ninguno, 1=truco, 2=retruco, 3=vale cuatro)            |
| 12       | Estado de envido            | 0-N   | Codificación del estado del canto de envido                                       |
| 13       | Es mano                     | 0-1   | Indicador de si el agente es mano                                                 |

#### Representación del mazo y jerarquía de cartas

El mazo del Truco Argentino se representa internamente mediante un conjunto de cuarenta cartas, cada una identificada por su valor numérico y su palo. Sin embargo, para el aprendizaje por refuerzo resulta más informativo transformar esta representación en un sistema de ranking que refleja directamente la fuerza relativa de cada carta en el enfrentamiento. Esta transformación evita que el agente deba aprender implícitamente la jerarquía no lineal del juego.

El ranking asigna valores del uno al catorce según la fuerza tradicional: el as de espadas ocupa el primer lugar, seguido del as de bastos, el siete de espadas, el siete de oros, y así sucesivamente hasta las cartas más débiles (los cuatros, cincos y seises). Las figuras ocupan posiciones intermedias. Esta codificación permite que el agente compare cartas directamente mediante sus valores numéricos, simplificando la evaluación de situaciones de combate.

Adicionalmente, cada carta posee un valor de envido independiente que se utiliza únicamente para el cálculo del tanto. Este valor corresponde al número literal de la carta para los naipes del uno al siete, mientras que las figuras (representadas internamente como diez, once y doce) contribuyen con cero puntos al envido. El cálculo del tanto sigue la regla tradicional: si el jugador posee dos o más cartas del mismo palo, suma veinte puntos de base más los valores de envido de las dos cartas de mayor valor en ese palo; en caso contrario, el tanto equivale al valor de envido de la carta individual más alta.

#### Espacio de acciones y restricciones dinámicas

El espacio de acciones se definió como un conjunto discreto de trece posibilidades que cubren todas las decisiones legales en el Truco: jugar la primera, segunda o tercera carta de la mano; cantar envido, envido-envido, real envido o falta envido; cantar truco, retruco o vale cuatro; aceptar una propuesta (quiero); rechazar una propuesta (no quiero); e ir al mazo (abandonar la mano). Esta enumeración exhaustiva permite modelar tanto las jugadas operativas como las apuestas y sus respuestas.

Un aspecto crítico de la implementación es el sistema de máscaras de acción (action masking), que restringe dinámicamente las acciones válidas según el contexto del juego. En cada turno, el entorno genera un vector booleano de trece posiciones que indica cuáles acciones son legalmente ejecutables. Este mecanismo es fundamental por varias razones:

Primero, evita que el agente proponga acciones imposibles según las reglas. Por ejemplo, no se puede cantar envido después de la primera ronda, no se puede jugar una carta que ya fue descartada, y no se puede cantar truco si ya se alcanzó el nivel máximo de apuesta.

Segundo, gestiona correctamente la precedencia de cantos y respuestas. Cuando un jugador canta truco, su oponente debe responder (aceptar, rechazar o re-cantar) antes de poder jugar cartas. De forma análoga, un canto de envido en la primera ronda tiene prioridad y debe resolverse antes de continuar con el juego de cartas. El envido incluso puede interrumpir un canto de truco si se realiza en el momento oportuno.

Tercero, controla las escaladas permitidas. El envido-envido solo puede cantarse como respuesta a un envido simple; el real envido puede cantarse directamente o como escalada desde cualquier nivel inferior de envido; la falta envido puede proponerse desde cualquier estado del envido. Para el truco, solo el jugador que aceptó el nivel anterior tiene derecho a elevar la apuesta.

Este sistema de máscaras garantiza que el aprendizaje se concentre exclusivamente en decisiones estratégicas válidas, evitando el desperdicio de capacidad del modelo en acciones ilegales y previniendo estados inconsistentes en la simulación.

#### Motor de reglas y gestión del flujo de juego

El corazón del entorno es un motor de lógica que implementa las reglas completas del Truco Argentino. Este componente gestiona el reparto de cartas (siguiendo una distribución equivalente al muestreo sin reemplazo de un mazo barajado), la alternancia de turnos, la resolución de enfrentamientos ronda por ronda, el cálculo de tantos, la acumulación de puntos y la determinación del ganador de cada mano y de la partida.

El flujo de una mano sigue la estructura tradicional: se reparten tres cartas a cada jugador, se determina quién es mano (alternando entre partidas), y se procede con la secuencia de jugadas. En la primera ronda, antes de que se jueguen cartas con el truco activo, se habilita la posibilidad de cantar envido y sus variantes. Una vez resuelta o descartada esta fase, los jugadores alternan jugando cartas, pudiendo en cualquier momento proponer truco (o sus escaladas) siempre que las condiciones lo permitan. Cada enfrentamiento de cartas determina un ganador de ronda (o empate, llamado parda), y la mano se resuelve cuando un jugador gana dos de las tres rondas, o cuando existen empates que según las reglas tradicionales definen un ganador.

La gestión de empates sigue las reglas tradicionales: si la primera ronda empata, gana la mano quien gane cualquiera de las siguientes; si la primera se gana y la segunda empata, gana el de la primera; si hay dos empates, gana quien gane la tercera; si hay tres empates consecutivos, gana el jugador mano. Estas reglas están implementadas exhaustivamente para garantizar la fidelidad del simulador al juego real.

Cuando un jugador rechaza un canto de truco, la mano termina inmediatamente y el oponente recibe los puntos correspondientes al nivel de truco vigente antes del rechazo. El ir al mazo tiene un efecto similar, otorgando puntos según el contexto (considerando si hubo cantos previos y si el envido ya se jugó).

#### Sistema de puntuación

El entorno implementa fielmente el sistema de puntuación del Truco Argentino. Al ganar una mano sin canto de truco, el ganador recibe un punto. Si se cantó truco y fue aceptado, el ganador obtiene dos puntos; con retruco aceptado son tres puntos, y con vale cuatro aceptado se otorgan cuatro puntos. Cuando un jugador rechaza un canto de truco, el oponente recibe los puntos correspondientes al nivel inmediatamente anterior (uno si rechaza el truco inicial, dos si rechaza el retruco, tres si rechaza el vale cuatro).

Los puntos del envido se asignan según la escalada de cantos: el envido simple vale dos puntos, el envido-envido acumula cuatro, el real envido otorga tres puntos (o cinco si se cantó sobre un envido previo), y la falta envido concede la diferencia entre treinta y los puntos del jugador que va ganando. Cuando un jugador rechaza un canto de envido, el oponente recibe los puntos del nivel anterior al rechazado (por ejemplo, un punto si rechaza un envido inicial, o dos si rechaza un real envido cantado sobre un envido simple).

La partida finaliza cuando algún jugador alcanza treinta puntos, momento en el cual se declara ganador.

#### Consideraciones de diseño

Varias decisiones de diseño merecen justificación explícita. La elección de un vector de observación compacto (trece dimensiones) responde a la necesidad de representar el estado del juego de forma eficiente sin sacrificar información estratégica crítica. Se excluyó deliberadamente el historial completo de cartas jugadas en rondas anteriores y se codificó únicamente la información de cartas visibles, siguiendo el principio de que la representación debe ser suficiente pero no redundante.

La separación entre la lógica del juego y la interfaz de entorno sigue un patrón de diseño modular que facilita el mantenimiento, las pruebas y la extensibilidad. El motor de reglas puede verificarse independientemente mediante partidas simuladas, mientras que el envoltorio de entorno se ocupa de la traducción entre el estado interno y las estructuras estándar requeridas por la interfaz Gymnasium.

La implementación del sistema de máscaras como un componente central (en lugar de un añadido opcional) refleja la naturaleza fuertemente restringida del Truco, donde las acciones válidas varían drásticamente según el contexto de la partida. Esta decisión garantiza que cualquier consulta al entorno sobre acciones disponibles devuelva información precisa y actualizada.

Finalmente, el diseño soporta simulación desde ambas perspectivas de jugador. El entorno puede reconfigurar qué jugador corresponde a cada participante mediante la especificación de un identificador, reordenando automáticamente las observaciones para mantener la consistencia del punto de vista.

### 3.5 Agentes y sus entrenamientos

#### 3.5.1 Q-Learning

##### Discretización del estado

El algoritmo Q-Learning requiere un espacio de estados discreto para construir su tabla de valores. Dado que el vector de observación original contiene valores continuos, se diseñó una función de codificación que transforma cada observación en una tupla discreta. Los componentes de esta codificación son:

- **Ranking de cartas propias:** se extraen los rankings de las cartas en mano (valores entre 1 y 14), se ordenan de menor a mayor y se completan con ceros si faltan cartas. Esto produce una tupla de tres elementos que representa la fuerza de la mano de forma ordenada.
- **Máximo ranking rival en mesa:** se identifica el ranking más alto entre las cartas que el oponente ha jugado. Un valor de cero indica que el rival aún no ha descartado ninguna carta.
- **Zona de puntos propia y rival:** en lugar de usar los puntos exactos (0-30), se discretiza en tres zonas: baja (0-15 puntos), media (16-25 puntos) y alta (26-30 puntos). Esta abstracción reduce la explosión combinatoria sin perder información estratégica relevante.
- **Indicador de ventaja:** un valor binario que señala si el agente va ganando o perdiendo en puntos.
- **Nivel de truco activo:** codificado como un entero que representa el nivel actual del canto de truco (sin canto, truco, retruco, vale cuatro).
- **Estado del envido:** un entero que indica el estado actual del canto de envido y si requiere respuesta.
- **Condición de mano:** un indicador binario que señala si el agente es mano en la ronda actual.
- **Número de ronda:** la ronda actual dentro de la mano (primera, segunda o tercera).

Esta codificación produce un espacio de estados manejable que captura las variables más relevantes para la toma de decisiones, evitando la explosión exponencial que ocurriría con una representación exhaustiva.

La implementación utiliza la variante Monte Carlo Q-Learning [1] descrita en la sección 2.4, donde cada **episodio** se define como una **mano completa** del Truco (desde el reparto de cartas hasta que un jugador gana la mano o se retira). La estrategia de exploración sigue el esquema epsilon-greedy con decaimiento cosenoidal detallado en la sección 2.4, aplicado específicamente con un ε inicial de 0.5.

##### Recompensas a la hora de entrenar

La recompensa utilizada para el entrenamiento se calcula al finalizar cada mano, tomando como base la diferencia de puntos obtenidos durante esa mano. Esta señal se construye de la siguiente manera:

- **Diferencia de puntos normalizada:** al concluir una mano, se computan los puntos ganados por cada jugador durante esa mano específica. La recompensa base se define como la diferencia entre los puntos del agente y los puntos del oponente, dividida por 30 (el puntaje máximo de una partida). Esta normalización produce valores en un rango aproximado de [-1, 1], facilitando la estabilidad del aprendizaje.

- **Penalización por retirarse:** si la última acción del agente fue ir al mazo (abandonar la mano), se aplica una penalización adicional de 0.1 sobre la recompensa calculada. Esta penalización desincentiva el abandono prematuro como estrategia frecuente.

- **Acotamiento del rango:** la recompensa final se limita al intervalo [-1, 1] para evitar valores extremos que podrían desestabilizar la actualización de los valores Q.

- **Inversión por perspectiva:** dado que el entrenamiento se realiza en modalidad self-play donde ambos jugadores aprenden simultáneamente, la recompensa se invierte según la perspectiva del jugador que realizó cada acción. Por ejemplo, si en una mano el Jugador 0 gana 2 puntos de truco y el Jugador 1 gana 1 punto de envido, la recompensa para el Jugador 0 es (2−1)/30 = +0.033, mientras que para el Jugador 1 es (1−2)/30 = −0.033. De esta forma, cada transición almacenada durante el episodio se actualiza con la recompensa correspondiente a la perspectiva del jugador que tomó esa decisión, reflejando la naturaleza de suma cero del juego.

##### Hiperparámetros del entrenamiento

Los valores utilizados para el entrenamiento del agente Q-Learning fueron:

- **α (alpha) = 0.1:** tasa de aprendizaje que determina la proporción en que se actualiza el valor Q con cada nueva experiencia. Un valor de 0.1 permite una actualización moderada que balancea la incorporación de nueva información con la retención del conocimiento previo.
- **γ (gamma) = 1.0:** factor de descuento que indica cuánto se valora el resultado futuro respecto al inmediato. Un valor de 1.0 significa que el resultado final de la mano tiene la misma importancia para todas las decisiones de la secuencia, sin atenuación temporal.
- **ε (epsilon) inicial = 0.5:** probabilidad inicial de exploración. Con este valor, el agente comienza eligiendo acciones aleatorias la mitad de las veces, lo que favorece el descubrimiento de estados y acciones diversos antes de explotar el conocimiento adquirido.
- **Decaimiento cosenoidal de ε:** la probabilidad de exploración decrece a lo largo del entrenamiento siguiendo la fórmula **ε(t) = ε₀ × cos(tπ / 2T)**, donde t es el episodio actual y T el número total de episodios. Esta función produce una curva suave que mantiene alta exploración en las primeras etapas (cuando el coseno está cerca de 1) y la reduce gradualmente hasta llegar a cero al final del entrenamiento (cuando el coseno alcanza 0). A diferencia del decaimiento lineal o exponencial, el decaimiento cosenoidal ofrece una transición más gradual en las fases intermedias, evitando reducciones bruscas de la exploración.

##### Proceso de entrenamiento

El entrenamiento del agente Q-Learning se realizó en múltiples fases iterativas, ajustando el enfoque según los comportamientos observados.

En la fase inicial, el agente fue entrenado exclusivamente mediante self-play. Sin embargo, se identificó un problema de convergencia prematura: durante las primeras iteraciones, donde la política es esencialmente aleatoria debido a la alta tasa de exploración, el agente acumulaba experiencias negativas frecuentes. Esto condujo a que el algoritmo convergiera hacia un mínimo local subóptimo caracterizado por una estrategia de evasión al riesgo. Concretamente, el agente aprendió a minimizar pérdidas retirándose prematuramente de las manos (ir al mazo), evitando así enfrentamientos que percibía como inciertos. Este comportamiento, aunque reduce las pérdidas inmediatas, impide desarrollar una estrategia competitiva efectiva.

Para corregir este fenómeno, se introdujo una penalización adicional sobre la acción de retirarse, descontando 0.1 puntos de la recompensa cada vez que el agente optaba por ir al mazo. Esta modificación alteró el balance de incentivos, haciendo que el abandono prematuro dejara de ser una estrategia dominante desde la perspectiva del aprendizaje.

Tras aplicar esta corrección, el agente fue reentrenado en self-play con resultados más prometedores. No obstante, el análisis del comportamiento emergente reveló un sesgo hacia un estilo de juego excesivamente agresivo. El agente tendía a cantar truco y envido con frecuencia desproporcionada, asumiendo riesgos que no siempre estaban justificados por la calidad de su mano. Esta característica, aunque puede ser efectiva contra oponentes pasivos, no representa una estrategia equilibrada.

Para moderar este comportamiento, se complementó el entrenamiento mediante partidas adicionales contra el agente racional. Este oponente, basado en heurísticas diseñadas manualmente, exhibe un estilo de juego más conservador y predecible. La exposición a este tipo de rival permitió que el agente Q-Learning ajustara su política, aprendiendo cuándo la agresividad es penalizada por un oponente que responde de manera consistente. El resultado fue un agente con un repertorio estratégico más diverso, capaz de alternar entre estilos según el contexto de la partida.

En términos cuantitativos, el entrenamiento consistió en un total de 20 millones de partidas en modalidad self-play, seguidas de 1 millón de partidas adicionales contra el agente racional.

#### 3.5.2 PPO (Proximal Policy Optimization)

La implementación se realizó utilizando la librería **Stable-Baselines3** [3], específicamente el módulo **MaskablePPO** de **sb3-contrib**, que proporciona una implementación del algoritmo PPO (sección 2.5) con soporte nativo para enmascaramiento de acciones (sección 2.5, Action Masking). La arquitectura utilizada es una red MLP que procesa directamente el vector de observación de trece dimensiones definido por el entorno base.

##### Wrapper de entorno para entrenamiento

El entrenamiento del agente PPO requirió la construcción de un envoltorio especializado sobre el entorno base. Este wrapper transforma el entorno de dos jugadores en uno de agente único, donde el modelo controla exclusivamente al jugador principal mientras un oponente configurable juega automáticamente el rol del rival.

Durante cada paso del entorno, si el turno corresponde al oponente, el wrapper ejecuta automáticamente sus acciones utilizando el agente oponente configurado (que puede ser aleatorio, racional, o una versión anterior del propio modelo en el caso de self-play). Esta abstracción permite que el algoritmo PPO interactúe con el entorno como si fuera un problema de decisión de agente único, simplificando significativamente el proceso de entrenamiento.

A diferencia de Q-Learning, que en self-play entrena ambos jugadores simultáneamente con recompensas invertidas, el agente PPO opera exclusivamente desde la perspectiva del Jugador 0. **El oponente no recibe recompensas ni aprende durante el entrenamiento de PPO**: es un agente externo fijo (aleatorio, racional, o un modelo previo) que simplemente ejecuta acciones cuando le corresponde el turno.

Las recompensas que recibe el agente PPO provienen del entorno y siempre están expresadas desde su perspectiva. Cuando el oponente realiza acciones durante su turno automático (gestionado por el wrapper), estas acciones pueden generar cambios en los puntos de la partida. El wrapper acumula las recompensas resultantes de los turnos del oponente y las suma a la recompensa del siguiente paso del agente. Por ejemplo, si el oponente canta envido y gana 2 puntos, esa pérdida se refleja como una recompensa negativa para el agente PPO. De esta forma, el agente percibe el efecto completo de las acciones del rival como parte de su propia señal de aprendizaje, sin necesidad de modelar explícitamente al oponente.

##### Hiperparámetros del entrenamiento

El entrenamiento del agente PPO utilizó los valores por defecto proporcionados por la implementación MaskablePPO de la librería sb3-contrib.

- **Learning rate = 3×10⁻⁴ - 1.5×10⁻⁴:** tasa de aprendizaje que controla la magnitud de las actualizaciones de los parámetros de la red neuronal en cada paso de optimización. Se entreno en diferentes etapas, empezando en 3x10⁻⁴ y bajandolo a 1.5x10⁻⁴ en las siguientes, para no olvidar lo aprendido en la primera.
- **n_steps = 2048:** número de pasos de interacción con el entorno que se recolectan antes de realizar una actualización de la política. Este valor determina el tamaño del buffer de experiencias.
- **batch_size = 64:** tamaño de los mini-lotes utilizados durante la optimización. Los 2048 pasos se dividen en lotes de 64 para calcular los gradientes.
- **n_epochs = 10:** número de épocas de optimización sobre los datos recolectados en cada actualización. La política se entrena 10 veces sobre el mismo conjunto de experiencias antes de recolectar nuevos datos.
- **gamma (γ) = 0.99:** factor de descuento que pondera la importancia de las recompensas futuras respecto a las inmediatas. Un valor cercano a 1 indica que el agente considera relevantes las consecuencias a largo plazo.
- **gae_lambda (λ) = 0.95:** parámetro de la Estimación de Ventaja Generalizada (GAE), que balancea el sesgo y la varianza en la estimación de la función de ventaja.
- **clip_range = 0.2:** rango de recorte para la función objetivo de PPO. Limita cuánto puede cambiar la razón entre la política nueva y la anterior, previniendo actualizaciones demasiado agresivas.
- **ent_coef = 0.0:** coeficiente de entropía que incentiva la exploración. Un valor de cero indica que no se añade bonificación por entropía en la función objetivo.
- **vf_coef = 0.5:** coeficiente que pondera la pérdida de la función de valor en la función objetivo total.
- **max_grad_norm = 0.5:** norma máxima para el recorte de gradientes, que previene actualizaciones inestables cuando los gradientes son muy grandes.
- **Arquitectura MLP:** red neuronal con política por defecto de Stable Baselines3, consistente en dos capas ocultas de 64 neuronas cada una con activación ReLU, tanto para la red de política como para la red de valor.

##### Proceso de entrenamiento

Durante el proceso de entrenamiento del agente PPO se observó un fenómeno de adaptación excesiva al estilo de juego del oponente utilizado. Este comportamiento, aunque esperable desde la perspectiva de la optimización, resultaba problemático para obtener un agente con estrategia equilibrada.

Cuando el entrenamiento se realizó exclusivamente en modalidad self-play (partiendo de un oponente aleatorio), el agente desarrolló un estilo de juego marcadamente agresivo. La dinámica emergente favorecía estrategias de mentira frecuente: el agente aprendió a cantar truco y envido de manera insistente, apostando a que el oponente eventualmente rechazaría las propuestas. Este comportamiento, aunque efectivo contra rivales que no discriminan bien cuándo aceptar o rechazar, no representa una estrategia robusta.

Por otro lado, cuando el entrenamiento se realizó únicamente contra el agente racional, el resultado fue el extremo opuesto: un estilo excesivamente conservador. Dado que el agente racional nunca miente (solo canta cuando su mano lo justifica según sus reglas), el agente PPO aprendió a rechazar sistemáticamente las apuestas del rival, interpretando cualquier canto como señal de mano fuerte. Esta política, aunque segura, desaprovecha oportunidades de ganancia y resulta predecible.

Para obtener una estrategia intermedia, se adoptó un enfoque de entrenamiento en dos fases. Primero, el agente fue entrenado mediante self-play para desarrollar una base estratégica que incluyera el uso de la mentira como herramienta válida. Posteriormente, se continuó el entrenamiento contra el agente racional, pero con una tasa de aprendizaje reducida y el mismo número de iteraciones. Esta segunda fase permitió moderar los excesos de la estrategia agresiva sin eliminar completamente la capacidad de mentir, resultando en un agente con un repertorio más balanceado: capaz de mentir cuando la situación lo amerita, pero también de reconocer cuándo la prudencia es preferible.

El entrenamiento total comprendió 50,000 timesteps en modalidad self-play (partiendo de un oponente aleatorio y tasa de aprendizaje de 3e-4), seguidos de tres sesiones de 20,000 timesteps cada una contra el agente racional (tasa de aprendizaje de 1.5e-4). Los timesteps corresponden a pasos de interacción con el entorno, donde cada partida completa involucra múltiples pasos. Este esquema progresivo permitió alcanzar una estrategia moderada que equilibra ambos estilos de juego.

### 3.6 Experimentos y resultados

Para evaluar el desempeño de los agentes desarrollados, se realizaron enfrentamientos de 1000 partidas entre cada par de agentes. A continuación se presentan los resultados organizados por agente evaluado.

#### 3.6.1 Agente Random

El agente aleatorio sirve como línea base para comparar el desempeño de los demás agentes.

| Oponente   | Win Rate | Puntos Promedio | % Mentiras Truco | % Mentiras Envido |
| ---------- | -------- | --------------- | ---------------- | ----------------- |
| Rational   | 5.5%     | 8.46            | 45%              | 64%               |
| Q-Learning | 23.2%    | 19.22           | 48%              | 62%               |
| PPO        | 50.7%    | 25.75           | 45%              | 62%               |

#### 3.6.2 Agente Racional

El agente racional implementa heurísticas diseñadas manualmente basadas en reglas del juego.

| Oponente   | Win Rate | Puntos Promedio | % Mentiras Truco | % Mentiras Envido |
| ---------- | -------- | --------------- | ---------------- | ----------------- |
| Random     | 94.5%    | 28.95           | 12%              | 0%                |
| Q-Learning | 57.9%    | 27.16           | 8%               | 0%                |
| PPO        | 51.6%    | 26.83           | 10%              | 0%                |

#### 3.6.3 Agente Q-Learning

El agente entrenado mediante Monte Carlo Q-Learning en self-play y contra el agente racional.

| Oponente | Win Rate | Puntos Promedio | % Mentiras Truco | % Mentiras Envido |
| -------- | -------- | --------------- | ---------------- | ----------------- |
| Random   | 77.7%    | 26.52           | 33%              | 56%               |
| Rational | 43.0%    | 25.37           | 31%              | 55%               |
| PPO      | 44.0%    | 24.15           | 35%              | 54%               |

#### 3.6.4 Agente PPO

El agente entrenado mediante Proximal Policy Optimization con action masking.

| Oponente   | Win Rate | Puntos Promedio | % Mentiras Truco | % Mentiras Envido |
| ---------- | -------- | --------------- | ---------------- | ----------------- |
| Random     | 49.5%    | 25.31           | 49%              | 62%               |
| Rational   | 47.8%    | 26.05           | 45%              | 62%               |
| Q-Learning | 56.0%    | 27.09           | 50%              | 62%               |

### 3.7 Visualización de resultados

Los siguientes gráficos de violín muestran la distribución de puntos obtenidos por partida para cada enfrentamiento entre agentes. Cada violin representa la densidad de probabilidad de los puntajes finales, permitiendo observar no solo la tendencia central sino también la dispersión y la forma de la distribución. Los puntos individuales superpuestos corresponden a cada una de las 1000 partidas simuladas.

![](/game/plots/images/randomvsrational_violin.png)

_Figura 1: Distribución de puntos por partida entre Random y Rational. Se observa la clara superioridad del agente Rational, con una distribución concentrada en valores altos._

![](/game/plots/images/randomvsq_learning_violin.png)

_Figura 2: Distribución de puntos por partida entre Random y Q-Learning. El agente Q-Learning muestra una distribución favorable con mayor concentración en puntajes altos._

![](/game/plots/images/randomvssb3_violin.png)

_Figura 3: Distribución de puntos por partida entre Random y PPO. Ambas distribuciones presentan solapamiento considerable, reflejando el rendimiento similar observado en el win rate._

![](/game/plots/images/rationalvsq_learning_violin.png)

_Figura 4: Distribución de puntos por partida entre Rational y Q-Learning. Se observa un enfrentamiento más equilibrado con distribuciones que se superponen significativamente._

![](/game/plots/images/rationalvssb3_violin.png)

_Figura 5: Distribución de puntos por partida entre Rational y PPO. Enfrentamiento muy parejo con distribuciones prácticamente simétricas._

![](/game/plots/images/q_learningvssb3_violin.png)

_Figura 6: Distribución de puntos por partida entre Q-Learning y PPO. Se observa una leve ventaja del agente PPO con distribución ligeramente desplazada hacia valores más altos._

#### Fuentes de puntos por agente

Los siguientes gráficos de área apilada muestran el desglose de las fuentes de puntos obtenidos por los agentes de aprendizaje por refuerzo a lo largo de múltiples partidas. Las áreas representan la contribución de cada fuente: **Envido** (puntos ganados por cantos de envido), **Truco** (puntos ganados en manos donde se cantó truco), **Cartas** (puntos ganados en manos sin canto de truco), y **Abandono** (puntos ganados cuando el oponente se retira o rechaza un canto). Esta visualización permite observar cómo cada agente explota las diferentes mecánicas del juego para acumular puntos.

Para estos gráficos se utilizó una muestra de 100 partidas en lugar de las 1000 empleadas en las demás métricas. Esta decisión responde a criterios de legibilidad: al incrementar el número de partidas representadas, las áreas apiladas se comprimen y dificultan la apreciación de las proporciones relativas de cada fuente de puntos, perdiendo valor informativo. La muestra reducida preserva la claridad visual sin comprometer la representatividad de los patrones observados.

![](/game/plots/images/q_learning_income_sources_area.png)

_Figura 7: Fuentes de puntos del agente Q-Learning contra diferentes oponentes. Se observa una distribución equilibrada entre las distintas fuentes, con contribución significativa del truco y el envido._

![](/game/plots/images/sb3_income_sources_area.png)

_Figura 8: Fuentes de puntos del agente PPO contra diferentes oponentes. El agente muestra mayor variabilidad en la composición de sus fuentes de puntos entre partidas._

#### Comparación de tasas de mentira

El siguiente gráfico de barras compara las tasas de mentira de todos los agentes, calculadas a partir de los enfrentamientos entre ellos. Para cada agente se muestra el porcentaje de cantos realizados con mano desfavorable, tanto para **Truco** (cuando el promedio de fuerza de la mano es menor que la del oponente) como para **Envido** (cuando los puntos de envido son menores a 25). Esta métrica permite caracterizar el estilo de juego de cada agente en términos de agresividad y uso del mentiras.

![](/game/plots/images/bluff_rate_multi.png)

_Figura 9: Comparación de tasas de mentira por agente. El agente Rational presenta las tasas más bajas (nunca miente en envido), mientras que PPO y Random muestran las tasas más altas. Q-Learning presenta un comportamiento intermedio, resultado de su entrenamiento mixto._

#### Matrices de enfrentamientos

Las siguientes matrices de calor (heatmaps) resumen el rendimiento de cada agente contra todos los demás en un formato compacto. Cada celda representa el resultado del agente de la fila contra el agente de la columna.

El primer heatmap muestra el **win rate** (proporción de victorias) de cada enfrentamiento. Colores verdes indican un win rate alto (ventaja del agente de la fila), mientras que colores rojos indican un win rate bajo (desventaja).

![](/game/plots/images/matchup_heatmap_win_rate.png)

_Figura 10: Matriz de win rate entre todos los agentes. Se observa que Rational domina contra Random (0.94), mientras que los enfrentamientos entre los agentes de RL y Rational son más equilibrados._

El segundo heatmap muestra la **diferencia promedio de puntos** por partida. Valores positivos (rojo) indican que el agente de la fila obtiene en promedio más puntos que su oponente; valores negativos (azul) indican lo contrario.

![](/game/plots/images/matchup_heatmap_avg_diff.png)

_Figura 11: Matriz de diferencia promedio de puntos entre todos los agentes. Esta métrica complementa el win rate mostrando la magnitud de la ventaja o desventaja en cada enfrentamiento._

---

## 4. Análisis y discusión de resultados

A continuación se analiza el desempeño de cada agente en función de las tres métricas definidas: win rate, puntos promedio y porcentaje de mentiras.

### 4.1 Agente Random

El agente aleatorio sirve como línea base fundamental para evaluar el desempeño de los demás agentes. Su comportamiento, al seleccionar acciones uniformemente entre las opciones válidas, representa el rendimiento mínimo esperable sin ningún tipo de estrategia.

#### Win rate

El win rate del agente Random revela una clara jerarquía entre los oponentes. Contra el agente Rational obtiene apenas un 5.5% de victorias, lo que demuestra la efectividad de las heurísticas diseñadas manualmente frente a un comportamiento puramente aleatorio. Contra Q-Learning el win rate mejora a 23.2%, indicando que aunque Q-Learning aprendió estrategias superiores al azar, no logra la consistencia del agente Rational. El resultado más llamativo es contra PPO, donde Random alcanza un 50.7% de victorias, prácticamente paridad. Este resultado sugiere que el agente PPO genero una politica la cual se ve parcialmente vulnerable a comportamientos impredecibles (ver Figura 3).

#### Puntos promedio

Los puntos promedio por partida confirman la interpretación del win rate. Contra Rational, el agente Random obtiene solo 8.46 puntos promedio, siendo ampliamente superado. Contra Q-Learning mejora a 19.22 puntos, y contra PPO alcanza 25.75 puntos (cercano al umbral de victoria de 30). Esta progresión sugiere que los agentes de aprendizaje por refuerzo manejan de manera diferente la incertidumbre: mientras Q-Learning desarrolló cierta robustez, PPO se ve más vulnerable a la variabilidad introducida por un oponente sin patrón predecible.

#### Porcentaje de mentiras

El agente Random presenta tasas de mentira del 45-48% en Truco y 62-64% en Envido, valores consistentes que reflejan la naturaleza probabilística de sus decisiones totalmente aleatorias, sin ningun patron detectable. Estas tasas sirven como referencia: cualquier agente con tasas similares pero sin mejora en win rate estaría efectivamente comportándose como aleatorio en términos de estrategia de canto.

#### Conclusión

El agente Random cumple su rol como baseline al ser consistentemente superado por agentes con estrategia, pero expone una peculiaridad interesante en el agente PPO: la dificultad para explotar oponentes sin patrones detectables.

### 4.2 Agente Racional

El agente Racional implementa heurísticas diseñadas manualmente basadas en las reglas del Truco Argentino. Su estrategia es determinística y conservadora: solo canta cuando su mano lo justifica según umbrales predefinidos, y nunca miente en el envido.

#### Win rate

El agente Racional demuestra un rendimiento dominante contra el agente Random con un 94.5% de victorias, validando la efectividad de una estrategia básica pero coherente frente al azar. Contra los agentes de aprendizaje por refuerzo, el rendimiento es más equilibrado pero mantiene ventaja: 57.9% contra Q-Learning y 51.6% contra PPO. Estos resultados son notables considerando que el agente Racional no aprende ni se adapta, sugiriendo que las heurísticas manuales capturan aspectos fundamentales del juego que los agentes de RL no lograron superar consistentemente (ver Figuras 4 y 5).

#### Puntos promedio

Los puntos promedio reflejan la consistencia del agente Racional. Contra Random obtiene 28.95 puntos (cercano al máximo de 30), demostrando eficiencia en cerrar partidas rápidamente. Contra Q-Learning y PPO los promedios son 27.16 y 26.83 respectivamente, valores altos que indican partidas competitivas donde el Racional frecuentemente alcanza o se acerca al objetivo de 30 puntos antes que su oponente.

#### Porcentaje de mentiras

El aspecto más distintivo del agente Racional es su comportamiento honesto: presenta tasas de mentira del 8-12% en Truco y exactamente 0% en Envido. El bajo porcentaje en Truco corresponde a situaciones límite donde las reglas permiten cantar, pero la mano resulta ser ligeramente inferior a la del oponente. La ausencia total de mentiras en Envido es por diseño: el agente solo canta envido cuando tiene 25 puntos o más, eliminando cualquier posibilidad de mentira según la definición de la métrica.

#### Conclusión

El agente Racional establece un estándar sólido que los agentes de RL no lograron superar de manera consistente. Su éxito radica en la explotación sistemática de las reglas del juego sin asumir riesgos innecesarios. Sin embargo, su predictibilidad lo hace vulnerable a oponentes humanos que podrían aprender a explotarlo, ventaja que los agentes de RL implementados no capitalizaron completamente. El contraste entre su honestidad absoluta en envido y las altas tasas de mentira de los otros agentes ilustra dos filosofías de juego opuestas.

### 4.3 Agente Q-Learning

El agente Q-Learning representa el primer enfoque de aprendizaje por refuerzo implementado, utilizando una tabla de valores Q con discretización del espacio de estados. Su entrenamiento combinó 20 millones de partidas en self-play seguidas de 1 millón contra el agente Racional.

#### Win rate

El agente Q-Learning muestra un rendimiento asimétrico según el oponente. Contra Random obtiene un sólido 77.7% de victorias, demostrando que efectivamente aprendió estrategias superiores al azar. Sin embargo, contra Rational y PPO el rendimiento cae a 43.0% y 44.0% respectivamente, quedando por debajo del 50%. Esto indica que el agente desarrolló competencias suficientes para explotar comportamientos aleatorios pero frente a estrategias claras consigue dar algo de pelea.(ver Figuras 2 y 4).

#### Puntos promedio

Los puntos promedio siguen el patrón del win rate: 26.52 contra Random (competitivo), 25.37 contra Rational y 24.15 contra PPO (ambos por debajo del oponente). El análisis de las fuentes de puntos (Figura 7) revela una distribución equilibrada entre envido, truco, cartas y puntos por abandono del rival, sugiriendo que el agente no especializó su estrategia en una mecánica particular sino que desarrolló un enfoque para aprovechar cada mecanica del juego.

#### Porcentaje de mentiras

Q-Learning presenta tasas de mentira del 31-35% en Truco y 54-56% en Envido, valores intermedios entre Random (~45%/~62%) y Rational (~10%/0%). Este posicionamiento intermedio es resultado directo del proceso de entrenamiento: el self-play inicial generó una tendencia agresiva que fue moderada por el entrenamiento posterior contra el agente Racional. El agente aprendió que mentir tiene un costo cuando el oponente responde de manera consistente, pero no eliminó completamente esta herramienta de su repertorio.

#### Conclusión

El agente Q-Learning logró aprender estrategias funcionales pero no alcanzó un nivel competitivo contra oponentes con estrategia. La discretización del espacio de estados, aunque permite manejar la tabla Q, puede haber limitado la capacidad del agente para capturar matices importantes del juego. Su tasa de mentira intermedia refleja el proceso de entrenamiento mixto y representa un equilibrio entre agresividad y prudencia, el cual fue capaz de darle cara al los demas agentes pero no logro superarlos.

### 4.4 Agente PPO

El agente PPO (Proximal Policy Optimization) representa el segundo enfoque de aprendizaje por refuerzo, utilizando una red neuronal como aproximador de función. Su entrenamiento comprendió 50,000 timesteps en self-play seguidos de tres sesiones de 20,000 timesteps contra el agente Racional con tasa de aprendizaje reducida.

#### Win rate

El agente PPO presenta el rendimiento más equilibrado pero también el más modesto entre los agentes de RL. Contra Random obtiene apenas 49.5% de victorias (prácticamente paridad), contra Rational 47.8%, y contra Q-Learning 56.0%. El único enfrentamiento donde muestra ventaja clara es contra Q-Learning. La paridad contra Random es el resultado más llamativo y sugiere que la política aprendida no logró generalizar bien ante comportamientos sin patrón (ver Figuras 3, 5 y 6).

#### Puntos promedio

Los puntos promedio son consistentes alrededor de 25-27 puntos independientemente del oponente: 25.31 contra Random, 26.05 contra Rational y 27.09 contra Q-Learning. Esta consistencia, combinada con win rates cercanos al 50%, indica partidas cerradas donde ambos jugadores se acercan al objetivo de 30 puntos. El análisis de fuentes de puntos (Figura 8) muestra como explota mas la mecanica del envido comparado con Q-Learning, sugiriendo una estrategia diferente a la hora de jugar.

#### Porcentaje de mentiras

PPO presenta las tasas de mentira más altas entre los agentes con estrategia: 45-50% en Truco y 62% en Envido, valores cercanos a los del agente Random. Esto indica que el entrenamiento genero una estrategia basada en la mentira obteniendo resultados ampliamente mejores a los del Random. Esto como dijimos antes se pudo haber reducido, ya que permitía una alta convergencia al estilo de juego del rival, pero se decidió que un punto medio es algo óptimo para explotar ambas características.

#### Conclusión

El agente PPO, a pesar de utilizar una arquitectura más sofisticada (red neuronal vs tabla Q), se mantiene parejo con todos sus oponentes. Su elevada tasa de mentira, la paridad frente al agente Random y el hecho de ser el que mayor resistencia opuso al agente Racional sugieren que posee la capacidad de competir con cualquier oponente. Además, al tratarse de una política no determinística, resulta el agente menos predecible de todos.

---

## 5. Conclusiones finales

El desarrollo de agentes de inteligencia artificial para el Truco Argentino representa un desafío que trasciende la mera implementación de algoritmos de aprendizaje por refuerzo. El juego combina información imperfecta, estocasticidad en el reparto de cartas, y un componente psicológico fundamental: la mentira como herramienta estratégica válida. Esta combinación genera un escenario donde no existe una estrategia óptima universal.

### Sobre la naturaleza estocástica del juego

A diferencia de juegos determinísticos como el ajedrez o el Go, el Truco Argentino introduce variabilidad irreducible desde el momento del reparto. Un agente puede ejecutar la decisión teóricamente correcta y aun así perder debido a la distribución de cartas. Esta característica implica que nunca se conseguirá un win rate absoluto: incluso el mejor agente posible perdería partidas por factores fuera de su control. Los resultados experimentales confirman esta realidad: ningún agente logró dominar consistentemente a todos los demás, y los enfrentamientos entre agentes de RL y el agente Racional se mantuvieron cercanos al 50%.

La mentira añade otra capa de complejidad. Un canto de truco o envido no transmite información verdadera sobre la mano del jugador, sino que es una apuesta estratégica que debe evaluarse considerando el historial del oponente, el contexto de la partida, y la tolerancia al riesgo. Los agentes desarrollados mostraron diferentes filosofías respecto a esta herramienta: mientras el Racional la evita casi completamente, PPO la utiliza con frecuencia similar al azar, y Q-Learning encontró un punto intermedio.

### Sobre la evaluación de agentes

Una conclusión importante de este trabajo es que no existe mejor forma de evaluar el comportamiento de un agente que jugando contra él. Las métricas cuantitativas (win rate, puntos promedio, tasa de mentira) proporcionan una caracterización útil pero incompleta. Cada oponente es un mundo: el agente que domina contra Random puede ser vulnerable ante Rational, y viceversa. Los gráficos de violin (Figuras 1-6) ilustran esta variabilidad, mostrando distribuciones de resultados que no pueden resumirse en un único número.

Esta observación tiene implicaciones prácticas: un agente de Truco no puede evaluarse de forma aislada. Su rendimiento depende fundamentalmente de las características del oponente, y la verdadera prueba de un agente es su capacidad de adaptarse o al menos mantener competitividad ante estilos de juego diversos.

### Logros y limitaciones

Se logró implementar un entorno de simulación completo del Truco Argentino, compatible con Gymnasium y extensible para futuras investigaciones. Los agentes de aprendizaje por refuerzo (Q-Learning y PPO) demostraron capacidad de aprender estrategias funcionales, aunque ninguno superó consistentemente al agente basado en heurísticas manuales. Esto sugiere que las reglas del juego, cuando se aplican de manera consistente, siempre sera fuerte contra políticas que no aprenden del juego constantemente.

### Trabajo futuro

#### Agente adaptativo con múltiples políticas

Una línea de mejora prometedora consiste en desarrollar un agente que no dependa de una única política fija, sino de un conjunto de políticas especializadas para diferentes estilos de oponente. Este agente incorporaría un motor de clasificación basado en aprendizaje automático que recolecte datos durante la partida (frecuencia de cantos, tasas de aceptación, patrones de juego de cartas) y estime el estilo del contrincante en tiempo real. Con esta clasificación, el agente seleccionaría dinámicamente la política más adecuada: una conservadora contra oponentes agresivos, una agresiva contra oponentes pasivos, etc. Este enfoque reconoce que cada jugador es un mundo y que la adaptación durante la partida es clave para el éxito en el Truco.

#### Algoritmos de minimización de arrepentimiento contrafactual (CFR)

Los algoritmos de la familia CFR [4] (Counterfactual Regret Minimization) representan el estado del arte para juegos de información imperfecta y suma cero. A diferencia de los métodos de aprendizaje por refuerzo tradicionales, CFR converge hacia un equilibrio de Nash aproximado, garantizando una estrategia no explotable a largo plazo. Esta familia de algoritmos ha demostrado resultados sobresalientes en dominios similares al Truco:

- **Libratus** (Carnegie Mellon University, 2017): Derrotó a jugadores profesionales de póker heads-up no-limit Texas Hold'em utilizando Monte Carlo CFR (MCCFR) y CFR+ [7].
- **Pluribus** (Facebook AI Research & CMU, 2019): Extendió el éxito a póker de seis jugadores, demostrando la escalabilidad del enfoque [6].

Durante el desarrollo de este proyecto se intentó implementar diversas variantes de CFR a traves de la libreria OpenSpiel [5] para el Truco Argentino:

- **Vanilla CFR:** La versión original requiere recorrer el árbol de juego completo en cada iteración, resultando computacionalmente inviable para el tamaño del espacio de estados del Truco.
- **Outcome Sampling CFR:** Variante más ligera que muestrea trayectorias en lugar de recorrer todo el árbol. A pesar de reducir significativamente los requisitos computacionales, no logró converger hacia una estrategia óptima con los recursos disponibles.
- **Deep CFR:** Combina CFR con redes neuronales para aproximar los valores de arrepentimiento. Los requerimientos de memoria y tiempo de entrenamiento excedieron la capacidad temporal del proyecto.
- **NeuRD (Neural Replicator Dynamics):** Enfoque alternativo basado en dinámica de replicadores. Presentó problemas de estabilidad numérica y tampoco convergió satisfactoriamente.

La conclusión es que, aunque CFR es teóricamente el enfoque más adecuado para el Truco, su implementación práctica requiere recursos computacionales sustancialmente mayores a los disponibles para este proyecto. Trabajo futuro podría explorar implementaciones optimizadas, abstracciones del espacio de estados, o acceso a infraestructura de cómputo de alto rendimiento.

---

## Nota sobre el uso de herramientas de IA

La redacción y estructuración de este documento fue asistida por el modelo de lenguaje Claude Opus 4.5 (Anthropic, 2025).

---

## 6. Referencias

[1] Wang, H., Emmerich, M., & Plaat, A. (2018). _Monte Carlo Q-learning for General Game Playing_. arXiv preprint arXiv:1802.05944. Recuperado de https://arxiv.org/abs/1802.05944

[2] Huang, S., & Ontañón, S. (2022). _A Closer Look at Invalid Action Masking in Policy Gradient Algorithms_. The International FLAIRS Conference Proceedings, 35. Recuperado de https://arxiv.org/abs/2006.14171

[3] Stable-Baselines3 Contributors. (2024). _MaskablePPO Documentation_. sb3-contrib. Recuperado el 2 de febrero de 2026, de https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

[4] Neller, T. W., & Lanctot, M. (2013). _An Introduction to Counterfactual Regret Minimization_. Technical Report.

[5] DeepMind. (2024). _OpenSpiel: A Framework for Reinforcement Learning in Games_. Recuperado el 3 de febrero de 2026, de https://openspiel.readthedocs.io/en/latest/index.html

[6] Brown, N., & Sandholm, T. (2019). _Superhuman AI for multiplayer poker_. Science, 365(6456), 885-890. DOI: 10.1126/science.aay2400

[7] Brown, N., & Sandholm, T. (2018). _Superhuman AI for heads-up no-limit poker: Libratus beats top professionals_. Science, 359(6374), 418-424. DOI: 10.1126/science.aao1733

[8] Farama Foundation. (2024). _Gymnasium: Create Custom Environments_. Recuperado el 3 de febrero de 2026, de https://gymnasium.farama.org/introduction/create_custom_env/

[9] Russell, S., & Norvig, P. (2020). _Artificial Intelligence: A Modern Approach_ (4th ed.). Pearson.

[10] Raoof, O., & Al-Raweshidy, H. (2011). _Theory of Games: An Introduction_. InTech. DOI: 10.5772/32940

[11] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). _Proximal Policy Optimization Algorithms_. arXiv preprint arXiv:1707.06347. Recuperado de https://arxiv.org/abs/1707.06347

[12] Asociación Argentina de Truco (ASART). (2017). _Reglamento de Juego_.
