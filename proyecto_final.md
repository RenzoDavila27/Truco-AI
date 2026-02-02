# Proyecto Final — Inteligencia Artificial 1
**Título:** Truco-AI: Agentes inteligentes para Truco Argentino (1v1, sin flor)  
**Autores:**  
**Fecha:**  
**Cátedra:** IA1 — UNCUYO  

---

## 1. Introducción

El Truco Argentino es un juego de cartas tradicional de dos jugadores, por turnos y de información imperfecta. Cada jugador recibe un conjunto reducido de cartas y, a lo largo de la mano, debe decidir cómo y cuándo jugar, cuándo aceptar o rechazar desafíos y cómo interpretar las señales del rival. La partida combina mecánicas de enfrentamiento directo con un componente fuerte de toma de decisiones bajo incertidumbre, ya que las cartas del oponente son desconocidas y la información disponible se construye a partir de las jugadas, los cantos y el contexto de la mano. En términos generales, se trata de un juego competitivo donde la estrategia depende tanto de la evaluación de la mano propia como de la lectura del oponente.

En este tipo de juegos, los **agentes tradicionales** suelen basarse en reglas fijas o heurísticas diseñadas a mano, y a menudo incorporan **parámetros de estilo** (agresividad, frecuencia de mentira, tolerancia al riesgo, etc.). En lugar de aprender, estos agentes aplican reglas predefinidas para elegir acciones “razonables” con la información parcial disponible. Este enfoque produce comportamientos coherentes y explicables, pero tiene limitaciones: al no aprender de la experiencia, no se adapta de forma dinámica a oponentes o contextos nuevos. Por eso, estos agentes son una buena línea base para comparar con enfoques de **Reinforcement Learning**, que aprenden políticas de acción a partir de la interacción con el entorno.

El Truco es un candidato adecuado para aplicar RL porque presenta los elementos clásicos de un entorno de aprendizaje: decisiones secuenciales, recompensas claras (puntos ganados o perdidos) y un espacio de estados rico en incertidumbre. Además, su naturaleza de información imperfecta permite estudiar comportamientos de exploración, adaptación y modelado implícito del oponente. Sin embargo, también plantea desafíos: el estado observado es parcial, las recompensas pueden ser retrasadas y la dinámica de apuestas introduce componentes estratégicos difíciles de capturar con reglas simples. En este sentido, el Truco resulta interesante para RL, pero exige un diseño cuidadoso de la representación del estado, la función de recompensa y la evaluación experimental.

Este informe se organiza de la siguiente manera: primero se presenta el marco teórico necesario para entender el problema y las técnicas empleadas; luego se describe el diseño experimental, las métricas y herramientas utilizadas; más adelante se reportan los resultados obtenidos y su análisis; y finalmente se exponen las conclusiones y el trabajo futuro, junto con la bibliografía consultada.

---

## 2. Marco teórico

### 2.1 Aprendizaje por refuerzo (RL)

El **aprendizaje por refuerzo** es un paradigma de aprendizaje automático donde un **agente** interactúa con un **entorno** tomando **acciones** y recibiendo **recompensas**. El objetivo del agente es aprender una **política** que maximice el beneficio esperado en el tiempo. En el Truco, el entorno es el juego, las acciones corresponden a las decisiones posibles (jugar una carta, cantar, aceptar o rechazar), y la recompensa se asocia al resultado en puntos.

Un concepto central es la formulación como **Proceso de Decisión de Markov Parcialmente Observable (POMDP)**. Formalmente, un POMDP se describe por:

- **S:** conjunto de estados reales.
- **A:** conjunto de acciones.
- **T(s'|s,a):** función de transición.
- **R(s,a):** recompensa.
- **O:** conjunto de observaciones.
- **Z(o|s',a):** modelo de observación.
- **γ:** factor de descuento.

En Truco, el estado completo incluye las cartas de ambos jugadores, pero el agente solo accede a una observación parcial (sus cartas e historial), lo que obliga a decidir bajo incertidumbre, integrando señales del rival y el contexto de la mano.

Además, al tratarse de un juego competitivo de dos jugadores, el análisis se vincula con nociones de **teoría de juegos**, como el **equilibrio de Nash**. En términos generales, un perfil de estrategias **(π₁, π₂)** es un equilibrio de Nash si ninguna parte puede mejorar su resultado cambiando unilateralmente su estrategia, es decir:

- **u₁(π₁, π₂) ≥ u₁(π₁', π₂)**
- **u₂(π₁, π₂) ≥ u₂(π₁, π₂')**

para cualquier estrategia alternativa **π₁'** o **π₂'**. En juegos de suma cero como el Truco, esta noción se relaciona con estrategias estables donde cada jugador se protege de la mejor respuesta del rival.

En esta clase de problemas también es habitual restringir las **acciones válidas** en cada turno para mantener la coherencia con las reglas del juego. Estas restricciones permiten que el agente concentre su aprendizaje en decisiones legítimas, evitando estados imposibles. En conjunto, estas características hacen del Truco un dominio natural para RL: decisiones secuenciales, información parcial y un fuerte componente estratégico que depende tanto de la mano propia como de la interpretación del oponente.

### 2.2 Políticas y evaluación de decisiones

Una **política** define cómo decide el agente en función de lo que observa. Para comparar alternativas, se utilizan **criterios de evaluación** que estiman qué tan conveniente es una acción según las recompensas esperadas. Estas estimaciones permiten ordenar las decisiones disponibles y elegir la opción que, en promedio, conduce a mejores resultados. En un juego como el Truco, esta evaluación debe considerar que el efecto de una decisión puede manifestarse varios turnos después y que el rival influye activamente en el resultado.

---

### 2.3 Algoritmos utilizados

#### 2.3.1 Agente Random

**Descripción:** selecciona una acción válida al azar en cada turno, sin usar información histórica ni estrategia.

**Justificación:** sirve como línea base mínima para medir si los métodos más sofisticados realmente aprenden o mejoran el rendimiento.
Además, establece un punto de comparación independiente de cualquier sesgo de diseño: si un agente entrenado no supera al azar, la señal de aprendizaje es débil o el modelado del entorno no está capturando bien la tarea.

#### 2.3.2 Agente Rational (basado en reglas)

**Descripción:** aplica reglas fijas para decidir (por ejemplo, criterios deterministas para cantar, aceptar o jugar cartas según la fuerza percibida).

**Justificación:** ofrece una política coherente y explicable, útil como baseline más fuerte que el azar y como referencia contra la cual comparar agentes entrenados.
En Truco, donde las decisiones dependen de criterios humanos como “mano fuerte” o “riesgo asumible”, este agente ayuda a evaluar si el aprendizaje automático realmente capta patrones estratégicos que superen heurísticas explícitas.

#### 2.3.3 Q-Learning

**Descripción:** aprende una función de acción-valor **Q(s,a)** que estima la recompensa esperada al ejecutar una acción **a** en un estado **s** y continuar con la política aprendida. La actualización clásica es:

**Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]**

**Justificación:** es un método model-free que permite aprender decisiones óptimas a partir de la interacción, incluso en entornos con dinámica compleja como el Truco, donde no se dispone de un modelo exacto del rival ni de transiciones explícitas.
También es apropiado cuando se quiere estudiar el impacto del aprendizaje incremental en un dominio con información parcial, ya que permite observar cómo se ajustan los valores de acción a partir de la experiencia.

#### 2.3.4 PPO (Proximal Policy Optimization)

**Descripción:** **PPO (Proximal Policy Optimization)** es un método de optimización de políticas que busca mejorar la estabilidad del entrenamiento evitando actualizaciones demasiado grandes. En lugar de cambiar la política libremente, impone un límite al cambio permitido en cada paso.

En términos simples, PPO compara la política nueva con la anterior y solo permite cambios si el beneficio esperado mejora sin desviarse demasiado. Si el cambio propuesto es excesivo, se recorta para mantener la actualización dentro de un rango seguro. Esto evita saltos bruscos que podrían degradar el desempeño y ayuda a que el aprendizaje sea más estable.

**Justificación:** PPO es robusto y estable en entornos complejos con espacios de acciones restringidos. En Truco, donde las decisiones cambian según el contexto de la mano y las apuestas, la estabilidad del entrenamiento ayuda a evitar políticas erráticas. Además, permite entrenar políticas parametrizadas capaces de generalizar a situaciones no vistas, lo que resulta valioso en un dominio con alto componente de incertidumbre.

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
  **Interpretación:** valores altos indican un estilo más bluff/agresivo; valores bajos reflejan un juego conservador. Se analiza junto al win rate para evaluar si el farol es efectivo o simplemente riesgoso.

### 3.2 Herramientas
- **Lenguaje:** Python 3.11+ (compatibilidad declarada en el proyecto).  
  **Motivo:** ecosistema maduro para IA/RL y rápida iteración experimental.

- **Entorno propio:** implementación completa del juego y wrapper tipo Gymnasium en `game/truco_env.py`.  
  **Motivo:** se requiere modelar reglas y dinámicas específicas del Truco (1v1, sin flor), que no existen en entornos estándar.

- **Librerías (versions mínimas declaradas):**
  - **gymnasium** (>=1.0,<2): interfaz de entornos y compatibilidad con agentes RL.
  - **numpy** (>=1.24): operaciones numéricas y manejo de vectores/estadísticas.
  - **torch** (>=2.0): soporte para modelos y optimización con redes neuronales.
  - **stable-baselines3** (>=2.0) y **sb3-contrib** (>=2.0): implementación de algoritmos modernos y utilidades adicionales.
  - **matplotlib** (>=3.7): generación de gráficos para análisis de resultados.
  **Motivo:** conjunto estándar en proyectos RL, con buen soporte y documentación.

- **Control de versiones:** Git (historial de commits, issues y ramas).  
  **Motivo:** trazabilidad del desarrollo, colaboración y evaluación del proceso.

- **Graficadores:** Matplotlib (gráficos de métricas y resultados).  
  **Motivo:** visualización clara y reproducible de la evolución del rendimiento.

### 3.3 Datos / simulación
- No hay dataset externo: se genera por self-play o partidas simuladas.
- Configuración de episodios / manos / cantidad de partidas.
- Parámetros principales (α, γ, ε, hands, etc.).

### 3.4 Experimentos y resultados esperados
- **E1:** Q-Learning en self-play.  
- **E2:** Q-Learning vs agente racional.  
- **E3:** Policy Gradient lineal vs random/rational.  
- **E4:** Policy Gradient NN vs random/rational.  
- **E5:** SB3 PPO vs agentes base.  

Incluir tablas/gráficos con:
- Comparativa de win rate.
- Puntos promedio.
- Tiempo de entrenamiento / episodios.

---

## 4. Análisis y discusión de resultados
- Interpretación de por qué algunos agentes superan a otros.
- Impacto del modelado parcial de información.
- Limitaciones del entorno / simulación.
- Calidad del entrenamiento (convergencia, sobreajuste, exploración).

---

## 5. Conclusiones finales
- Logros alcanzados.
- Qué quedó pendiente.
- Trabajo futuro (opponent modeling, nuevas recompensas, arquitecturas más profundas, mejores métricas).
