# Linea de tiempo del

Este documento resume la evolucion del proyecto con base en el historial de commits y las decisiones del chat. Si algun punto no aparece en el chat, se indica que proviene del repositorio.

## Linea de tiempo (resumen por etapas)

1) Arranque del proyecto
- Se crea el entorno virtual y la estructura base del juego.
- Se definen cartas, ranking y constantes.
- Se inicia el modelado de estado y acciones.

2) Logica 1v1 y simulacion en consola
- Se separa logica de juego del control de turnos.
- Se habilita 1v1 real en consola y modo human/debug en render.

3) Partidas completas
- El entorno deja de jugar solo rondas y pasa a partidas completas hasta 30.
- Se ajusta suma de puntos con tope y reinicio de mano.

4) Agentes y registro
- Se crea agente random y racional.
- Se agrega un registro de agentes para elegir en consola y en scripts.

5) Scripts de agente vs agente
- Se agrega simulador de agente vs agente.

6) Scripts para estadisticas
- Se agrega simulador de partidas, donde se guardan diferentes datos de la misma para su posterior analisis

7) Q-Learning
- Se crea agente Q-Learning
- Se crea script de entrenamiento y persistencia de Q-Table.

8) Analisis de Q-Table
- Se crea script de analisis para inspeccionar distribuciones de Q-values.
- Se propone cambiar el metodo de entrenamiento a Monte Carlo.



## Problemas encontrados y soluciones principales

- Empates de ronda y turnos incorrectos:
  Se ajusto la logica para que el turno solo cambie al tirar cartas y que los empates se resuelvan en rondas siguientes o por mano.

- Acciones permitidas en mascara:
  Se corrigio la mascara para evitar truco durante respuesta de envido y para permitir envido en ronda 1 si el truco no fue aceptado.

- Cancelacion de truco con envido:
  Se aseguro que el envido puede cancelar el truco pendiente en ronda 1 y que el turno vuelva correctamente.

- Turnos al cantar:
  Se separo el turno de respuesta del turno de tirar carta para evitar cambios indebidos.

- Q-Table con valores 0 en cantos:
  Se detecto que los cantos no recibian credito por recompensa tardia.
  Se cambio el entrenamiento a Monte Carlo para propagar retornos.
