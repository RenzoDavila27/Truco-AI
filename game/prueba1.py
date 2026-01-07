from truco_env import TrucoEnv
from constantes import Acciones

env = TrucoEnv()
obs, info = env.reset()

done = False
while not done:
    # Ejemplo: jugar siempre la primera carta disponible
    action = Acciones.JUGAR_CARTA_1.value
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    print("reward:", reward)
