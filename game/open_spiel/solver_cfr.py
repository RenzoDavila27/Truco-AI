import os
import pickle
import pyspiel
from open_spiel.python.algorithms import external_sampling_mccfr

# Importamos tu juego para que se registre en OpenSpiel
import truco_argentino 

POLICY_FILENAME = "truco_mccfr_policy.pkl"
SOLVER_STATE_FILENAME = "truco_mccfr_solver.pkl"

def _policy_dir():
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, "policys")


def _policy_path():
    return os.path.join(_policy_dir(), POLICY_FILENAME)


def _solver_path():
    return os.path.join(_policy_dir(), SOLVER_STATE_FILENAME)


def save_policy(policy, filename=None):
    if filename is None:
        filename = _policy_path()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(policy, f)
    print(f"Política guardada en {filename}")


def load_policy(filename=None):
    if filename is None:
        filename = _policy_path()
    with open(filename, "rb") as f:
        return pickle.load(f)


def _save_solver_state(solver, filename=None):
    if filename is None:
        filename = _solver_path()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(solver, f)
    print(f"Estado del solver guardado en {filename}")


def _load_solver_state(filename=None):
    if filename is None:
        filename = _solver_path()
    if not os.path.exists(filename):
        return None
    with open(filename, "rb") as f:
        return pickle.load(f)


def _create_solver(game):
    return external_sampling_mccfr.ExternalSamplingSolver(
        game,
        average_type=external_sampling_mccfr.AverageType.SIMPLE
    )


def train_mccfr(iterations=10000, policy_path=None, solver_path=None):
    print("Cargando Truco Argentino...")
    game = pyspiel.load_game("truco_argentino")

    if solver_path is None:
        solver_path = _solver_path()
    if policy_path is None:
        policy_path = _policy_path()

    solver = None
    if os.path.exists(solver_path):
        try:
            solver = _load_solver_state(solver_path)
            print(f"Solver cargado desde {solver_path}.")
        except Exception as exc:
            print(f"No se pudo cargar el solver previo: {exc}. Se reinicia entrenamiento.")

    if solver is None:
        if os.path.exists(policy_path):
            print(f"Policy encontrada en {policy_path}, pero se reinician arrepentimientos.")
        solver = _create_solver(game)

    print(f"Iniciando entrenamiento por {iterations} iteraciones...")

    completed = 0
    try:
        for i in range(iterations):
            solver.iteration()
            completed += 1
            if iterations >= 10 and (i + 1) % (iterations // 10) == 0:
                print(f"Iteración {i + 1}/{iterations} completada.")
    except KeyboardInterrupt:
        print(f"Entrenamiento interrumpido en iteración {completed}. Guardando policy.")

    average_policy = solver.average_policy()
    save_policy(average_policy, policy_path)
    try:
        _save_solver_state(solver, solver_path)
    except Exception as exc:
        print(f"No se pudo guardar el solver: {exc}")

    return average_policy, completed


# --- Bloque de Prueba ---
if __name__ == "__main__":
    policy, completed = train_mccfr(iterations=10000)
    print(f"Iteraciones completas: {completed}")
