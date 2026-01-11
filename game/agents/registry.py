import importlib.util
import os

from agents.random_agent import RandomAgent
from agents.rational_agent import RationalAgent


def _load_q_learning_agent():
    base_dir = os.path.dirname(__file__)
    agent_path = os.path.join(base_dir, "RL-Agents", "agent_q_learning.py")
    spec = importlib.util.spec_from_file_location("agent_q_learning", agent_path)
    if spec is None or spec.loader is None:
        raise ImportError("No se pudo cargar agent_q_learning.py.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.QLearningAgent


def get_agent_registry():
    return {
        "random": RandomAgent,
        "rational": RationalAgent,
        "q_learning": _load_q_learning_agent(),
    }


def create_agent(name):
    registry = get_agent_registry()
    if name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Agente desconocido: {name}. Disponibles: {available}")
    return registry[name]()
