from agents.random_agent import RandomAgent
from agents.rational_agent import RationalAgent


def get_agent_registry():
    return {
        "random": RandomAgent,
        "rational": RationalAgent,
    }


def create_agent(name):
    registry = get_agent_registry()
    if name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Agente desconocido: {name}. Disponibles: {available}")
    return registry[name]()
