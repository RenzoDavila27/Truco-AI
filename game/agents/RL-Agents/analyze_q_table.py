import os
import pickle
import sys
from collections import defaultdict

GAME_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if GAME_DIR not in sys.path:
    sys.path.insert(0, GAME_DIR)

from constantes import Acciones

QTABLE_PATH = os.path.join(os.path.dirname(__file__), "q_tables", "q_table.pkl")


def _load_q_table(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _action_groups():
    return {
        "cartas": {
            Acciones.JUGAR_CARTA_1.value,
            Acciones.JUGAR_CARTA_2.value,
            Acciones.JUGAR_CARTA_3.value,
        },
        "envido": {
            Acciones.ENVIDO.value,
            Acciones.ENVIDO_ENVIDO.value,
            Acciones.REAL_ENVIDO.value,
            Acciones.FALTA_ENVIDO.value,
        },
        "truco": {
            Acciones.TRUCO.value,
            Acciones.RETRUCO.value,
            Acciones.VALE_CUATRO.value,
        },
        "respuestas": {
            Acciones.QUIERO.value,
            Acciones.NO_QUIERO.value,
        },
        "mazo": {Acciones.IR_AL_MAZO.value},
    }


def _summarize_q_values(q_table):
    total = len(q_table)
    zero_count = sum(1 for v in q_table.values() if v == 0)
    nonzero = [v for v in q_table.values() if v != 0]
    if nonzero:
        min_v = min(nonzero)
        max_v = max(nonzero)
        avg_v = sum(nonzero) / len(nonzero)
    else:
        min_v = max_v = avg_v = 0
    return {
        "total": total,
        "zero_count": zero_count,
        "nonzero_count": len(nonzero),
        "min_nonzero": min_v,
        "max_nonzero": max_v,
        "avg_nonzero": avg_v,
    }


def _summarize_by_action(q_table):
    stats = defaultdict(list)
    for (state, action), value in q_table.items():
        stats[action].append(value)

    out = {}
    for action, values in stats.items():
        zero_count = sum(1 for v in values if v == 0)
        nonzero = [v for v in values if v != 0]
        out[action] = {
            "count": len(values),
            "zero_count": zero_count,
            "nonzero_count": len(nonzero),
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "avg": sum(values) / len(values) if values else 0,
        }
    return out


def _summarize_by_group(q_table):
    groups = _action_groups()
    stats = {name: [] for name in groups}
    for (state, action), value in q_table.items():
        for name, action_set in groups.items():
            if action in action_set:
                stats[name].append(value)
                break
    out = {}
    for name, values in stats.items():
        zero_count = sum(1 for v in values if v == 0)
        nonzero = [v for v in values if v != 0]
        out[name] = {
            "count": len(values),
            "zero_count": zero_count,
            "nonzero_count": len(nonzero),
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "avg": sum(values) / len(values) if values else 0,
        }
    return out


def _top_states_by_action(q_table, action_id, top_n=5):
    entries = [((state, action), value) for (state, action), value in q_table.items() if action == action_id]
    entries.sort(key=lambda item: item[1], reverse=True)
    return entries[:top_n]


def main():
    q_table = _load_q_table(QTABLE_PATH)
    summary = _summarize_q_values(q_table)
    print("Resumen general")
    for k, v in summary.items():
        print(f"- {k}: {v}")

    print("\nResumen por accion")
    by_action = _summarize_by_action(q_table)
    for action_id, info in sorted(by_action.items()):
        action_name = Acciones(action_id).name if action_id in Acciones._value2member_map_ else str(action_id)
        print(f"- {action_name}: {info}")

    print("\nResumen por grupo")
    by_group = _summarize_by_group(q_table)
    for name, info in by_group.items():
        print(f"- {name}: {info}")

    print("\nTop Q por acciones de respuesta (QUIERO/NO_QUIERO)")
    for action_id in [Acciones.QUIERO.value, Acciones.NO_QUIERO.value]:
        action_name = Acciones(action_id).name
        top_entries = _top_states_by_action(q_table, action_id)
        print(f"{action_name}:")
        for (state_action, value) in top_entries:
            print(f"  - Q={value} | state={state_action[0]}")


if __name__ == "__main__":
    main()
