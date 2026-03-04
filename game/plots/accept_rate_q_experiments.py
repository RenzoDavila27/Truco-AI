"""
Compare acceptance rates across Q-Learning experiments.

Runs each experiment's Q-table against a fixed opponent (default: rational)
and plots the Truco/Envido acceptance rates side by side, showing whether
each policy is more aggressive (accepts more) or conservative (declines more).

Usage:
    python game/plots/accept_rate_q_experiments.py --games 1000
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow imports from game/
GAME_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if GAME_DIR not in sys.path:
    sys.path.insert(0, GAME_DIR)

from agent_bluff_matchup import main as bluff_main  # noqa: E402

EXPERIMENTS = [
    {
        "label": "Exp 1\n(Self-play)",
        "q_table": "exp1_selfplay_2026.pkl",
    },
    {
        "label": "Exp 2\n(vs Racional)",
        "q_table": "exp2_vs_rational_789987.pkl",
    },
    {
        "label": "Exp 3\n(Mixto)",
        "q_table": "exp3_mix_rational_2026.pkl",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare acceptance rates across Q-Learning experiments."
    )
    parser.add_argument(
        "--opponent", default="rational", help="Fixed opponent to play against."
    )
    parser.add_argument(
        "--games", type=int, default=1000, help="Games per experiment."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-run simulations even if CSVs exist."
    )
    return parser.parse_args()


def load_accept_counts(csv_path: Path, agent_name: str = "q_learning") -> dict:
    """Load acceptance counts from a bluff results CSV.

    Returns dict with raw counts per response type for truco and envido,
    only counting responses from the specified agent.
    """
    truco = {"accept": 0, "decline": 0, "raise": 0}
    envido = {"accept": 0, "decline": 0, "raise": 0}

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            responder = row.get("responder_agent", "")
            if responder != agent_name:
                continue
            response_type = row.get("response_type", "")
            if response_type not in ("accept", "decline", "raise"):
                continue

            event_type = row.get("event_type", "")
            if event_type == "truco":
                truco[response_type] += 1
            elif event_type == "envido":
                envido[response_type] += 1

    return {"truco": truco, "envido": envido}


def _add_labels(ax, bars, values, bottoms, totals):
    """Add percentage labels inside stacked bar segments."""
    for bar, val, bot, total in zip(bars, values, bottoms, totals):
        if total == 0 or val == 0:
            continue
        pct = val / total
        if pct > 0.05:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bot + val / 2,
                f"{pct:.0%}",
                ha="center", va="center", fontsize=10, fontweight="bold", color="white",
            )


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "resultados"

    csv_paths = []
    agent_labels = []

    for exp in EXPERIMENTS:
        label_clean = (
            exp["label"]
            .replace("\n", "_")
            .replace(" ", "")
            .replace("(", "")
            .replace(")", "")
        )
        csv_name = f"ql_{label_clean}_vs_{args.opponent}_bluff_results.csv"
        csv_path = results_dir / csv_name

        if not csv_path.exists() or args.force:
            print(f"Running: {exp['label']} vs {args.opponent} ({args.games} games)...")
            bluff_main(
                agent_0="q_learning",
                agent_1=args.opponent,
                games=args.games,
                output_name=csv_name,
                summary_name=csv_name.replace("_results.csv", "_summary.txt"),
                q_table_j0=exp["q_table"],
                agent_0_label=label_clean,
            )
        else:
            print(f"Using existing: {csv_path.name}")

        csv_paths.append(csv_path)
        agent_labels.append(exp["label"])

    # Load counts per experiment
    all_counts = [load_accept_counts(p) for p in csv_paths]

    x = np.arange(len(agent_labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Truco ---
    t_accept = [c["truco"]["accept"] for c in all_counts]
    t_decline = [c["truco"]["decline"] for c in all_counts]
    t_raise = [c["truco"]["raise"] for c in all_counts]
    t_total = [a + d + r for a, d, r in zip(t_accept, t_decline, t_raise)]

    b1 = ax1.bar(x, t_accept, width, label="Acepta", color="#27AE60")
    b2 = ax1.bar(x, t_decline, width, bottom=t_accept, label="Rechaza", color="#E74C3C")
    b3 = ax1.bar(x, t_raise, width,
                 bottom=[a + d for a, d in zip(t_accept, t_decline)],
                 label="Sube apuesta", color="#F39C12")

    _add_labels(ax1, b1, t_accept, [0] * len(x), t_total)
    _add_labels(ax1, b2, t_decline, t_accept, t_total)
    _add_labels(ax1, b3, t_raise, [a + d for a, d in zip(t_accept, t_decline)], t_total)

    ax1.set_xticks(x)
    ax1.set_xticklabels(agent_labels)
    ax1.set_ylabel("Cantidad de cantos recibidos")
    ax1.set_title("Respuestas a Truco")
    ax1.legend(loc="upper right", frameon=False)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    # --- Envido ---
    e_accept = [c["envido"]["accept"] for c in all_counts]
    e_decline = [c["envido"]["decline"] for c in all_counts]
    e_raise = [c["envido"]["raise"] for c in all_counts]
    e_total = [a + d + r for a, d, r in zip(e_accept, e_decline, e_raise)]

    b1 = ax2.bar(x, e_accept, width, label="Acepta", color="#27AE60")
    b2 = ax2.bar(x, e_decline, width, bottom=e_accept, label="Rechaza", color="#E74C3C")
    b3 = ax2.bar(x, e_raise, width,
                 bottom=[a + d for a, d in zip(e_accept, e_decline)],
                 label="Sube apuesta", color="#F39C12")

    _add_labels(ax2, b1, e_accept, [0] * len(x), e_total)
    _add_labels(ax2, b2, e_decline, e_accept, e_total)
    _add_labels(ax2, b3, e_raise, [a + d for a, d in zip(e_accept, e_decline)], e_total)

    ax2.set_xticks(x)
    ax2.set_xticklabels(agent_labels)
    ax2.set_ylabel("Cantidad de cantos recibidos")
    ax2.set_title("Respuestas a Envido")
    ax2.legend(loc="upper right", frameon=False)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(
        f"Distribución de respuestas por experimento Q-Learning (vs {args.opponent})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    output_dir = project_root / "game" / "plots" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"accept_rate_q_experiments_vs_{args.opponent}.png"
    fig.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
