"""
Compare bluff rates across Q-Learning experiments.

Runs each experiment's Q-table against a fixed opponent (default: rational)
and plots the Truco/Envido bluff rates side by side.

Usage:
    python game/plots/bluff_rate_q_experiments.py --games 1000
"""

import argparse
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
from plots.bluff_rate_bars import load_bluff_rates  # noqa: E402

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
        description="Compare bluff rates across Q-Learning experiments."
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


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "resultados"

    csv_paths = []
    agent_labels = []

    for exp in EXPERIMENTS:
        label_clean = exp["label"].replace("\n", "_").replace(" ", "").replace("(", "").replace(")", "")
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

    # Load bluff rates - we use "q_learning" as the agent name in the CSV
    # but we need to collect per-experiment
    truco_rates = []
    envido_rates = []

    for csv_path in csv_paths:
        rates = load_bluff_rates([csv_path], ["q_learning"])
        truco_rates.append(rates.get("q_learning", {}).get("truco", 0.0))
        envido_rates.append(rates.get("q_learning", {}).get("envido", 0.0))

    # Plot
    x = np.arange(len(agent_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars_truco = ax.bar(x - width / 2, truco_rates, width, label="Truco", color="#2C7FB8")
    bars_envido = ax.bar(x + width / 2, envido_rates, width, label="Envido", color="#F39C34")

    # Add value labels on bars
    for bar in bars_truco:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.1%}",
                ha="center", va="bottom", fontsize=10)
    for bar in bars_envido:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.1%}",
                ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(agent_labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Tasa de mentira")
    ax.set_title(f"Tasa de mentira por experimento Q-Learning (vs {args.opponent})")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_dir = project_root / "game" / "plots" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"bluff_rate_q_experiments_vs_{args.opponent}.png"
    fig.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
