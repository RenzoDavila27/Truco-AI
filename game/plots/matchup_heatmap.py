import argparse
import csv
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a matchup heatmap (win rate or average point diff)."
    )
    parser.add_argument(
        "--results-dir",
        default="resultados",
        help="Directory containing matchup CSV files",
    )
    parser.add_argument(
        "--metric",
        choices=["win_rate", "avg_diff"],
        default="win_rate",
        help="Metric to display in the heatmap",
    )
    parser.add_argument(
        "--agents",
        default="",
        help="Comma-separated list of agents to include (optional)",
    )
    return parser.parse_args()


def load_matchup_stats(csv_path: Path) -> tuple[str, str, float, float]:
    wins_j0 = 0
    games = 0
    diff_sum = 0.0
    agent_0 = ""
    agent_1 = ""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not agent_0:
                agent_0 = row["agent_0"]
                agent_1 = row["agent_1"]
            winner = row["winner"]
            if winner == "J0":
                wins_j0 += 1
            diff_sum += float(row["points_j0"]) - float(row["points_j1"])
            games += 1
    if games == 0:
        raise ValueError(f"No games found in {csv_path}")
    win_rate_j0 = wins_j0 / games
    avg_diff_j0 = diff_sum / games
    return agent_0, agent_1, win_rate_j0, avg_diff_j0


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    stats: dict[tuple[str, str], dict[str, float]] = {}
    agents: set[str] = set()

    for csv_path in results_dir.glob("*results.csv"):
        if "bluff" in csv_path.name or "sources" in csv_path.name:
            continue
        agent_0, agent_1, win_rate_j0, avg_diff_j0 = load_matchup_stats(csv_path)
        agents.update([agent_0, agent_1])
        stats[(agent_0, agent_1)] = {
            "win_rate": win_rate_j0,
            "avg_diff": avg_diff_j0,
        }
        stats[(agent_1, agent_0)] = {
            "win_rate": 1.0 - win_rate_j0,
            "avg_diff": -avg_diff_j0,
        }

    if args.agents:
        agent_list = [name.strip() for name in args.agents.split(",") if name.strip()]
    else:
        agent_list = sorted(agents)

    if not agent_list:
        raise ValueError("No agents found to plot.")

    size = len(agent_list)
    matrix = np.full((size, size), np.nan, dtype=float)

    for i, row_agent in enumerate(agent_list):
        for j, col_agent in enumerate(agent_list):
            if row_agent == col_agent:
                continue
            metric = stats.get((row_agent, col_agent))
            if metric:
                matrix[i, j] = metric[args.metric]

    plt.figure(figsize=(8, 7))
    if args.metric == "avg_diff":
        max_abs = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
        norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
        cmap = "RdBu_r"
    else:
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        cmap = "RdYlGn"

    im = plt.imshow(matrix, cmap=cmap, norm=norm)
    plt.colorbar(im, fraction=0.046, pad=0.04, label=args.metric)

    plt.xticks(range(size), agent_list, rotation=35, ha="right")
    plt.yticks(range(size), agent_list)

    for i in range(size):
        for j in range(size):
            if not np.isfinite(matrix[i, j]):
                continue
            value = matrix[i, j]
            text = f"{value:.2f}"
            plt.text(j, i, text, ha="center", va="center", color="#1f1f1f")

    plt.title(f"Matchup heatmap: {args.metric}")
    plt.grid(False)
    plt.tight_layout()

    output_dir = Path("game") / "plots" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"matchup_heatmap_{args.metric}.png"
    plt.savefig(output_path, dpi=150)


if __name__ == "__main__":
    main()
