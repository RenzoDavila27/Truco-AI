import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a boxplot of points for a given agent matchup."
    )
    parser.add_argument("--agent-0", required=True, help="Name of agent 0")
    parser.add_argument("--agent-1", required=True, help="Name of agent 1")
    parser.add_argument(
        "--results-dir",
        default="resultados",
        help="Directory containing matchup CSV files",
    )
    parser.add_argument(
        "--auto-games",
        type=int,
        default=1000,
        help="Number of games to simulate if matchup CSV is missing",
    )
    return parser.parse_args()


def ensure_matchup_csv(project_root: Path, results_dir: Path, agent_0: str, agent_1: str, auto_games: int) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / f"{agent_0}vs{agent_1}results.csv"
    swapped = results_dir / f"{agent_1}vs{agent_0}results.csv"
    if target.exists() or swapped.exists():
        return

    generator = project_root / "game" / "agent_matchup.py"
    subprocess.run(
        [
            sys.executable,
            str(generator),
            "--agent-0",
            agent_0,
            "--agent-1",
            agent_1,
            "--games",
            str(auto_games),
        ],
        check=True,
        cwd=project_root,
    )


def load_points(csv_path: Path, swap: bool) -> tuple[list[int], list[int]]:
    points_j0: list[int] = []
    points_j1: list[int] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            points_j0.append(int(row["points_j0"]))
            points_j1.append(int(row["points_j1"]))
    if swap:
        points_j0, points_j1 = points_j1, points_j0
    return points_j0, points_j1


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / args.results_dir

    ensure_matchup_csv(project_root, results_dir, args.agent_0, args.agent_1, args.auto_games)

    matchup_name = f"{args.agent_0}vs{args.agent_1}"
    csv_path = results_dir / f"{matchup_name}results.csv"
    swap = False

    if not csv_path.exists():
        swapped_matchup = f"{args.agent_1}vs{args.agent_0}"
        swapped_path = results_dir / f"{swapped_matchup}results.csv"
        if swapped_path.exists():
            csv_path = swapped_path
            swap = True
        else:
            raise FileNotFoundError(
                f"No matchup CSV found for {matchup_name} or {swapped_matchup} in {results_dir}"
            )

    points_j0, points_j1 = load_points(csv_path, swap=swap)

    output_dir = Path("game") / "plots" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{matchup_name}_boxplot.png"

    plt.figure(figsize=(8, 6))
    box_data = [points_j0, points_j1]
    box = plt.boxplot(
        box_data,
        tick_labels=[args.agent_0, args.agent_1],
        patch_artist=True,
        widths=0.6,
        medianprops={"color": "#333333", "linewidth": 1.5},
        boxprops={"edgecolor": "#333333", "linewidth": 1.5},
        whiskerprops={"color": "#333333", "linewidth": 1.2},
        capprops={"color": "#333333", "linewidth": 1.2},
    )

    palette = ["#2C7FB8", "#F39C34"]
    for patch, color in zip(box["boxes"], palette, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.95)

    rng = np.random.default_rng(42)
    for idx, values in enumerate(box_data, start=1):
        jitter = rng.normal(0, 0.05, size=len(values))
        x_positions = np.full(len(values), idx) + jitter
        plt.scatter(
            x_positions,
            values,
            s=5,
            color="#3A3A3A",
            alpha=0.85,
            zorder=3,
        )

    plt.title(f"Points per game: {args.agent_0} vs {args.agent_1}")
    plt.ylabel("Points")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


if __name__ == "__main__":
    main()
