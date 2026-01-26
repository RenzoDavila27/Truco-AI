import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate bluff rate bars per agent (Truco and Envido)."
    )
    parser.add_argument("--agent-0", help="Name of agent 0 (for 2-agent mode)")
    parser.add_argument("--agent-1", help="Name of agent 1 (for 2-agent mode)")
    parser.add_argument(
        "--agents",
        default="",
        help="Comma-separated list of agents to include (multi-agent mode)",
    )
    parser.add_argument(
        "--results-dir",
        default="resultados",
        help="Directory containing bluff CSV files",
    )
    parser.add_argument(
        "--auto-games",
        type=int,
        default=1000,
        help="Number of games to simulate if bluff CSV is missing",
    )
    return parser.parse_args()


def ensure_bluff_csv(project_root: Path, results_dir: Path, agent_0: str, agent_1: str, auto_games: int) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / f"{agent_0}vs{agent_1}bluff_results.csv"
    swapped = results_dir / f"{agent_1}vs{agent_0}bluff_results.csv"
    if target.exists():
        return target
    if swapped.exists():
        return swapped

    generator = project_root / "game" / "agent_bluff_matchup.py"
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
    if target.exists():
        return target
    if swapped.exists():
        return swapped
    raise FileNotFoundError(f"No bluff results generated for {agent_0} vs {agent_1}")


def load_bluff_rates(csv_paths: list[Path], agent_names: list[str]) -> dict[str, dict[str, float]]:
    totals = {
        agent: {"truco_calls": 0, "truco_bluffs": 0, "envido_calls": 0, "envido_bluffs": 0}
        for agent in agent_names
    }

    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                caller = row.get("caller_agent", "")
                if caller not in totals:
                    continue
                event_type = row.get("event_type", "")
                if event_type == "truco":
                    totals[caller]["truco_calls"] += 1
                    if row.get("caller_weaker_hand") in ["1", "1.0", "True", "true"]:
                        totals[caller]["truco_bluffs"] += 1
                elif event_type == "envido":
                    totals[caller]["envido_calls"] += 1
                    if row.get("caller_weaker_envido") in ["1", "1.0", "True", "true"]:
                        totals[caller]["envido_bluffs"] += 1

    rates = {}
    for agent, stats in totals.items():
        truco_rate = (
            stats["truco_bluffs"] / stats["truco_calls"] if stats["truco_calls"] else 0.0
        )
        envido_rate = (
            stats["envido_bluffs"] / stats["envido_calls"] if stats["envido_calls"] else 0.0
        )
        rates[agent] = {"truco": truco_rate, "envido": envido_rate}
    return rates


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / args.results_dir

    if args.agents:
        agents = [name.strip() for name in args.agents.split(",") if name.strip()]
        if len(agents) < 2:
            raise ValueError("Provide at least two agents in --agents.")
        csv_paths = list(results_dir.glob("*bluff_results.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"No bluff_results.csv files found in {results_dir}")
    else:
        if not args.agent_0 or not args.agent_1:
            raise ValueError("Use --agent-0 and --agent-1 or provide --agents.")
        csv_paths = [
            ensure_bluff_csv(
                project_root, results_dir, args.agent_0, args.agent_1, args.auto_games
            )
        ]
        agents = [args.agent_0, args.agent_1]

    rates = load_bluff_rates(csv_paths, agents)

    labels = agents
    x = np.arange(len(labels))
    width = 0.35

    truco_rates = [rates[agent]["truco"] for agent in labels]
    envido_rates = [rates[agent]["envido"] for agent in labels]

    plt.figure(figsize=(8, 6))
    plt.bar(x - width / 2, truco_rates, width, label="Truco", color="#2C7FB8")
    plt.bar(x + width / 2, envido_rates, width, label="Envido", color="#F39C34")

    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Tasa de mentira")
    plt.title("Tasa de mentira por agente")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()

    output_dir = Path("game") / "plots" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.agents:
        output_name = "bluff_rate_multi.png"
    else:
        output_name = f"{args.agent_0}vs{args.agent_1}_bluff_rate.png"
    output_path = output_dir / output_name
    plt.savefig(output_path, dpi=150)


if __name__ == "__main__":
    main()
