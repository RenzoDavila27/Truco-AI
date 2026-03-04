"""
Grafica las losses de PPO (policy_gradient_loss, value_loss, entropy_loss, loss)
a partir del CSV generado durante el entrenamiento por LossLoggerCallback.
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt

GAME_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if GAME_DIR not in sys.path:
    sys.path.insert(0, GAME_DIR)


def _format_millions(value: float) -> str:
    millions = value / 1_000_000
    if abs(millions - round(millions)) < 1e-6:
        return f"{int(round(millions))}M"
    return f"{millions:.1f}M"


def _read_loss_csv(csv_path: str) -> dict:
    """Lee el CSV de losses y devuelve listas por columna."""
    data = {
        "timesteps": [],
        "policy_gradient_loss": [],
        "value_loss": [],
        "entropy_loss": [],
        "loss": [],
    }
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["timesteps"].append(int(row["timesteps"]))
            data["policy_gradient_loss"].append(float(row["policy_gradient_loss"]))
            data["value_loss"].append(float(row["value_loss"]))
            data["entropy_loss"].append(float(row["entropy_loss"]))
            data["loss"].append(float(row["loss"]))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grafica las losses de PPO a partir del CSV de entrenamiento."
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(
            os.path.abspath(os.path.join(GAME_DIR, "..")),
            "resultados",
            "ppo_training_losses.csv",
        ),
        help="Ruta al CSV de losses generado durante el entrenamiento.",
    )
    parser.add_argument(
        "--output",
        default="ppo_loss.png",
        help="Nombre del gráfico de salida.",
    )
    parser.add_argument(
        "--title-suffix",
        default="",
        help="Texto extra para el título (ej: nombre del oponente).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(
            f"No se encontró el CSV de losses: {args.csv}\n"
            "Ejecutá el entrenamiento con sb3_league_train.py para generarlo."
        )

    data = _read_loss_csv(args.csv)
    n = len(data["timesteps"])
    print(f"Leídas {n} filas de losses desde {args.csv}")

    if n == 0:
        raise ValueError("El CSV está vacío.")

    # ── Graficar ───────────────────────────────────────────────────
    plots_dir = os.path.join(GAME_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    x = data["timesteps"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    loss_configs = [
        ("Clipped Surrogate Loss", data["policy_gradient_loss"], "#E74C3C", axes[0, 0]),
        ("Value Loss",             data["value_loss"],            "#3498DB", axes[0, 1]),
        ("Entropy Loss",           data["entropy_loss"],          "#2ECC71", axes[1, 0]),
        ("Total Loss",             data["loss"],                  "#8E44AD", axes[1, 1]),
    ]

    for title, y_values, color, ax in loss_configs:
        ax.plot(x, y_values, linewidth=1.2, color=color, alpha=0.85)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.35)

    # X ticks formateados
    for ax in axes[1]:
        ax.set_xlabel("Timesteps")
        if x:
            min_x, max_x = min(x), max(x)
            if min_x == max_x:
                tick_values = [min_x]
            else:
                tick_count = min(10, len(x))
                step = (max_x - min_x) / (tick_count - 1)
                tick_values = [min_x + step * i for i in range(tick_count)]
            tick_labels = [_format_millions(v) for v in tick_values]
            ax.set_xticks(tick_values)
            ax.set_xticklabels(tick_labels, rotation=0, ha="center")

    suptitle = "PPO Training Losses"
    if args.title_suffix:
        suptitle += f" ({args.title_suffix})"
    fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = os.path.join(plots_dir, os.path.basename(args.output))
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Gráfico guardado en {output_path}")


if __name__ == "__main__":
    main()
