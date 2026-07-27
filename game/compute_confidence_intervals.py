"""
Calcula intervalos de confianza para los resultados del proyecto Truco-AI.

- Win rates: Intervalo de Wilson al 95%
- Métricas continuas (puntos, manos): media ± 1.96 * SE

Genera:
1. Tablas Markdown con ICs para copiar al documento
2. Gráficos de curvas de evaluación con bandas de confianza
"""

import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Rutas ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "resultados")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


# ── Funciones estadísticas ─────────────────────────────────────────────────

def wilson_ci(successes, n, z=1.96):
    """Intervalo de confianza Wilson al 95% para una proporción binomial."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return p_hat, lo, hi


def mean_ci(values, z=1.96):
    """Media ± z * SE para una lista de valores."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    se = math.sqrt(var / n)
    return mean, mean - z * se, mean + z * se


def fmt_pct_ci(p, lo, hi):
    """Formatea un porcentaje con IC Wilson."""
    return f"{p*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]"


def fmt_mean_ci(mean, lo, hi):
    """Formatea una media con IC."""
    margin = (hi - lo) / 2
    return f"{mean:.2f} ± {margin:.2f}"


# ── Lectura de datos ───────────────────────────────────────────────────────

def read_results_csv(filename):
    """Lee un CSV de resultados de matchup (partida por partida)."""
    path = os.path.join(RESULTS_DIR, filename)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def read_eval_csv(filename):
    """Lee un CSV de evaluación durante entrenamiento."""
    path = os.path.join(RESULTS_DIR, filename)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ── Cálculo de ICs para tablas de resultados globales ──────────────────────

def compute_matchup_ci(filename, agent_j0_name, perspective="j0"):
    """
    Calcula ICs para un enfrentamiento.
    perspective: 'j0' o 'j1' — desde qué jugador se reportan los resultados.
    """
    rows = read_results_csv(filename)
    n = len(rows)

    if perspective == "j0":
        wins = sum(1 for r in rows if r["winner"] == "J0")
        points = [int(r["points_j0"]) for r in rows]
        hands_won = [int(r["hands_won_j0"]) for r in rows]
    else:
        wins = sum(1 for r in rows if r["winner"] == "J1")
        points = [int(r["points_j1"]) for r in rows]
        hands_won = [int(r["hands_won_j1"]) for r in rows]

    hands_played = [int(r["hands_played"]) for r in rows]

    wr_p, wr_lo, wr_hi = wilson_ci(wins, n)
    pts_mean, pts_lo, pts_hi = mean_ci(points)
    hp_mean, hp_lo, hp_hi = mean_ci(hands_played)
    hw_mean, hw_lo, hw_hi = mean_ci(hands_won)

    return {
        "n": n,
        "winrate": fmt_pct_ci(wr_p, wr_lo, wr_hi),
        "points": fmt_mean_ci(pts_mean, pts_lo, pts_hi),
        "hands_played": fmt_mean_ci(hp_mean, hp_lo, hp_hi),
        "hands_won": fmt_mean_ci(hw_mean, hw_lo, hw_hi),
    }


def compute_head_to_head_ci(filename, perspective="j0"):
    """Calcula IC Wilson para el win rate de un enfrentamiento directo."""
    rows = read_results_csv(filename)
    n = len(rows)
    if perspective == "j0":
        wins = sum(1 for r in rows if r["winner"] == "J0")
    else:
        wins = sum(1 for r in rows if r["winner"] == "J1")
    p, lo, hi = wilson_ci(wins, n)
    return p, lo, hi, n


# ── Sección 3.7.1: Resultados globales ─────────────────────────────────────

def print_global_results():
    print("=" * 80)
    print("SECCIÓN 3.7.1 — RESULTADOS GLOBALES CON IC95%")
    print("=" * 80)

    # Los enfrentamientos están organizados como agent_j0 vs agent_j1
    # Necesitamos reportar desde la perspectiva del agente evaluado

    agents = {
        "Random": [
            ("randomvsrationalresults.csv", "Rational", "j0"),
            ("randomvsq_learningresults.csv", "Q-Learning", "j0"),
            ("randomvssb3_leagueresults.csv", "PPO", "j0"),
        ],
        "Rational": [
            ("randomvsrationalresults.csv", "Random", "j1"),
            ("rationalvsq_learningresults.csv", "Q-Learning", "j0"),
            ("rationalvssb3_leagueresults.csv", "PPO", "j0"),
        ],
        "Q-Learning": [
            ("randomvsq_learningresults.csv", "Random", "j1"),
            ("rationalvsq_learningresults.csv", "Rational", "j1"),
            ("q_learningvssb3_leagueresults.csv", "PPO", "j0"),
        ],
        "PPO": [
            ("randomvssb3_leagueresults.csv", "Random", "j1"),
            ("rationalvssb3_leagueresults.csv", "Rational", "j1"),
            ("q_learningvssb3_leagueresults.csv", "Q-Learning", "j1"),
        ],
    }

    for agent_name, matchups in agents.items():
        print(f"\n##### Agente {agent_name}\n")
        print("| Oponente   | Win Rate | Puntos Prom. | Manos Jugadas | Manos Ganadas |")
        print("| ---------- | -------- | ------------ | ------------- | ------------- |")
        for csv_file, opp_name, persp in matchups:
            ci = compute_matchup_ci(csv_file, agent_name, persp)
            print(f"| {opp_name:<10} | {ci['winrate']:<22} | {ci['points']:<18} | {ci['hands_played']:<18} | {ci['hands_won']:<18} |")
        print(f"\n_Nota: Los intervalos en Win Rate corresponden a IC95% Wilson. "
              f"Para las demás métricas se reporta media ± 1.96 × error estándar (n=1000)._")


# ── Sección 3.6.1.3: Q-Learning enfrentamientos directos ──────────────────

def print_qlearning_head_to_head():
    print("\n" + "=" * 80)
    print("SECCIÓN 3.6.1.3 — Q-LEARNING ENFRENTAMIENTOS DIRECTOS CON IC95%")
    print("=" * 80)

    # exp1 vs exp2: summary_exp1exp2 → J0=exp1, J1=exp2 → 536-464
    # exp1 vs exp3: summary_exp1exp3 → J0=exp1, J1=exp3 → 649-351
    # exp2 vs exp3: summary_exp2exp3 → J0=exp2, J1=exp3 → 537-463

    matchups = {
        "exp1 vs exp2": ("q_learningvsq_learningresults.csv", 536, 464, 1000),
        "exp1 vs exp3": ("q_learningvsq_learningresults.csv", 649, 351, 1000),
        "exp2 vs exp3": ("q_learningvsq_learningresults.csv", 537, 463, 1000),
    }

    # Usamos los summaries directamente
    data = [
        ("Exp. 1 vs Exp. 2", 536, 1000),
        ("Exp. 1 vs Exp. 3", 649, 1000),
        ("Exp. 2 vs Exp. 3", 537, 1000),
    ]

    # También el resultado de semillas
    print("\n**Enfrentamiento entre semillas (Exp. 2):**")
    p, lo, hi = wilson_ci(499, 1000)
    print(f"  Semilla 789987 vs 321123: 499/1000 = {fmt_pct_ci(p, lo, hi)}")
    print(f"  El intervalo contiene el 50%, consistente con políticas equivalentes.\n")

    print("| Win rate (fila vs columna)          | Exp. 1 (self-play) | Exp. 2 (vs racional) | Exp. 3 (mixto) |")
    print("| ----------------------------------- | ------------------ | -------------------- | -------------- |")

    # Exp1 row
    p12, lo12, hi12 = wilson_ci(536, 1000)
    p13, lo13, hi13 = wilson_ci(649, 1000)
    print(f"| **Exp. 1 (self-play)**     | —                  | {fmt_pct_ci(p12, lo12, hi12):<22} | {fmt_pct_ci(p13, lo13, hi13):<22} |")

    # Exp2 row (complementarios)
    p21, lo21, hi21 = wilson_ci(464, 1000)
    p23, lo23, hi23 = wilson_ci(537, 1000)
    print(f"| **Exp. 2 (vs racional)**   | {fmt_pct_ci(p21, lo21, hi21):<22} | —                    | {fmt_pct_ci(p23, lo23, hi23):<22} |")

    # Exp3 row
    p31, lo31, hi31 = wilson_ci(351, 1000)
    p32, lo32, hi32 = wilson_ci(463, 1000)
    print(f"| **Exp. 3 (mixto)**         | {fmt_pct_ci(p31, lo31, hi31):<22} | {fmt_pct_ci(p32, lo32, hi32):<22} | —              |")

    print(f"\n_Nota: Los valores entre corchetes corresponden al IC95% Wilson (n=1000 partidas por enfrentamiento)._")


# ── Sección 3.6.2.3: PPO enfrentamientos directos ─────────────────────────

def print_ppo_head_to_head():
    print("\n" + "=" * 80)
    print("SECCIÓN 3.6.2.3 — PPO ENFRENTAMIENTOS DIRECTOS CON IC95%")
    print("=" * 80)

    # exp1 vs exp2: 491-509
    # exp1 vs exp3: 557-443
    # exp2 vs exp3: 527-473

    print("\n| Win rate (fila vs columna)          | Exp. 1 (self-play) | Exp. 2 (equilibrada) | Exp. 3 (heurísticos) |")
    print("| ----------------------------------- | ------------------ | -------------------- | -------------------- |")

    # Exp1 row
    p12, lo12, hi12 = wilson_ci(491, 1000)
    p13, lo13, hi13 = wilson_ci(557, 1000)
    print(f"| **Exp. 1 (self-play)**              | —                  | {fmt_pct_ci(p12, lo12, hi12):<22} | {fmt_pct_ci(p13, lo13, hi13):<22} |")

    # Exp2 row
    p21, lo21, hi21 = wilson_ci(509, 1000)
    p23, lo23, hi23 = wilson_ci(527, 1000)
    print(f"| **Exp. 2 (equilibrada)**            | {fmt_pct_ci(p21, lo21, hi21):<22} | —                    | {fmt_pct_ci(p23, lo23, hi23):<22} |")

    # Exp3 row
    p31, lo31, hi31 = wilson_ci(443, 1000)
    p32, lo32, hi32 = wilson_ci(473, 1000)
    print(f"| **Exp. 3 (énfasis en heurísticos)** | {fmt_pct_ci(p31, lo31, hi31):<22} | {fmt_pct_ci(p32, lo32, hi32):<22} | —                    |")

    print(f"\n_Nota: Los valores entre corchetes corresponden al IC95% Wilson (n=1000 partidas por enfrentamiento)._")

    # Análisis de solapamiento
    print("\n**Análisis de solapamiento de ICs:**")
    pairs = [
        ("Exp.1 vs Exp.2", 491, 1000),
        ("Exp.1 vs Exp.3", 557, 1000),
        ("Exp.2 vs Exp.3", 527, 1000),
    ]
    for name, w, n in pairs:
        p, lo, hi = wilson_ci(w, n)
        contains_50 = lo <= 0.5 <= hi
        print(f"  {name}: {fmt_pct_ci(p, lo, hi)} — {'contiene' if contains_50 else 'NO contiene'} el 50%")


# ── Gráficos con bandas de confianza ──────────────────────────────────────

def plot_eval_curve_with_ci(csv_file, title, xlabel, output_path,
                            x_col="episode", wins_col="wins",
                            losses_col="losses", x_scale=1):
    """Genera gráfico de curva de evaluación con banda IC95% Wilson."""
    rows = read_eval_csv(csv_file)

    x_vals = []
    wr_vals = []
    lo_vals = []
    hi_vals = []

    for row in rows:
        x = int(row[x_col]) * x_scale
        wins = int(row[wins_col])
        losses = int(row[losses_col])
        n = wins + losses
        p, lo, hi = wilson_ci(wins, n)
        x_vals.append(x)
        wr_vals.append(p * 100)
        lo_vals.append(lo * 100)
        hi_vals.append(hi * 100)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_vals, wr_vals, color="#2196F3", linewidth=2, label="Win Rate")
    ax.fill_between(x_vals, lo_vals, hi_vals, alpha=0.2, color="#2196F3",
                    label="IC95% Wilson")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico guardado: {output_path}")


def generate_all_eval_plots():
    print("\n" + "=" * 80)
    print("GENERANDO GRÁFICOS CON BANDAS DE CONFIANZA")
    print("=" * 80)

    ci_dir = os.path.join(PLOTS_DIR, "ci_bands")
    os.makedirs(ci_dir, exist_ok=True)

    # ── Q-Learning: curvas de evaluación ──
    qlearn_evals = [
        ("exp1_selfplay_vs_random.csv", "Q-Learning Exp.1 (self-play) vs Random", "Episodios"),
        ("exp1_selfplay_vs_rational.csv", "Q-Learning Exp.1 (self-play) vs Rational", "Episodios"),
        ("exp2_vs_rational_seed789987_vs_random.csv", "Q-Learning Exp.2 (semilla 789987) vs Random", "Episodios"),
        ("exp2_vs_rational_seed789987_vs_rational.csv", "Q-Learning Exp.2 (semilla 789987) vs Rational", "Episodios"),
        ("exp2_vs_rational_seed321123_vs_random.csv", "Q-Learning Exp.2 (semilla 321123) vs Random", "Episodios"),
        ("exp2_vs_rational_seed321123_vs_rational.csv", "Q-Learning Exp.2 (semilla 321123) vs Rational", "Episodios"),
        ("exp3_mix_vs_random.csv", "Q-Learning Exp.3 (mixto) vs Random", "Episodios"),
        ("exp3_mix_vs_rational.csv", "Q-Learning Exp.3 (mixto) vs Rational", "Episodios"),
    ]

    for csv_file, title, xlabel in qlearn_evals:
        basename = os.path.splitext(csv_file)[0] + "_ci.png"
        output_path = os.path.join(ci_dir, basename)
        plot_eval_curve_with_ci(
            csv_file, title, xlabel, output_path,
            x_col="episode", wins_col="wins", losses_col="losses"
        )

    # ── PPO: curvas de evaluación ──
    ppo_evals = [
        ("sb3_league_eval_vs_random.csv", "PPO Exp.3 (énfasis heurísticos) vs Random", "Timesteps"),
        ("sb3_league_eval_vs_rational.csv", "PPO Exp.3 (énfasis heurísticos) vs Rational", "Timesteps"),
    ]

    for csv_file, title, xlabel in ppo_evals:
        basename = os.path.splitext(csv_file)[0] + "_ci.png"
        output_path = os.path.join(ci_dir, basename)
        plot_eval_curve_with_ci(
            csv_file, title, xlabel, output_path,
            x_col="timesteps", wins_col="wins", losses_col="losses"
        )

    # ── Q-Learning: gráfico combinado por experimento ──
    generate_combined_qlearn_plot(ci_dir)
    generate_combined_ppo_plot(ci_dir)


def generate_combined_qlearn_plot(ci_dir):
    """Gráfico combinado de convergencia Q-Learning: todos los experimentos."""
    experiments = {
        "Exp.1 (self-play)": {
            "vs_random": "exp1_selfplay_vs_random.csv",
            "vs_rational": "exp1_selfplay_vs_rational.csv",
            "color": "#F44336",
        },
        "Exp.2 s.789987": {
            "vs_random": "exp2_vs_rational_seed789987_vs_random.csv",
            "vs_rational": "exp2_vs_rational_seed789987_vs_rational.csv",
            "color": "#4CAF50",
        },
        "Exp.2 s.321123": {
            "vs_random": "exp2_vs_rational_seed321123_vs_random.csv",
            "vs_rational": "exp2_vs_rational_seed321123_vs_rational.csv",
            "color": "#8BC34A",
        },
        "Exp.3 (mixto)": {
            "vs_random": "exp3_mix_vs_random.csv",
            "vs_rational": "exp3_mix_vs_rational.csv",
            "color": "#2196F3",
        },
    }

    for opponent_label, csv_key in [("Random", "vs_random"), ("Rational", "vs_rational")]:
        fig, ax = plt.subplots(figsize=(14, 7))

        for exp_name, exp_data in experiments.items():
            rows = read_eval_csv(exp_data[csv_key])
            x_vals, wr_vals, lo_vals, hi_vals = [], [], [], []
            for row in rows:
                x = int(row["episode"])
                wins = int(row["wins"])
                losses = int(row["losses"])
                n = wins + losses
                p, lo, hi = wilson_ci(wins, n)
                x_vals.append(x)
                wr_vals.append(p * 100)
                lo_vals.append(lo * 100)
                hi_vals.append(hi * 100)

            color = exp_data["color"]
            ax.plot(x_vals, wr_vals, color=color, linewidth=2, label=exp_name)
            ax.fill_between(x_vals, lo_vals, hi_vals, alpha=0.15, color=color)

        ax.set_xlabel("Episodios", fontsize=12)
        ax.set_ylabel("Win Rate (%)", fontsize=12)
        ax.set_title(f"Convergencia Q-Learning vs {opponent_label} (con IC95% Wilson)", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        output_path = os.path.join(ci_dir, f"q_learning_convergence_vs_{opponent_label.lower()}_ci.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Gráfico combinado Q-Learning vs {opponent_label}: {output_path}")


def generate_combined_ppo_plot(ci_dir):
    """Gráfico de convergencia PPO con bandas de IC."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, (opponent_label, csv_file) in zip(axes, [
        ("Random", "sb3_league_eval_vs_random.csv"),
        ("Rational", "sb3_league_eval_vs_rational.csv"),
    ]):
        rows = read_eval_csv(csv_file)
        x_vals, wr_vals, lo_vals, hi_vals = [], [], [], []
        for row in rows:
            x = int(row["timesteps"])
            wins = int(row["wins"])
            losses = int(row["losses"])
            n = wins + losses
            p, lo, hi = wilson_ci(wins, n)
            x_vals.append(x)
            wr_vals.append(p * 100)
            lo_vals.append(lo * 100)
            hi_vals.append(hi * 100)

        ax.plot(x_vals, wr_vals, color="#9C27B0", linewidth=2, label="Win Rate")
        ax.fill_between(x_vals, lo_vals, hi_vals, alpha=0.2, color="#9C27B0",
                        label="IC95% Wilson")
        ax.set_xlabel("Timesteps", fontsize=12)
        ax.set_ylabel("Win Rate (%)", fontsize=12)
        ax.set_title(f"PPO Exp.3 vs {opponent_label}", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

    plt.suptitle("Convergencia PPO (con IC95% Wilson)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path = os.path.join(ci_dir, "ppo_convergence_combined_ci.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico combinado PPO: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_global_results()
    print_qlearning_head_to_head()
    print_ppo_head_to_head()
    generate_all_eval_plots()
