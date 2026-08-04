from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# =====================================================
# Path
# =====================================================


ROOT = Path(__file__).resolve().parents[1].parents[0]
RESULT_DIR = ROOT / "results" / "stress_test"
SUMMARY_FILE = RESULT_DIR / "stress_summary.csv"
RESULT_FILE = RESULT_DIR / "stress_results.csv"


# =====================================================
# Normalization
# =====================================================


def min_max_normalize(series):
    minimum = series.min()
    maximum = series.max()
    if maximum - minimum < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - minimum) / (maximum - minimum)


# =====================================================
# Load
# =====================================================


def load_data():
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(SUMMARY_FILE)
    if not RESULT_FILE.exists():
        raise FileNotFoundError(RESULT_FILE)
    summary = pd.read_csv(SUMMARY_FILE)
    results = pd.read_csv(RESULT_FILE)
    return summary, results


# =====================================================
# Degradation analysis
# =====================================================


def calculate_degradation(summary):
    normal = summary[summary["stress_case"] == "Normal"][[
        "algorithm",
        "fitness",
        "energy_kWh",
        "carbon_emission",
        "electricity_cost",
        "sla_violation"
    ]].rename(columns={
        "fitness": "fitness_normal",
        "energy_kWh": "energy_normal",
        "carbon_emission": "carbon_normal",
        "electricity_cost": "cost_normal",
        "sla_violation": "sla_normal"
    })
    stress = summary[summary["stress_case"] != "Normal"].copy()
    degradation = stress.merge(normal, on="algorithm", how="left")
    degradation["fitness_degradation"] = (degradation["fitness"] - degradation["fitness_normal"]) / \
                                         (degradation["fitness_normal"] + 1e-12)
    degradation["energy_degradation"] = (degradation["energy_kWh"] - degradation["energy_normal"]) / \
                                        (degradation["energy_normal"] + 1e-12)
    degradation["carbon_degradation"] = (degradation["carbon_emission"] - degradation["carbon_normal"]) / \
                                        (degradation["carbon_normal"] + 1e-12)
    degradation["sla_degradation"] = degradation["sla_violation"] - degradation["sla_normal"]
    return degradation


# =====================================================
# Robustness score
# =====================================================


def calculate_robustness(df):
    df["fitness_deg_norm"] = min_max_normalize(df["fitness_degradation"])
    df["energy_deg_norm"] = min_max_normalize(df["energy_degradation"])
    df["carbon_deg_norm"] = min_max_normalize(df["carbon_degradation"])
    df["sla_deg_norm"] = min_max_normalize(df["sla_degradation"])
    df["robustness_score"] = 0.4 * df["fitness_deg_norm"] + \
                             0.2 * df["energy_deg_norm"] + \
                             0.2 * df["carbon_deg_norm"] + \
                             0.2 * df["sla_deg_norm"]
    return df


# =====================================================
# Ranking
# =====================================================


def generate_rank(df):
    rank = df.groupby("algorithm")["robustness_score"].mean().reset_index()
    rank = rank.sort_values("robustness_score")
    rank["rank"] = range(1, len(rank) + 1)
    return rank


def generate_optimizer_rank(df):
    optimizer = df[df["algorithm"].isin(["ORA", "DE"])]
    rank = optimizer.groupby("algorithm")["robustness_score"].mean().reset_index()
    rank = rank.sort_values("robustness_score")
    rank["rank"] = range(1, len(rank) + 1)
    return rank


# =====================================================
# ORA vs DE comparison
# =====================================================


def generate_comparison(df):
    return df[df["algorithm"].isin(["ORA", "DE"])][[
        "algorithm",
        "stress_case",
        "fitness_degradation",
        "energy_degradation",
        "carbon_degradation",
        "sla_degradation",
        "robustness_score"
    ]]


# =====================================================
# Correct statistical test
# =====================================================


def wilcoxon_test(results):
    records = []
    for stress_case in ["High_Workload", "Extreme_Workload"]:
        ora = results[(results["algorithm"] == "ORA") & (results["stress_case"] == stress_case)].copy()
        de = results[(results["algorithm"] == "DE") & (results["stress_case"] == stress_case)].copy()
        merge = ora.merge(de, on=["run", "scenario", "day"], suffixes=("_ora", "_de"))

        for metric in ["fitness", "sla_violation"]:
            x = merge[f"{metric}_ora"].values
            y = merge[f"{metric}_de"].values
            if len(x) < 5:
                continue

            # Difference
            diff = x - y

            # remove zero difference
            diff = diff[diff != 0]
            n = len(diff)
            if n < 5:
                continue

            # Wilcoxon test
            stat, p_value = wilcoxon(x, y)

            # ---------------------------------
            # Correct effect size calculation
            # ---------------------------------

            mean_w = (n * (n + 1)) / 4
            std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            z = (stat - mean_w) / std_w
            effect_size = z / np.sqrt(n)
            records.append({
                "stress_case": stress_case,
                "metric": metric,
                "sample_size": n,
                "wilcoxon_stat": stat,
                "p_value": p_value,
                "z_value": z,
                "effect_size_r": effect_size
            })

    return pd.DataFrame(records)


# =====================================================
# Main
# =====================================================


def run_analyze_stress():
    print("=" * 70)
    print("Stress Analysis")
    print("=" * 70)
    summary, results = load_data()
    degradation = calculate_degradation(summary)
    degradation = calculate_robustness(degradation)
    degradation.to_csv(RESULT_DIR / "stress_degradation.csv", index=False)
    generate_rank(degradation).to_csv(RESULT_DIR / "stress_robustness_rank.csv", index=False)
    generate_optimizer_rank(degradation).to_csv(RESULT_DIR / "optimization_robustness_rank.csv", index=False)
    generate_comparison(degradation).to_csv(RESULT_DIR / "stress_comparison.csv", index=False)
    wilcoxon_test(results).to_csv(RESULT_DIR / "stress_statistics.csv", index=False)
    print("Stress analysis completed.")
