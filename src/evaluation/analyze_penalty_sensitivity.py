from pathlib import Path
import pandas as pd

# =====================================================
# Path
# =====================================================


ROOT = Path(__file__).resolve().parents[1].parents[0]
RESULT_DIR = ROOT / "results" / "penalty_sensitivity"
RESULT_FILE = RESULT_DIR / "penalty_results.csv"


# =====================================================
# Summary
# =====================================================


def generate_summary(results):
    summary = results.groupby(["algorithm", "sla_weight"]).agg({
        "fitness": ["mean", "std"],
        "energy_kWh": "mean",
        "electricity_cost": "mean",
        "carbon_emission": "mean",
        "sla_violation": "mean"
    }).reset_index()
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    return summary


# =====================================================
# Sensitivity effect
# =====================================================


def generate_effect_analysis(summary):
    records = []
    metrics = [
        "fitness_mean",
        "energy_kWh_mean",
        "electricity_cost_mean",
        "carbon_emission_mean",
        "sla_violation_mean"
    ]
    for algorithm in summary["algorithm"].unique():
        data = summary[summary["algorithm"] == algorithm]

        for metric in metrics:
            values = data[metric]
            minimum = values.min()
            maximum = values.max()
            if abs(minimum) < 1e-12:
                change = 0
            else:
                change = (maximum - minimum) / abs(minimum)
            records.append({
                "algorithm": algorithm,
                "metric": metric,
                "min_value": minimum,
                "max_value": maximum,
                "relative_change": change
            })
    return pd.DataFrame(records)


# =====================================================
# Stability analysis
# =====================================================


def generate_stability(summary):
    stability = summary.groupby("algorithm").agg({
        "fitness_mean": ["mean", "std"],
        "fitness_std": "mean",
        "sla_violation_mean": ["mean", "std"]
    }).reset_index()
    stability.columns = ["_".join(col).strip("_") for col in stability.columns]
    return stability


# =====================================================
# Trade-off table
# =====================================================


def generate_tradeoff(summary):
    return summary[[
        "algorithm",
        "sla_weight",
        "fitness_mean",
        "energy_kWh_mean",
        "carbon_emission_mean",
        "sla_violation_mean"
    ]]


# =====================================================
# Main
# =====================================================


def run_analyze_penalty_sensitivity():
    print("=" * 80)
    print("Penalty robustness analysis")
    print("=" * 80)
    if not RESULT_FILE.exists():
        raise FileNotFoundError(RESULT_FILE)
    results = pd.read_csv(RESULT_FILE)
    print(f"Loaded samples: {len(results)}")
    summary = generate_summary(results)
    summary.to_csv(RESULT_DIR / "penalty_summary_analysis.csv", index=False)
    effect = generate_effect_analysis(summary)
    effect.to_csv(RESULT_DIR / "penalty_effect_analysis.csv", index=False)
    stability = generate_stability(summary)
    stability.to_csv(RESULT_DIR / "penalty_stability.csv", index=False)
    tradeoff = generate_tradeoff(summary)
    tradeoff.to_csv(RESULT_DIR / "penalty_tradeoff.csv", index=False)
    print()
    print("Generated:")
    print("penalty_summary_analysis.csv")
    print("penalty_effect_analysis.csv")
    print("penalty_stability.csv")
    print("penalty_tradeoff.csv")
    print()
    print(RESULT_DIR)
