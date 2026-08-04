from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon

# =====================================================
# Path
# =====================================================


ROOT = Path(__file__).resolve().parents[1].parents[0]
RESULT_DIR = ROOT / "results" / "uncertainty_test"
RESULT_FILE = RESULT_DIR / "uncertainty_results.csv"


# =====================================================
# Load
# =====================================================


def load_results():
    if not RESULT_FILE.exists():
        raise FileNotFoundError(RESULT_FILE)
    return pd.read_csv(RESULT_FILE)


# =====================================================
# Calculate uncertainty deviation
# =====================================================


def calculate_degradation(df):
    """
    Absolute uncertainty deviation
    D = |Fu-F0| / F0
    """
    reference = df.groupby("algorithm")["fitness"].mean().rename("fitness_reference")
    result = df.merge(reference, on="algorithm")
    result["fitness_deviation"] = (abs(result["fitness"] - result["fitness_reference"])) / \
                                  (result["fitness_reference"] + 1e-12)
    return result


# =====================================================
# Summary
# =====================================================


def summarize(df):
    summary = df.groupby(["algorithm", "uncertainty_case"]).agg({
        "fitness_deviation": ["mean", "std"],
        "energy_kWh": ["mean", "std"],
        "carbon_emission": ["mean", "std"],
        "sla_violation": ["mean", "std"]
    }).reset_index()
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    return summary


# =====================================================
# Overall uncertainty robustness
# =====================================================


def generate_robustness_rank(summary):
    rank = summary.groupby("algorithm").agg({
        "fitness_deviation_mean": "mean",
        "fitness_deviation_std": "mean"
    }).reset_index()
    rank["uncertainty_score"] = rank["fitness_deviation_mean"] + 0.5 * rank["fitness_deviation_std"]
    rank = rank.sort_values("uncertainty_score")
    rank["rank"] = range(1, len(rank) + 1)
    return rank


# =====================================================
# Combined uncertainty
# =====================================================


def generate_combined_rank(summary):
    combined = summary[summary["uncertainty_case"] == "Combined"].copy()
    combined["uncertainty_score"] = combined["fitness_deviation_mean"] + 0.5 * combined["fitness_deviation_std"]
    combined = combined.sort_values("uncertainty_score")
    combined["rank"] = range(1, len(combined) + 1)
    return combined


# =====================================================
# ORA vs DE comparison
# =====================================================


def generate_comparison(summary):
    return summary[summary["algorithm"].isin(["ORA", "DE"])]


# =====================================================
# Statistical test
# =====================================================


def statistical_test(df):
    results = []
    for case in df["uncertainty_case"].unique():
        ora = df[(df["algorithm"] == "ORA") & (df["uncertainty_case"] == case)]["fitness_deviation"].values
        de = df[(df["algorithm"] == "DE") & (df["uncertainty_case"] == case)]["fitness_deviation"].values

        if len(ora) != len(de):
            continue
        if len(ora) < 5:
            continue

        stat, p = wilcoxon(ora, de)
        results.append({
            "uncertainty_case": case,
            "sample_size": len(ora),
            "wilcoxon_stat": stat,
            "p_value": p
        })

    return pd.DataFrame(results)


# =====================================================
# Main
# =====================================================


def run_analyze_uncertainty():
    print("=" * 70)
    print("Uncertainty Robustness Analysis")
    print("=" * 70)
    raw = load_results()
    degradation = calculate_degradation(raw)
    degradation.to_csv(RESULT_DIR / "uncertainty_degradation.csv", index=False)
    summary = summarize(degradation)
    generate_robustness_rank(summary).to_csv(RESULT_DIR / "uncertainty_robustness_rank.csv", index=False)
    generate_combined_rank(summary).to_csv(RESULT_DIR / "uncertainty_combined_rank.csv", index=False)
    generate_comparison(summary).to_csv(RESULT_DIR / "uncertainty_comparison.csv", index=False)
    statistical_test(degradation).to_csv(RESULT_DIR / "uncertainty_statistics.csv", index=False)
    print()
    print("Uncertainty analysis finished.")
    print(RESULT_DIR)
