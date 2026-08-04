from pathlib import Path
import pandas as pd

# =====================================================
# Path
# =====================================================


ROOT = Path(__file__).resolve().parents[1].parents[0]
RESULT_DIR = ROOT / "results" / "slsqp_baseline"
RESULT_FILE = RESULT_DIR / "slsqp_results.csv"


# =====================================================
# Performance analysis
# =====================================================


def analyze_performance(results):
    analysis = results.groupby("scenario").agg({
        "success": "mean",
        "iterations": ["mean", "std"],
        "fitness": ["mean", "std"],
        "energy_kWh": "mean",
        "electricity_cost": "mean",
        "carbon_emission": "mean",
        "sla_violation": "mean",
        "peak_penalty": "mean",
        "pue_penalty": "mean"
    }).reset_index()
    analysis.columns = ["_".join(col).strip("_") for col in analysis.columns]
    analysis.rename(columns={"success_mean": "success_rate"}, inplace=True)
    return analysis


# =====================================================
# Feasibility analysis
# =====================================================


def analyze_feasibility(results):
    feasibility = results.groupby("scenario").agg({
        "success": ["mean", "sum"],
        "sla_violation": ["mean", "max"],
        "peak_penalty": ["mean", "max"],
        "pue_penalty": ["mean", "max"]
    }).reset_index()
    feasibility.columns = ["_".join(col).strip("_") for col in feasibility.columns]
    feasibility.rename(columns={
        "success_mean": "success_rate",
        "success_sum": "successful_runs"
    }, inplace=True)
    return feasibility


# =====================================================
# Scenario summary
# =====================================================


def scenario_summary(results):
    summary = results.groupby("scenario").agg({
        "fitness": "mean",
        "energy_kWh": "mean",
        "electricity_cost": "mean",
        "carbon_emission": "mean",
        "sla_violation": "mean",
        "iterations": "mean"
    }).reset_index()
    return summary


# =====================================================
# Main
# =====================================================


def run_analyze_slsqp():
    print("=" * 80)
    print("SLSQP baseline analysis")
    print("=" * 80)
    if not RESULT_FILE.exists():
        raise FileNotFoundError(RESULT_FILE)
    results = pd.read_csv(RESULT_FILE)
    print(f"Loaded samples: {len(results)}")
    performance = analyze_performance(results)
    performance.to_csv(RESULT_DIR / "slsqp_analysis.csv", index=False)
    feasibility = analyze_feasibility(results)
    feasibility.to_csv(RESULT_DIR / "slsqp_feasibility.csv", index=False)
    summary = scenario_summary(results)
    summary.to_csv(RESULT_DIR / "slsqp_scenario_summary.csv", index=False)
    print()
    print("Performance analysis:")
    print(performance.to_string(index=False))
    print()
    print("Feasibility analysis:")
    print(feasibility.to_string(index=False))
    print()
    print("Saved files:")
    print(RESULT_DIR / "slsqp_analysis.csv")
    print(RESULT_DIR / "slsqp_feasibility.csv")
    print(RESULT_DIR / "slsqp_scenario_summary.csv")
