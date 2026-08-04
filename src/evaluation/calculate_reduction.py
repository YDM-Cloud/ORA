import pandas as pd
from pathlib import Path

# =====================================================
# Paths
# =====================================================

ROOT = Path(__file__).resolve().parents[1].parents[0]
RESULT_DIR = Path(f"{ROOT}/results/energy_case")
OPT_FILE = RESULT_DIR / "optimization_results.csv"
BASELINE_FILE = RESULT_DIR / "baseline_results.csv"
OUTPUT_DIR = RESULT_DIR / "reduction"


# =====================================================
# Load
# =====================================================

def load_data():
    opt = pd.read_csv(OPT_FILE)
    baseline = pd.read_csv(BASELINE_FILE)
    return opt, baseline


# =====================================================
# Merge baseline
# =====================================================

def merge_baseline(opt, baseline):
    baseline = baseline.rename(columns={
        "electricity_cost": "baseline_cost",
        "carbon_emission": "baseline_carbon",
        "sla_violation": "baseline_sla"
    })
    merged = opt.merge(baseline, on=["scenario", "day"], how="left")
    return merged


# =====================================================
# Reduction
# =====================================================

def calculate_reduction(df):
    df["cost_reduction_%"] = (df["baseline_cost"] - df["electricity_cost"]) / df["baseline_cost"] * 100
    df["carbon_reduction_%"] = (df["baseline_carbon"] - df["carbon_emission"]) / df["baseline_carbon"] * 100
    df["sla_change"] = df["sla_violation"] - df["baseline_sla"]
    return df


# =====================================================
# Runtime reduction
# =====================================================

def calculate_runtime_reduction(df):
    runtime_ref = df.groupby("algorithm")["runtime_seconds"].mean()
    if "DE" not in runtime_ref.index:
        raise ValueError("DE runtime reference not found")
    de_runtime = runtime_ref["DE"]
    df["runtime_reduction_vs_DE_%"] = (de_runtime - df["runtime_seconds"]) / de_runtime * 100
    return df


# =====================================================
# Summary
# =====================================================

def build_summary(df):
    summary = df.groupby("algorithm").agg(
        mean_cost_reduction=("cost_reduction_%", "mean"),
        mean_carbon_reduction=("carbon_reduction_%", "mean"),
        mean_runtime_reduction_vs_DE=("runtime_reduction_vs_DE_%", "mean"),
        mean_sla_change=("sla_change", "mean"),
        mean_fitness=("fitness", "mean"),
        mean_runtime=("runtime_seconds", "mean")
    ).reset_index()
    return summary


# =====================================================
# Main
# =====================================================

def run_calculate_reduction():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    opt, baseline = load_data()
    print("Optimization:", opt.shape)
    print("Baseline:", baseline.shape)
    print("Merge baseline...")
    merged = merge_baseline(opt, baseline)
    result = calculate_reduction(merged)
    result = calculate_runtime_reduction(result)
    result.to_csv(OUTPUT_DIR / "reduction_results.csv", index=False)
    summary = build_summary(result)
    summary.to_csv(OUTPUT_DIR / "reduction_summary.csv", index=False)
    rank = summary.sort_values("mean_cost_reduction", ascending=False)
    rank.to_csv(OUTPUT_DIR / "algorithm_reduction_rank.csv", index=False)
    print("\nReduction summary:")
    print(summary.to_string(index=False))
    print("\nSaved:", OUTPUT_DIR)
