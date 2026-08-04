import os
import pandas as pd
import numpy as np

# =====================================================
# Path
# =====================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "results", "energy_case")
OUTPUT_DIR = os.path.join(ROOT, "results", "feasibility_audit")
os.makedirs(OUTPUT_DIR, exist_ok=True)
INPUT_FILES = ["optimization_results.csv", "baseline_results.csv"]

# =====================================================
# Constraint configuration
# =====================================================

SLA_THRESHOLD = 1e-6
PEAK_MARGIN = 1.10


# =====================================================
# Utility
# =====================================================


def safe_mean(df, col):
    if col in df.columns:
        return df[col].mean()
    return 0.0


def calculate_dynamic_peak_limit(df):
    """
    Dynamic peak power constraint

    P_limit =
        95 percentile peak power
        * safety margin

    """
    if "peak_power" not in df.columns:
        return None
    p95 = df["peak_power"].quantile(0.95)
    return p95 * PEAK_MARGIN


# =====================================================
# Feasibility Metrics
# =====================================================


def calculate_sla_metrics(df):
    if "sla_violation" not in df.columns:
        return {"sla_violation_rate": 0, "mean_sla_violation": 0}

    violation = df["sla_violation"] > SLA_THRESHOLD
    return {
        "sla_violation_rate": violation.mean(),
        "mean_sla_violation": df["sla_violation"].mean()
    }


def calculate_peak_metrics(df, peak_limit):
    if "peak_power" not in df.columns or peak_limit is None:
        return {"peak_violation_rate": 0, "mean_peak_violation": 0}
    violation = np.maximum(df["peak_power"] - peak_limit, 0)
    return {
        "peak_violation_rate": (violation > 0).mean(),
        "mean_peak_violation": violation.mean()
    }


def calculate_full_feasibility(df, peak_limit):
    feasible = np.ones(len(df), dtype=bool)
    if "sla_violation" in df.columns:
        feasible &= (df["sla_violation"] <= SLA_THRESHOLD)
    if "peak_power" in df.columns and peak_limit is not None:
        feasible &= (df["peak_power"] <= peak_limit)
    return feasible.mean()


def calculate_mean_violation(df, peak_limit):
    violations = []
    if "sla_violation" in df.columns:
        violations.append(df["sla_violation"] / (df["sla_violation"].max() + 1e-12))
    if "peak_power" in df.columns and peak_limit is not None:
        peak = np.maximum(df["peak_power"] - peak_limit, 0)
        violations.append(peak / (peak.max() + 1e-12))
    if len(violations) == 0:
        return 0
    return pd.concat(violations, axis=1).mean(axis=1).mean()


# =====================================================
# Main
# =====================================================


def run():
    print("=" * 70)
    print("Running Feasibility Audit")
    print("=" * 70)
    datasets = []

    # ----------------------------
    # Load
    # ----------------------------

    for file in INPUT_FILES:
        path = os.path.join(RESULT_DIR, file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            datasets.append(df)

    if len(datasets) == 0:
        raise FileNotFoundError("No optimization results found.")
    data = pd.concat(datasets, ignore_index=True)

    # ----------------------------
    # Dynamic peak limit
    # ----------------------------

    peak_limit = calculate_dynamic_peak_limit(data)
    print("Dynamic peak limit:", peak_limit)
    summary = []
    details = []

    # ----------------------------
    # Algorithm analysis
    # ----------------------------

    for algorithm, group in data.groupby("algorithm"):
        sla_metrics = calculate_sla_metrics(group)
        peak_metrics = calculate_peak_metrics(group, peak_limit)
        row = {
            "algorithm": algorithm,
            "scenario": group["scenario"].iloc[0] if "scenario" in group.columns else "unknown",
            "objective": safe_mean(group, "fitness"),
            "energy_kWh": safe_mean(group, "energy_kWh"),
            "electricity_cost": safe_mean(group, "electricity_cost"),
            "carbon_emission": safe_mean(group, "carbon_emission"),
            "runtime_seconds": safe_mean(group, "runtime_seconds"),
            "peak_limit": peak_limit,
            **sla_metrics,
            **peak_metrics,
            "fully_feasible_rate": calculate_full_feasibility(group, peak_limit),
            "mean_violation": calculate_mean_violation(group, peak_limit)
        }
        summary.append(row)
        temp = group.copy()
        temp["algorithm"] = algorithm
        details.append(temp)

    summary = pd.DataFrame(summary)
    detail = pd.concat(details, ignore_index=True)
    summary.to_csv(os.path.join(OUTPUT_DIR, "feasibility_summary.csv"), index=False)
    detail.to_csv(os.path.join(OUTPUT_DIR, "feasibility_detail.csv"), index=False)
    print()
    print("Saved feasibility audit results.")
    print(summary)


if __name__ == "__main__":
    run()
