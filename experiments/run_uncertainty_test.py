import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

# =====================================================
# Path
# =====================================================


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, project_path
from src.optimization.objective import EnergySchedulingObjective
from src.evaluation.analyze_uncertainty import run_analyze_uncertainty

# =====================================================
# Configuration
# =====================================================


_CONFIG = load_config("energy_case")
UNCERTAINTY_CONFIG = _CONFIG["uncertainty_test"]
DATA_FILE = project_path(_CONFIG["paths"]["final_input"])
RESULT_DIR = project_path(_CONFIG["paths"]["result_dir"])
OUTPUT_DIR = ROOT / "results" / "uncertainty_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALGORITHMS = UNCERTAINTY_CONFIG["algorithms"]
CASES = UNCERTAINTY_CONFIG["cases"]
RANDOM_RUNS = UNCERTAINTY_CONFIG["random_runs"]


# =====================================================
# Data processing
# =====================================================


def convert_daily_hourly(day_df):
    columns = [
        "base_it_power_kw",
        "ai_power_kw",
        "request_rate",
        "temperature",
        "humidity",
        "cooling_degree",
        "sla_min_workload_ratio",
        "P_IT_kw",
        "P_cooling_kw",
        "P_facility_kw",
        "pue",
        "electricity_price",
        "carbon_intensity",
        "carbon_rate_kg_per_h",
        "carbon_emission_kg_per_interval"
    ]
    return day_df.set_index("timestamp")[columns].resample("1h").mean().reset_index()


# =====================================================
# Uncertainty injection
# =====================================================


def apply_uncertainty(data, case):
    uncertain = data.copy()
    factor_record = {}
    if case["type"] == "workload":
        factor = np.random.uniform(1 - case["error_range"], 1 + case["error_range"])
        uncertain["request_rate"] *= factor
        uncertain["ai_power_kw"] *= factor
        factor_record["workload_factor"] = factor
    elif case["type"] == "price":
        factor = np.random.uniform(1 - case["error_range"], 1 + case["error_range"])
        uncertain["electricity_price"] *= factor
        factor_record["price_factor"] = factor
    elif case["type"] == "carbon":
        factor = np.random.uniform(1 - case["error_range"], 1 + case["error_range"])
        uncertain["carbon_intensity"] *= factor
        factor_record["carbon_factor"] = factor
    elif case["type"] == "combined":
        workload_factor = np.random.uniform(1 - case["workload_error"], 1 + case["workload_error"])
        price_factor = np.random.uniform(1 - case["price_error"], 1 + case["price_error"])
        carbon_factor = np.random.uniform(1 - case["carbon_error"], 1 + case["carbon_error"])
        uncertain["request_rate"] *= workload_factor
        uncertain["ai_power_kw"] *= workload_factor
        uncertain["electricity_price"] *= price_factor
        uncertain["carbon_intensity"] *= carbon_factor
        factor_record["workload_factor"] = workload_factor
        factor_record["price_factor"] = price_factor
        factor_record["carbon_factor"] = carbon_factor
    return uncertain, factor_record


# =====================================================
# Evaluation
# =====================================================


def evaluate_solution(objective, solution):
    metrics = objective.evaluate_solution(solution)
    metrics["fitness"] = objective(solution)
    return metrics


# =====================================================
# Main
# =====================================================


def run():
    start = time.time()
    print("=" * 80)
    print("Uncertainty-aware Robustness Test")
    print("=" * 80)
    schedule_file = RESULT_DIR / "daily_schedule.csv"
    schedule_df = pd.read_csv(schedule_file)
    schedule_df = schedule_df[schedule_df["algorithm"].isin(ALGORITHMS)].copy()
    schedule_df["schedule"] = schedule_df["schedule"].apply(json.loads)
    print("Algorithms:", ALGORITHMS)
    print("Samples:", len(schedule_df))
    data = pd.read_csv(DATA_FILE)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    results = []
    total = len(schedule_df) * len(CASES) * RANDOM_RUNS
    counter = 0

    for _, row in schedule_df.iterrows():
        algorithm = row["algorithm"]
        scenario = row["scenario"]
        day = row["day"]
        run_id = row["run"]
        solution = np.array(row["schedule"], dtype=float)
        day_df = data[(data["scenario"] == scenario) & (data["timestamp"].dt.date == pd.to_datetime(day).date())].copy()

        if day_df.empty:
            continue
        hourly_data = convert_daily_hourly(day_df)

        # =============================================
        # Freeze original baseline
        # =============================================

        base_objective = EnergySchedulingObjective(hourly_data)
        baseline_reference = {
            "energy": base_objective.baseline_energy,
            "carbon": base_objective.baseline_carbon,
            "cost": base_objective.baseline_cost
        }

        for case_name, case in CASES.items():
            for seed in range(RANDOM_RUNS):
                counter += 1
                print(f"[{counter}/{total}] {algorithm} | {case_name} | seed={seed}")
                np.random.seed(seed)
                uncertain_data, factors = apply_uncertainty(hourly_data, case)
                objective = EnergySchedulingObjective(uncertain_data, baseline_reference=baseline_reference)
                metrics = evaluate_solution(objective, solution)
                results.append({
                    "run": run_id,
                    "scenario": scenario,
                    "day": day,
                    "algorithm": algorithm,
                    "uncertainty_case": case_name,
                    "seed": seed,
                    **factors,
                    **metrics
                })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_DIR / "uncertainty_results.csv", index=False)
    summary = result_df.groupby(["algorithm", "uncertainty_case"]).agg({
        "fitness": ["mean", "std"],
        "energy_kWh": ["mean", "std"],
        "carbon_emission": ["mean", "std"],
        "sla_violation": ["mean", "std"]
    }).reset_index()
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary.to_csv(OUTPUT_DIR / "uncertainty_summary.csv", index=False)
    print()
    print("Uncertainty test finished.")
    print(f"Runtime: {time.time() - start:.2f}s")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    run()
    run_analyze_uncertainty()
