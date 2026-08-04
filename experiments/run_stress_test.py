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
from src.evaluation.analyze_stress import run_analyze_stress

# =====================================================
# Configuration
# =====================================================


_CONFIG = load_config("energy_case")
STRESS_CONFIG = _CONFIG["stress_test"]
DATA_FILE = project_path(_CONFIG["paths"]["final_input"])
RESULT_DIR = project_path(_CONFIG["paths"]["result_dir"])
OUTPUT_DIR = ROOT / "results" / "stress_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET_ALGORITHMS = (STRESS_CONFIG["algorithms"])
STRESS_CASES = (STRESS_CONFIG["cases"])


# =====================================================
# Data Processing
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
    hourly = day_df.set_index("timestamp")[columns].resample("1h").mean().reset_index()
    return hourly


# =====================================================
# Stress transformation
# =====================================================


def apply_stress(data, config):
    stressed = data.copy()
    stressed["request_rate"] *= config["request_scale"]
    stressed["ai_power_kw"] *= config["power_scale"]
    stressed["sla_min_workload_ratio"] = np.minimum(stressed["sla_min_workload_ratio"] * config["sla_scale"], 1.0)
    return stressed


# =====================================================
# Evaluation
# =====================================================


def evaluate_solution(objective, solution):
    metrics = objective.evaluate_solution(solution)
    metrics["fitness"] = objective(solution)
    return metrics


# =====================================================
# Experiment
# =====================================================


def run():
    start = time.time()
    print("=" * 80)
    print("Coupled Workload-Energy Stress Test")
    print("=" * 80)
    schedule_file = RESULT_DIR / "daily_schedule.csv"

    if not schedule_file.exists():
        raise FileNotFoundError(schedule_file)

    schedule_df = pd.read_csv(schedule_file)
    schedule_df = schedule_df[schedule_df["algorithm"].isin(TARGET_ALGORITHMS)].copy()
    schedule_df["schedule"] = schedule_df["schedule"].apply(json.loads)
    print("Algorithms:", TARGET_ALGORITHMS)
    print("Samples:", len(schedule_df))
    data = pd.read_csv(DATA_FILE)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    results = []
    total = len(schedule_df) * len(STRESS_CASES)
    counter = 0

    for _, row in schedule_df.iterrows():
        algorithm = row["algorithm"]
        scenario = row["scenario"]
        day = row["day"]
        run_id = row["run"]
        solution = np.array(row["schedule"], dtype=float)
        day_df = data[
            (data["scenario"] == scenario) &
            (data["timestamp"].dt.date == pd.to_datetime(day).date())
            ].copy()

        if day_df.empty:
            continue
        hourly_data = convert_daily_hourly(day_df)

        for stress_name, stress in STRESS_CASES.items():
            counter += 1
            print(f"[{counter}/{total}] {algorithm} | {stress_name}")
            stressed_data = apply_stress(hourly_data, stress)
            objective = EnergySchedulingObjective(stressed_data)
            metrics = evaluate_solution(objective, solution)
            results.append({
                "run": run_id,
                "scenario": scenario,
                "day": day,
                "algorithm": algorithm,
                "stress_case": stress_name,
                "stress_level": stress["level"],
                "request_scale": stress["request_scale"],
                "power_scale": stress["power_scale"],
                "sla_scale": stress["sla_scale"],
                **metrics
            })

    # =================================================
    # Save raw results
    # =================================================

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "stress_results.csv", index=False)
    summary = pd.DataFrame(results) \
        .groupby(["algorithm", "stress_case", "stress_level"]) \
        .mean(numeric_only=True).reset_index()
    summary.to_csv(OUTPUT_DIR / "stress_summary.csv", index=False)
    print("Stress raw results saved.")

    # =================================================
    # Call evaluation
    # =================================================

    print()
    print("Running stress evaluation...")
    run_analyze_stress()
    print()
    print("=" * 80)
    print("Stress test finished.")
    print(f"Runtime: {time.time() - start:.2f}s")
    print(OUTPUT_DIR)
    print("=" * 80)


if __name__ == "__main__":
    run()
