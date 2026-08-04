import sys
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
from src.evaluation.analyze_penalty_sensitivity import run_analyze_penalty_sensitivity
from src.algorithms.ORA import ORA
from src.algorithms.DE import DE

# =====================================================
# Configuration
# =====================================================


CONFIG = load_config("energy_case")
DEFAULT_CONFIG = load_config("default")
PENALTY_CONFIG = CONFIG["penalty_sensitivity"]
DATA_FILE = project_path(CONFIG["paths"]["final_input"])
OUTPUT_DIR = ROOT / "results" / "penalty_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALGORITHMS = PENALTY_CONFIG["algorithms"]
SCENARIOS = PENALTY_CONFIG["selected_scenarios"]
MAX_DAYS = PENALTY_CONFIG["max_days"]
RUNS = PENALTY_CONFIG["runs"]
SLA_WEIGHTS = PENALTY_CONFIG["sla_weights"]


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
# Day selection
# =====================================================


def select_days(data):
    days = data["timestamp"].dt.date.unique()
    if len(days) <= MAX_DAYS:
        return days
    rng = np.random.default_rng(42)
    return np.sort(rng.choice(days, MAX_DAYS, replace=False))


# =====================================================
# Algorithm initialization
# =====================================================


def build_optimizer(name, dimension):
    if name == "ORA":
        return ORA(
            pop_size=50,
            dim=dimension,
            lb=np.zeros(dimension),
            ub=np.ones(dimension),
            max_iter=500
        )
    elif name == "DE":
        return DE(
            pop_size=50,
            dim=dimension,
            lb=np.zeros(dimension),
            ub=np.ones(dimension),
            max_iter=500
        )
    else:
        raise ValueError(f"Unsupported algorithm: {name}")


# =====================================================
# Evaluation
# =====================================================


def evaluate_solution(objective, solution):
    metrics = objective.evaluate_solution(solution)
    metrics["fitness"] = objective(solution)
    return metrics


# =====================================================
# Main Experiment
# =====================================================


def run():
    start = time.time()
    print("=" * 80)
    print("Penalty Weight Sensitivity Analysis")
    print("=" * 80)
    data = pd.read_csv(DATA_FILE)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    tasks = []

    # =================================================
    # Prepare experiment tasks
    # =================================================

    for scenario in SCENARIOS:
        scenario_data = data[data["scenario"] == scenario].copy()
        selected_days = select_days(scenario_data)
        print(f"{scenario}: {len(selected_days)} days selected")
        for day in selected_days:
            for algorithm in ALGORITHMS:
                for sla_weight in SLA_WEIGHTS:
                    for run_id in range(RUNS):
                        tasks.append((scenario, day, algorithm, sla_weight, run_id))

    total = len(tasks)
    print()
    print(f"Total tasks: {total}")
    print()
    results = []

    # =================================================
    # Run experiments
    # =================================================

    for idx, (scenario, day, algorithm, sla_weight, run_id) in enumerate(tasks, start=1):
        print("-" * 80)
        print(f"[{idx}/{total}] {algorithm} | {scenario} | day={day} | SLA weight={sla_weight} | run={run_id}")
        day_df = data[(data["scenario"] == scenario) & (data["timestamp"].dt.date == day)].copy()
        hourly_data = convert_daily_hourly(day_df)

        if len(hourly_data) != 24:
            print("Skip incomplete day")
            continue

        # =================================================
        # Dynamic objective weights
        # =================================================

        default_weights = DEFAULT_CONFIG["objective"]["weights"]
        weights = {
            "energy": default_weights["energy"],
            "carbon": default_weights["carbon"],
            "cost": default_weights["cost"],
            "sla": sla_weight,
            "peak": default_weights["peak"],
            "pue": default_weights["pue"]
        }
        objective = EnergySchedulingObjective(hourly_data, weights=weights)
        optimizer = build_optimizer(algorithm, objective.dimension)
        solution = optimizer.optimize(objective)[0]
        solution = np.clip(solution, 0, 1)
        metrics = evaluate_solution(objective, solution)
        print(f"Fitness={metrics['fitness']:.6f} | SLA={metrics['sla_violation']:.6f}"
              f" | Carbon={metrics['carbon_emission']:.3f}")
        results.append({
            "scenario": scenario,
            "day": str(day),
            "algorithm": algorithm,
            "sla_weight": sla_weight,
            "run": run_id,
            **metrics
        })

    # =================================================
    # Save raw results
    # =================================================

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_DIR / "penalty_results.csv", index=False)
    print()
    print("Saved penalty_results.csv")

    # =================================================
    # Summary
    # =================================================

    summary = result_df.groupby(["algorithm", "sla_weight"]).agg({
        "fitness": ["mean", "std"],
        "energy_kWh": "mean",
        "electricity_cost": "mean",
        "carbon_emission": "mean",
        "sla_violation": "mean"
    }).reset_index()
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary.to_csv(OUTPUT_DIR / "penalty_summary.csv", index=False)
    print()
    print("Saved penalty_summary.csv")
    print()
    print("=" * 80)
    print("Penalty sensitivity finished.")
    print(f"Completed tasks: {len(result_df)}/{total}")
    print(f"Runtime: {time.time() - start:.2f}s")
    print(OUTPUT_DIR)
    print("=" * 80)


if __name__ == "__main__":
    run()
    run_analyze_penalty_sensitivity()
