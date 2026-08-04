import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, project_path
from src.optimization.objective import EnergySchedulingObjective
from src.evaluation.analyze_slsqp import run_analyze_slsqp

# =====================================================
# Configuration
# =====================================================


CONFIG = load_config("energy_case")
SLSQP_CONFIG = CONFIG["slsqp_baseline"]
DATA_FILE = project_path(CONFIG["paths"]["final_input"])
OUTPUT_DIR = ROOT / "results" / "slsqp_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCENARIOS = SLSQP_CONFIG["selected_scenarios"]
MAX_DAYS = SLSQP_CONFIG["max_days"]
RUNS = SLSQP_CONFIG["runs"]
MAX_ITER = SLSQP_CONFIG["max_iter"]
INITIAL_SOLUTION = SLSQP_CONFIG["initial_solution"]
FTOL = SLSQP_CONFIG["ftol"]


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
    days = (data["timestamp"].dt.date.unique())
    if len(days) <= MAX_DAYS:
        return days
    rng = np.random.default_rng(42)
    selected = rng.choice(days, size=MAX_DAYS, replace=False)
    return np.sort(selected)


# =====================================================
# SLSQP solver
# =====================================================


def optimize_slsqp(objective):
    dim = objective.dimension
    x0 = np.ones(dim) * INITIAL_SOLUTION
    bounds = [(0, 1) for _ in range(dim)]
    history = []

    def callback(xk):
        history.append(objective(xk))

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        callback=callback,
        options={"maxiter": MAX_ITER, "ftol": FTOL, "disp": False})

    # ensure final point recorded

    history.append(objective(result.x))
    return result, history


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
    print("SLSQP baseline optimization")
    print("=" * 80)
    data = pd.read_csv(DATA_FILE)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    tasks = []

    for scenario in SCENARIOS:
        scenario_data = data[data["scenario"] == scenario].copy()
        selected_days = select_days(scenario_data)
        print(f"{scenario}: {len(selected_days)} days selected")

        for day in selected_days:
            for run_id in range(RUNS):
                tasks.append((scenario, day, run_id))

    total = len(tasks)
    print()
    print(f"Total tasks: {total}")
    results = []
    convergence = []

    for idx, (scenario, day, run_id) in enumerate(tasks, 1):
        print("-" * 80)
        print(f"[{idx}/{total}] {scenario} | {day} | run={run_id}")
        day_df = data[(data["scenario"] == scenario) & (data["timestamp"].dt.date == day)].copy()
        hourly = convert_daily_hourly(day_df)
        objective = EnergySchedulingObjective(hourly)
        result, history = optimize_slsqp(objective)
        solution = np.clip(result.x, 0, 1)
        metrics = evaluate_solution(objective, solution)
        print(f"Success={result.success}, Iter={result.nit}, Fitness={metrics['fitness']:.6f}")
        results.append({
            "scenario": scenario,
            "day": str(day),
            "run": run_id,
            "success": result.success,
            "iterations": result.nit,
            "message": result.message,
            **metrics
        })
        for i, value in enumerate(history):
            convergence.append({
                "scenario": scenario,
                "day": str(day),
                "run": run_id,
                "iteration": i,
                "fitness": value
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_DIR / "slsqp_results.csv", index=False)
    pd.DataFrame(convergence).to_csv(OUTPUT_DIR / "slsqp_convergence.csv", index=False)
    summary = result_df.groupby("scenario").mean(numeric_only=True).reset_index()
    summary.to_csv(OUTPUT_DIR / "slsqp_summary.csv", index=False)
    print()
    print("=" * 80)
    print("SLSQP finished")
    print(f"Completed: {len(result_df)}/{total}")
    print(f"Runtime: {time.time() - start:.2f}s")
    print(OUTPUT_DIR)
    print("=" * 80)


if __name__ == "__main__":
    run()
    run_analyze_slsqp()
