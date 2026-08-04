import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, project_path
from src.optimization.objective import EnergySchedulingObjective
from src.algorithms.ORA import ORA
from src.algorithms.DE import DE
from src.algorithms.PSO import PSO
from src.algorithms.MGO import MGO
from src.evaluation.statistics import run_statistics

# =====================================================
# Configuration
# =====================================================

CONFIG = load_config("energy_case")
EXP = CONFIG["scalability"]
DATA_FILE = project_path(CONFIG["paths"]["final_input"])
RESULT_DIR = project_path(CONFIG["paths"]["scalability_result_dir"])
STAT_DIR = RESULT_DIR / "statistics"

# =====================================================
# Algorithms
# =====================================================

ALGORITHMS = {
    "ORA": ORA,
    "DE": DE,
    "PSO": PSO,
    "MGO": MGO
}

# =====================================================
# Data
# =====================================================

NUMERIC_COLUMNS = [
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


def convert_hourly(df):
    return df.set_index("timestamp")[NUMERIC_COLUMNS].resample("1h").mean().reset_index()


def build_horizon_data(scenario_df, horizon):
    scenario_df = scenario_df.copy()
    scenario_df["date"] = scenario_df["timestamp"].dt.date
    days = sorted(scenario_df["date"].unique())
    selected_days = days[:horizon // 24]
    data = scenario_df[scenario_df["date"].isin(selected_days)]
    return convert_hourly(data)


# =====================================================
# Optimizer
# =====================================================

def create_optimizer(name, dim):
    lb = np.zeros(dim)
    ub = np.ones(dim)
    return ALGORITHMS[name](EXP["population"], dim, lb, ub, EXP["max_iterations"])


# =====================================================
# Experiment
# =====================================================

def run_scalability():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("ORA Scalability Experiment")
    print("=" * 80)
    print("\nLoading dataset...")
    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print("Dataset shape:", df.shape)
    print("Scenarios:", EXP["selected_scenarios"])
    print("Horizons:", EXP["horizons"])
    print("Algorithms:", EXP["algorithms"])
    print("Runs:", EXP["runs"])
    total_tasks = len(EXP["horizons"]) * len(EXP["selected_scenarios"]) * len(EXP["algorithms"]) * EXP["runs"]
    current_task = 0
    results = []
    convergence = []

    for horizon in EXP["horizons"]:
        print("\n" + "=" * 80)
        print(f"Horizon: {horizon} h")
        print("=" * 80)

        for scenario in EXP["selected_scenarios"]:
            print(f"\nScenario: {scenario}")
            scenario_df = df[df["scenario"] == scenario].copy()
            hourly = build_horizon_data(scenario_df, horizon)
            objective = EnergySchedulingObjective(hourly)
            dim = objective.dimension
            print("Optimization dimension:", dim)

            for algorithm in EXP["algorithms"]:
                print(f"\nAlgorithm: {algorithm}")
                for run in range(EXP["runs"]):
                    current_task += 1
                    print(f"[{current_task}/{total_tasks}] Run {run + 1}/{EXP['runs']}")
                    np.random.seed(EXP["random_seed"] + run)
                    optimizer = create_optimizer(algorithm, dim)
                    start = time.time()
                    best_pos, best_fit, curve = optimizer.optimize(objective)
                    runtime = time.time() - start
                    metrics = objective.evaluate_solution(best_pos)
                    results.append({
                        "scale": horizon,
                        "dimension": dim,
                        "scenario": scenario,
                        "algorithm": algorithm,
                        "run": run,
                        "fitness": best_fit,
                        "runtime_seconds": runtime,
                        **metrics
                    })
                    convergence.append({
                        "scale": horizon,
                        "dimension": dim,
                        "scenario": scenario,
                        "algorithm": algorithm,
                        "run": run,
                        "curve": json.dumps(curve.tolist())
                    })
                    print(f"Result | fitness={best_fit:.6e} | runtime={runtime:.2f}s")

    print("\nSaving results...")
    result_file = RESULT_DIR / "scalability_results.csv"
    pd.DataFrame(results).to_csv(result_file, index=False)
    pd.DataFrame(convergence).to_csv(RESULT_DIR / "scalability_convergence.csv", index=False)
    print("Results saved:")
    print(result_file)
    print("\nRunning statistical analysis...")
    run_statistics(result_file, STAT_DIR, mode="scalability")
    print("\nStatistics saved:")
    print(STAT_DIR)
    print("\n" + "=" * 80)
    print("ORA scalability experiment finished.")
    print("=" * 80)


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    start = datetime.now()
    run_scalability()
    print("\nTotal time:", datetime.now() - start)
