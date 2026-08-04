import sys
import time
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
EXP = CONFIG["multinode_scalability"]
DATA_FILE = project_path(CONFIG["paths"]["final_input"])
RESULT_DIR = project_path(CONFIG["paths"]["multinode_result_dir"])
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


# =====================================================
# Multi-node generation
# =====================================================

def build_multi_node_data(hourly, nodes, seed):
    np.random.seed(seed)
    data_list = []
    load = hourly["request_rate"].values
    hours = len(hourly)
    weights = np.random.dirichlet(np.ones(nodes), size=hours)

    for node in range(nodes):
        data = hourly.copy()
        data["request_rate"] = load * weights[:, node] * nodes
        data["electricity_price"] *= np.random.uniform(0.8, 1.3)
        data["carbon_intensity"] *= np.random.uniform(0.7, 1.4)
        data["pue"] += np.random.uniform(0.05, 0.25)
        data["temperature"] += np.random.normal(0, 2, hours)
        data_list.append(data)

    return pd.concat(data_list, ignore_index=True)


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

def run_multinode_scalability():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("ORA Multi-node Scalability Experiment")
    print("=" * 80)
    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    results = []
    total = len(EXP["nodes"]) * len(EXP["selected_scenarios"]) * len(EXP["algorithms"]) * EXP["runs"]
    count = 0

    for nodes in EXP["nodes"]:
        print("\nNodes:", nodes)

        for scenario in EXP["selected_scenarios"]:
            print("Scenario:", scenario)
            scenario_df = df[df["scenario"] == scenario].copy()
            scenario_df["date"] = scenario_df["timestamp"].dt.date
            date = sorted(scenario_df["date"].unique())[0]
            hourly = convert_hourly(scenario_df[scenario_df["date"] == date])
            multi_data = build_multi_node_data(hourly, nodes, 2026)
            objective = EnergySchedulingObjective(multi_data)
            dim = objective.dimension
            print("Dimension:", dim)

            for algorithm in EXP["algorithms"]:
                for run in range(EXP["runs"]):
                    count += 1
                    print(f"[{count}/{total}] {algorithm} run={run + 1}")
                    np.random.seed(2026 + run)
                    optimizer = create_optimizer(algorithm, dim)
                    start = time.time()
                    best_pos, best_fit, curve = optimizer.optimize(objective)
                    runtime = time.time() - start
                    metrics = objective.evaluate_solution(best_pos)
                    results.append({
                        "scale": nodes,
                        "dimension": dim,
                        "scenario": scenario,
                        "algorithm": algorithm,
                        "run": run,
                        "fitness": best_fit,
                        "runtime_seconds": runtime,
                        **metrics
                    })
                    print(f"fitness={best_fit:.6e} time={runtime:.2f}s")

    result_file = RESULT_DIR / "multinode_results.csv"
    pd.DataFrame(results).to_csv(result_file, index=False)
    run_statistics(result_file, STAT_DIR, mode="scalability")
    print("\nFinished.")
    print(RESULT_DIR)


if __name__ == "__main__":
    start = datetime.now()
    run_multinode_scalability()
    print("Total time:", datetime.now() - start)
