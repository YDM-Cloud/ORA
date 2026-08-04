import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.algorithms.DE import DE
from src.algorithms.GTO import GTO
from src.algorithms.MGO import MGO
from src.algorithms.MPC import MPC
from src.algorithms.PSO import PSO
from src.algorithms.ORA import ORA, ORA_NoArchive, ORA_NoResonance
from src.config import load_config, project_path
from src.optimization.objective import EnergySchedulingObjective
from src.evaluation.statistics import run_statistics
from src.evaluation.statistics_scheduling_performance import run_statistics_scheduling_performance
from src.evaluation.calculate_reduction import run_calculate_reduction

# =====================================================
# Configuration
# =====================================================

_CONFIG = load_config("energy_case")
_EXPERIMENT = _CONFIG["optimization"]
DATA_FILE = project_path(_CONFIG["paths"]["final_input"])
RESULT_DIR = project_path(_CONFIG["paths"]["result_dir"])

# =====================================================
# Algorithm configuration
# =====================================================

ALGORITHM_CLASSES = {
    "MPC": MPC,
    "ORA": ORA,
    "ORA_NoResonance": ORA_NoResonance,
    "ORA_NoArchive": ORA_NoArchive,
    "DE": DE,
    "PSO": PSO,
    "GTO": GTO,
    "MGO": MGO
}


def create_optimizers(dim, lb, ub, experiment):
    optimizers = {}
    for name in experiment["algorithms"]:
        kwargs = {}
        if name == "ORA" and "sensitivity" in experiment:
            kwargs["sensitivity"] = experiment["sensitivity"]
        optimizers[name] = ALGORITHM_CLASSES[name](
            experiment["population"], dim, lb, ub, experiment["max_iterations"], **kwargs)
    return optimizers


# =====================================================
# Data processing
# =====================================================

def convert_daily_hourly(day_df):
    numeric_columns = [
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
    hourly = day_df.set_index("timestamp")[numeric_columns].resample("1h").mean().reset_index()
    return hourly


# =====================================================
# Scenario selection
# =====================================================

def select_scenarios(df, experiment):
    all_scenarios = df["scenario"].unique()
    mode = experiment["scenario_mode"]
    if mode == "all":
        return all_scenarios
    if mode == "first":
        return [all_scenarios[0]]
    if mode == "selected":
        return experiment["selected_scenarios"]
    raise ValueError(f"Invalid scenario mode: {mode}")


def select_representative_days(df, sample_size, seed):
    reference_scenario = _CONFIG["scenario_model"]["scenarios"][0]["name"]
    reference = df[df["scenario"].eq(reference_scenario)].copy()
    if reference.empty:
        raise ValueError(f"Reference scenario not found: {reference_scenario}")
    reference["date"] = reference["timestamp"].dt.date
    daily = reference.groupby("date").agg(
        temperature=("temperature", "mean"),
        carbon_intensity=("carbon_intensity", "mean"),
        electricity_price=("electricity_price", "mean"))
    daily["season"] = daily.index.map(lambda date: (
        "spring" if date.month in (3, 4, 5)
        else "summer" if date.month in (6, 7, 8)
        else "autumn" if date.month in (9, 10, 11)
        else "winter"))

    if sample_size >= len(daily):
        result = daily.reset_index()
        result["selection_reason"] = "full_year"
        return result

    special_dates = {
        "high_temperature": set(daily.nlargest(2, "temperature").index),
        "low_carbon": set(daily.nsmallest(2, "carbon_intensity").index),
        "high_price": set(daily.nlargest(2, "electricity_price").index)
    }
    selected = set().union(*special_dates.values())
    season_order = ("spring", "summer", "autumn", "winter")
    season_quotas = {
        season: sample_size // 4 + (index < sample_size % 4) for index, season in enumerate(season_order)
    }
    rng = np.random.default_rng(seed)

    for season, quota in season_quotas.items():
        selected_in_season = sum(daily.loc[date, "season"] == season for date in selected)
        needed = quota - selected_in_season
        candidates = daily[daily["season"].eq(season) & ~daily.index.isin(selected)].index.to_numpy()
        if needed < 0 or len(candidates) < needed:
            raise ValueError(f"Cannot sample {quota} representative {season} days")
        selected.update(rng.choice(candidates, size=needed, replace=False))

    result = daily.loc[sorted(selected)].reset_index()
    result["selection_reason"] = "seasonal_random"
    for reason, dates in special_dates.items():
        mask = result["date"].isin(dates)
        result.loc[mask, "selection_reason"] += f"|{reason}"
    return result


# =====================================================
# Experiment
# =====================================================

def run_experiment(experiment=None, result_dir=None, enable_statistics=True, enable_calculate_reduction=False):
    experiment = _EXPERIMENT if experiment is None else experiment
    result_dir = RESULT_DIR if result_dir is None else Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("Loading dataset...")
    print("=" * 80)
    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    scenarios = select_scenarios(df, experiment)
    print("Scenarios:")
    [print(scenario) for scenario in scenarios]
    selected_days = select_representative_days(df, sample_size=experiment["max_days"], seed=2023)
    selected_dates = set(selected_days["date"])
    selected_days.to_csv(result_dir / "selected_days.csv", index=False)
    print(f"Selected {len(selected_days)} representative days")

    optimization_results = []
    daily_schedule = []
    convergence_results = []
    solution_records = []

    for run_id in range(experiment["runs"]):
        for scenario in scenarios:
            print("\nScenario:", scenario)
            scenario_df = df[df["scenario"] == scenario].copy()
            scenario_df["date"] = scenario_df["timestamp"].dt.date
            days = [item for item in scenario_df.groupby("date") if item[0] in selected_dates]
            if len(days) != len(selected_dates):
                raise ValueError(f"{scenario} does not cover every selected day")
            for day_index, (_, day_df) in enumerate(days):
                print("Day", day_index + 1)
                hourly_data = convert_daily_hourly(day_df)
                if len(hourly_data) != 24:
                    print("Skip incomplete day")
                    continue
                objective = EnergySchedulingObjective(hourly_data, experiment.get("objective_weights"))
                dim = objective.dimension
                lb = np.zeros(dim)
                ub = np.ones(dim)
                print("Optimization dimension:", dim)
                optimizers = create_optimizers(dim, lb, ub, experiment)
                for alg_name, optimizer in optimizers.items():
                    np.random.seed(2026 + run_id)
                    start = time.time()
                    best_pos, best_fit, curve = optimizer.optimize(objective)
                    runtime = time.time() - start
                    metrics = objective.evaluate_solution(best_pos)
                    day_str = str(hourly_data["timestamp"].iloc[0].date())
                    optimization_results.append({
                        "run": run_id,
                        "scenario": scenario,
                        "day": day_str,
                        "algorithm": alg_name,
                        "fitness": best_fit,
                        **metrics,
                        "runtime_seconds": runtime
                    })
                    daily_schedule.append({
                        "run": run_id,
                        "scenario": scenario,
                        "day": day_str,
                        "algorithm": alg_name,
                        "schedule": json.dumps(best_pos.tolist())
                    })
                    convergence_results.append({
                        "run": run_id,
                        "scenario": scenario,
                        "day": day_str,
                        "algorithm": alg_name,
                        "curve": json.dumps(curve.tolist())
                    })
                    print(f"{alg_name}: fitness={best_fit:.4e}, time={runtime:.2f}s")

    results_df = pd.DataFrame(optimization_results)
    results_df.to_csv(result_dir / "optimization_results.csv", index=False)
    pd.DataFrame(daily_schedule).to_csv(result_dir / "daily_schedule.csv", index=False)
    pd.DataFrame(convergence_results).to_csv(result_dir / "convergence_results.csv", index=False)
    summary = results_df.groupby(["scenario", "algorithm"]).agg(
        mean_fitness=("fitness", "mean"),
        std_fitness=("fitness", "std"),
        mean_runtime=("runtime_seconds", "mean")
    )
    metric_columns = [c for c in results_df.columns if c not in
                      ["run", "scenario", "day", "algorithm", "fitness", "runtime_seconds"]]

    for col in metric_columns:
        if np.issubdtype(results_df[col].dtype, np.number):
            summary[f"mean_{col}"] = results_df.groupby(["scenario", "algorithm"])[col].mean()

    summary = summary.sort_values("mean_fitness")
    summary["rank"] = summary.groupby(level="scenario")["mean_fitness"].rank(method="dense")
    summary.to_csv(result_dir / "optimization_summary.csv")
    if enable_statistics:
        statistics_dir = result_dir / "statistics"
        run_statistics(result_dir / "optimization_results.csv", statistics_dir)
    if enable_calculate_reduction:
        run_calculate_reduction()
    run_statistics_scheduling_performance(RESULT_DIR)
    print("\nFinished.")
    print(result_dir)


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    start = datetime.now()
    run_experiment(enable_calculate_reduction=True)
    print("Total time:", datetime.now() - start)
