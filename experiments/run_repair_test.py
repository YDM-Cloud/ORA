import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, project_path
from src.optimization.objective import EnergySchedulingObjective
from experiments.run_energy_case import convert_daily_hourly

# =====================================================
# Settings
# =====================================================


CONFIG = load_config("energy_case")
REPAIR_CONFIG = CONFIG["repair_test"]
ENERGY_RESULT_DIR = project_path(CONFIG["paths"]["result_dir"])
DATA_FILE = project_path(CONFIG["paths"]["final_input"])
OUTPUT_DIR = project_path('results/repair_test')
SCENARIOS = REPAIR_CONFIG["selected_scenarios"]
ALGORITHMS = REPAIR_CONFIG["algorithms"]
RUNS = REPAIR_CONFIG["runs"]
TOLERANCE = REPAIR_CONFIG["tolerance"]


# =====================================================
# Data Loading
# =====================================================


def load_dataset():
    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def get_hourly_data(df, scenario, day):
    scenario_df = df[df["scenario"].eq(scenario)].copy()
    scenario_df["date"] = scenario_df["timestamp"].dt.date
    day_df = scenario_df[scenario_df["date"].eq(pd.to_datetime(day).date())]
    if day_df.empty:
        raise ValueError(f"Missing data: {scenario}, {day}")
    return convert_daily_hourly(day_df)


# =====================================================
# Schedule Parser
# =====================================================


def parse_schedule(value):
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list):
        return np.asarray(value, dtype=float)

    value = str(value).replace("[", "").replace("]", "").replace(",", " ")
    return np.asarray([float(x) for x in value.split() if x.strip()], dtype=float)


# =====================================================
# SLA Repair
# =====================================================


def repair_sla_schedule(schedule, hourly_data):
    repaired = schedule.copy()
    sla_min = hourly_data["sla_min_workload_ratio"].to_numpy(dtype=float)
    repaired = np.maximum(repaired, sla_min)
    repaired = np.clip(repaired, 0, 1)
    return repaired


# =====================================================
# Evaluation
# =====================================================


def evaluate_schedule(hourly_data, schedule):
    objective = EnergySchedulingObjective(hourly_data)
    metrics = objective.evaluate_solution(schedule)
    fitness = objective(schedule)
    return {
        "fitness": fitness,
        "energy": metrics["energy_kWh"],
        "cost": metrics["electricity_cost"],
        "carbon": metrics["carbon_emission"],
        "sla": metrics["sla_violation"],
        "peak": metrics["peak_penalty"],
        "pue": metrics["pue_penalty"]
    }


# =====================================================
# Summary
# =====================================================


def build_summary(df):
    return df.groupby(["Scenario", "Algorithm"]).agg(
        Delta_Fitness_Mean=("Delta_Fitness", "mean"),
        Delta_Fitness_Std=("Delta_Fitness", "std"),
        SLA_Before=("Before_SLA", "mean"),
        SLA_After=("After_SLA", "mean"),
        Cost_Change=("Cost_Change", "mean"),
        Carbon_Change=("Carbon_Change", "mean")
    ).reset_index()


# =====================================================
# Main
# =====================================================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    optimization_file = ENERGY_RESULT_DIR / "optimization_results.csv"
    schedule_file = ENERGY_RESULT_DIR / "daily_schedule.csv"
    optimization_results = pd.read_csv(optimization_file)
    schedules = pd.read_csv(schedule_file)
    dataset = load_dataset()
    total_tasks = len(SCENARIOS) * len(ALGORITHMS) * RUNS
    current = 0
    records = []
    print("=" * 70)
    print("SLA Feasibility Repair Test")
    print("=" * 70)
    print(f"Scenarios : {len(SCENARIOS)}")
    print(f"Algorithms: {len(ALGORITHMS)}")
    print(f"Runs      : {RUNS}")
    print(f"Total task: {total_tasks}")
    print("=" * 70)

    for scenario in SCENARIOS:
        for algorithm in ALGORITHMS:
            for run in range(RUNS):
                current += 1
                print("\n" + "-" * 70)
                print(f"Task {current}/{total_tasks}")
                print(f"{scenario} | {algorithm} | Run {run}")
                schedule_row = schedules[
                    (schedules["scenario"] == scenario) &
                    (schedules["algorithm"] == algorithm) &
                    (schedules["run"] == run)]
                result_row = optimization_results[
                    (optimization_results["scenario"] == scenario) &
                    (optimization_results["algorithm"] == algorithm) &
                    (optimization_results["run"] == run)]
                if schedule_row.empty or result_row.empty:
                    print("Missing result, skip.")
                    continue

                day = result_row.iloc[0]["day"]
                hourly_data = get_hourly_data(dataset, scenario, day)
                schedule = parse_schedule(schedule_row.iloc[0]["schedule"])
                print("[1/3] Original evaluation")
                before = evaluate_schedule(hourly_data, schedule)
                print("[2/3] SLA repair")
                repaired = repair_sla_schedule(schedule, hourly_data)
                print("[3/3] Re-evaluation")
                after = evaluate_schedule(hourly_data, repaired)
                delta = (after["fitness"] - before["fitness"])
                impact = delta / max(abs(before["fitness"]), TOLERANCE) * 100
                records.append({
                    "Scenario": scenario,
                    "Algorithm": algorithm,
                    "Run": run,
                    "Day": day,
                    "Before_Fitness": before["fitness"],
                    "After_Fitness": after["fitness"],
                    "Delta_Fitness": delta,
                    "Repair_Impact_Percent": impact,
                    "Before_Energy": before["energy"],
                    "After_Energy": after["energy"],
                    "Energy_Change": after["energy"] - before["energy"],
                    "Before_Cost": before["cost"],
                    "After_Cost": after["cost"],
                    "Cost_Change": after["cost"] - before["cost"],
                    "Before_Carbon": before["carbon"],
                    "After_Carbon": after["carbon"],
                    "Carbon_Change": after["carbon"] - before["carbon"],
                    "Before_SLA": before["sla"],
                    "After_SLA": after["sla"]
                })
                print(f"Completed ΔF={delta:.8f}")

    results = pd.DataFrame(records)
    results.to_csv(OUTPUT_DIR / "repair_results.csv", index=False)
    summary = build_summary(results)
    summary.to_csv(OUTPUT_DIR / "repair_summary.csv", index=False)
    print("\n" + "=" * 70)
    print("Repair Test Finished")
    print("=" * 70)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
