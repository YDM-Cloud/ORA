import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_processing.run_pipeline import run_pipeline
from src.config import load_config, project_path
from src.optimization.objective import EnergySchedulingObjective

_CONFIG = load_config("energy_case")
_BASELINE = _CONFIG["baseline"]
DATA_FILE = project_path(_CONFIG["paths"]["final_input"])
RESULT_DIR = project_path(_CONFIG["paths"]["result_dir"])
RESULT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RESULT_DIR / "baseline_results.csv"

# =====================================================
# User setting
# =====================================================
# "all":
# all scenarios
# "first":
# first scenario only
SCENARIO_MODE = _BASELINE["scenario_mode"]

# None:
# full year
# integer:
# first N days
MAX_DAYS = _BASELINE["max_days"]


# =====================================================
# Convert 15min to hourly
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
# Scenario selection
# =====================================================

def select_scenarios(df):
    all_scenarios = df["scenario"].unique()
    if SCENARIO_MODE == "all":
        return all_scenarios
    elif SCENARIO_MODE == "first":
        return [all_scenarios[0]]
    else:
        raise ValueError("Invalid SCENARIO_MODE")


# =====================================================
# Generate baseline
# =====================================================

def generate_baseline():
    print("=" * 70)
    print("Loading dataset...")
    print("=" * 70)
    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    scenarios = select_scenarios(df)
    results = []

    for scenario in scenarios:
        print("Scenario:", scenario)
        scenario_df = df[df["scenario"] == scenario].copy()
        scenario_df["date"] = scenario_df["timestamp"].dt.date
        daily_groups = list(scenario_df.groupby("date"))

        if MAX_DAYS is not None:
            daily_groups = daily_groups[:MAX_DAYS]
        for day_index, (_, day_df) in enumerate(daily_groups):
            hourly_data = convert_daily_hourly(day_df)
            if len(hourly_data) != 24:
                continue
            objective = EnergySchedulingObjective(hourly_data)

            # =====================================
            # Baseline:
            # full AI workload execution
            # x = 1
            # =====================================

            x_baseline = np.ones(objective.dimension)
            metrics = objective.evaluate_solution(x_baseline)
            results.append({
                "scenario": scenario,
                "day": str(hourly_data["timestamp"].iloc[0].date()),
                "energy_kWh": metrics["energy_kWh"],
                "electricity_cost": metrics["electricity_cost"],
                "carbon_emission": metrics["carbon_emission"],
                "peak_power": metrics["peak_power"],
                "average_pue": metrics["average_pue"],
                "sla_violation": metrics["sla_violation"]
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_FILE, index=False)
    print()
    print("Saved:")
    print(OUTPUT_FILE)
    print("Rows:", len(result_df))


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    run_pipeline()
    generate_baseline()
