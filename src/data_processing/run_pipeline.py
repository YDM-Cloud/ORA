import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import load_config, project_path
from src.data_processing.feature_engineering import build_dataset
from src.data_processing.prepare_dataset import prepare_dataset
from src.data_processing.preprocess_ai_offline import preprocess_ai_offline
from src.data_processing.preprocess_ai_power import preprocess_ai_power
from src.data_processing.preprocess_ai_rate import preprocess_ai_rate
from src.data_processing.preprocess_ai_weather import preprocess_ai_weather
from src.data_processing.preprocess_electricty import preprocess_electricity
from src.data_processing.preprocess_esif_2023 import preprocess_esif
from src.data_processing.preprocess_eia_carbon import preprocess_eia_carbon
from src.optimization.objective import EnergySchedulingObjective

_CONFIG = load_config("energy_case")
_OBJECTIVE_WEIGHTS = load_config("default")["objective"]["weights"]
_WINDOW = _CONFIG["data_window"]
WINDOW_START = pd.Timestamp(_WINDOW["start"])
WINDOW_END = WINDOW_START + pd.Timedelta(days=_WINDOW["days"])
EXPECTED_ROWS_PER_SCENARIO = _WINDOW["days"] * 24 * 4
EXPECTED_SCENARIOS = {scenario["name"] for scenario in _CONFIG["scenario_model"]["scenarios"]}
FINAL_DATASET = project_path(_CONFIG["paths"]["final_input"])
REPORT_FILE = FINAL_DATASET.with_name("data_quality_report.json")
MAX_CARBON_INTENSITY = _CONFIG["carbon"]["max_valid_carbon_intensity"]

REQUIRED_COLUMNS = (
    "timestamp",
    "scenario_id",
    "scenario",
    "scenario_name",
    "workload_type",
    "request_category",
    "weather_level",
    "carbon_level",
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
    "esif_interpolated",
    "carbon_interpolated",
    "price_source",
    "price_tariff",
    "tariff_effective_date",
    "price_source_file"
)
NUMERIC_COLUMNS = (
    "scenario_id",
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
    "esif_interpolated",
    "carbon_interpolated"
)
STEPS = (
    ("preprocess ESIF", preprocess_esif),
    ("extract measured AI power profiles", preprocess_ai_power),
    ("extract online request profiles", preprocess_ai_rate),
    ("extract offline batch profile", preprocess_ai_offline),
    ("preprocess measured weather", preprocess_ai_weather),
    ("preprocess PSCo commercial TOU tariff", preprocess_electricity),
    ("rebuild EIA carbon intensity", preprocess_eia_carbon),
    ("build physical feature dataset", build_dataset),
    ("build six energy scenarios", prepare_dataset)
)


def validate_dataset(data=FINAL_DATASET):
    if isinstance(data, (str, Path)):
        data = pd.read_csv(data)
    else:
        data = data.copy()

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data[list(NUMERIC_COLUMNS)] = data[list(NUMERIC_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    ordered = data.sort_values(["scenario", "timestamp"])
    intervals = ordered.groupby("scenario")["timestamp"].diff().dropna()
    coverage = ordered.groupby("scenario")["timestamp"].agg(rows="size", start="min", end="max")
    configured_scenarios = set(data["scenario"])
    scenario_means = data.groupby("scenario")[[
        "ai_power_kw",
        "request_rate",
        "temperature",
        "pue",
        "carbon_intensity"
    ]].mean().reindex(EXPECTED_SCENARIOS)
    normal = scenario_means.loc["Scenario_1_Online_Normal"]
    peak = scenario_means.loc["Scenario_2_Online_Peak"]
    hot = scenario_means.loc["Scenario_4_Hot_Weather"]
    high_carbon = scenario_means.loc["Scenario_5_High_Carbon"]
    stress = scenario_means.loc["Scenario_6_Stress_Test"]
    checks = {
        "required_values_present": not data[list(REQUIRED_COLUMNS)].isna().any().any(),
        "finite_numeric_values": bool(np.isfinite(data[list(NUMERIC_COLUMNS)].to_numpy()).all()),
        "six_configured_scenarios": configured_scenarios == EXPECTED_SCENARIOS,
        "scenario_drivers_are_separated": bool(
            peak["request_rate"] > normal["request_rate"]
            and peak["ai_power_kw"] >= normal["ai_power_kw"]
            and hot["temperature"] > normal["temperature"]
            and hot["pue"] > normal["pue"]
            and high_carbon["carbon_intensity"]
            > normal["carbon_intensity"]
            and stress["request_rate"] > normal["request_rate"]
            and stress["pue"] > normal["pue"]
            and stress["carbon_intensity"] > normal["carbon_intensity"]),
        "artificial_parameters_absent": not {"alpha", "GPU_number", }.intersection(data.columns),
        "unique_scenario_timestamps": not data.duplicated(["scenario", "timestamp"]).any(),
        "regular_15min_intervals": bool(len(intervals) and intervals.eq(pd.Timedelta(minutes=15)).all()),
        "configured_window_coverage": bool(
            coverage["rows"].eq(EXPECTED_ROWS_PER_SCENARIO).all()
            and coverage["start"].eq(WINDOW_START).all()
            and coverage["end"].eq(WINDOW_END - pd.Timedelta(minutes=15)).all()),
        "positive_base_power": bool(data["base_it_power_kw"].gt(0).all()),
        "nonnegative_ai_power": bool(data["ai_power_kw"].ge(0).all()),
        "nonnegative_request_rate": bool(data["request_rate"].ge(0).all()),
        "it_power_components_match": bool(
            np.allclose(data["P_IT_kw"], data["base_it_power_kw"] + data["ai_power_kw"])),
        "nonnegative_cooling_power": bool(data["P_cooling_kw"].ge(0).all()),
        "facility_matches_pue": bool(
            np.allclose(data["P_facility_kw"], data["P_IT_kw"] * data["pue"], rtol=1e-6, atol=1e-6)),
        "pue_in_physical_range": bool(data["pue"].between(1.0, 3.0).all()),
        "temperature_in_physical_range": bool(data["temperature"].between(-50, 60).all()),
        "humidity_in_physical_range": bool(data["humidity"].between(0, 100).all()),
        "price_in_model_range": bool(data["electricity_price"].between(-1.0, 10.0).all()),
        "real_price_provenance": bool(
            data["price_source"].eq("Public Service Company of Colorado").all()
            and data["price_tariff"].eq("Secondary Voltaic Time-of-Use Service Section B (SPVTOU-B)").all()
            and data["tariff_effective_date"].eq("2023-01-01").all()),
        "carbon_in_configured_range": bool(data["carbon_intensity"].between(0.0, MAX_CARBON_INTENSITY).all()),
        "sla_in_unit_interval": bool(data["sla_min_workload_ratio"].between(0, 1).all()),
        "binary_provenance_flags": bool(
            data["esif_interpolated"].isin((0, 1)).all()
            and data["carbon_interpolated"].isin((0, 1)).all())
    }
    failures = [name for name, passed in checks.items() if not passed]
    report = {
        "rows": int(len(data)),
        "scenarios": sorted(data["scenario"].unique().tolist()),
        "checks": checks,
        "esif_interpolated_fraction": float(data["esif_interpolated"].mean()),
        "carbon_interpolated_fraction": float(data["carbon_interpolated"].mean()),
        "objective_ready": not failures
    }
    if failures:
        raise ValueError("Dataset checks failed: " + ", ".join(failures))
    return report


def objective_smoke_test(data):
    scenario = data["scenario"].iloc[0]
    sample = data[data["scenario"].eq(scenario)].copy()
    sample["timestamp"] = pd.to_datetime(sample["timestamp"])
    first_day = sample["timestamp"].dt.normalize().min()
    sample = sample[sample["timestamp"].dt.normalize().eq(first_day)]
    columns = [
        "base_it_power_kw",
        "ai_power_kw",
        "request_rate",
        "pue",
        "electricity_price",
        "carbon_intensity",
        "sla_min_workload_ratio"
    ]
    hourly = (sample.set_index("timestamp")[columns].resample("1h").mean().reset_index())
    if len(hourly) != 24:
        raise ValueError(f"Objective smoke test needs 24 hourly rows, got {len(hourly)}")

    objective = EnergySchedulingObjective(hourly)
    solution = np.ones(objective.dimension)
    fitness = float(objective(solution))
    metrics = {name: float(value) for name, value in objective.evaluate_solution(solution).items()}
    if not np.isfinite([fitness, *metrics.values()]).all():
        raise ValueError("Objective smoke test produced non-finite values.")
    return {
        "scenario": scenario,
        "date": first_day.date().isoformat(),
        "hourly_rows": len(hourly),
        "dimension": objective.dimension,
        "fitness": fitness,
        "metrics": metrics,
        "objective_weight_sum": float(sum(_OBJECTIVE_WEIGHTS.values()))
    }


def run_pipeline():
    for name, function in STEPS:
        print(f"\n=== {name} ===")
        function()
    data = pd.read_csv(FINAL_DATASET)
    report = validate_dataset(data)
    report["objective_smoke_test"] = objective_smoke_test(data)
    report["scope"] = ("Measured Llama-3.1-70B power and request profiles, measured NLR "
                       "weather, EIA-930 carbon intensity, and PSCo SPVTOU-B tariff.")
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDataset: {FINAL_DATASET}")
    print(f"Quality report: {REPORT_FILE}")
    return report
