import numpy as np
import pandas as pd
from src.config import load_config, project_path
from src.data_processing.feature_engineering import select_carbon_profile, select_weather_profile

_CONFIG = load_config("energy_case")
_PATHS = _CONFIG["paths"]
_MODEL = _CONFIG["scenario_model"]
FEATURE_FILE = project_path(_PATHS["feature_dataset"])
REQUEST_FILE = project_path(_PATHS["request_profiles"])
OFFLINE_FILE = project_path(_PATHS["offline_profiles"])
WEATHER_FILE = project_path(_PATHS["weather_features"])
CARBON_POOL_FILE = project_path(_PATHS["carbon_pool"])
SCENARIO_DIR = project_path(_PATHS["scenario_dir"])
OUTPUT_FILE = project_path(_PATHS["final_input"])


def _profile_value(profile, column, length, default):
    if column not in profile:
        return np.full(length, default)
    return np.resize(profile[column].to_numpy(), length)


def create_scenarios(base, workload_profiles, weather_profiles, carbon_profiles):
    normal_cooling = _profile_value(weather_profiles["normal"], "cooling_degree", len(base), 0)
    frames = []

    for spec in _MODEL["scenarios"]:
        result = base.copy()
        workload = workload_profiles[spec["workload"]]
        weather = weather_profiles[spec["weather"]]
        carbon = carbon_profiles[spec["carbon"]]
        result["scenario_id"] = spec["id"]
        result["scenario"] = spec["name"]
        result["scenario_name"] = spec["name"]
        result["workload_type"] = "offline_batch" if spec["workload"] == "offline_batch" else "online"
        result["request_category"] = {
            "online_normal": "normal",
            "online_peak": "peak",
            "offline_batch": "batch"
        }[spec["workload"]]
        result["weather_level"] = spec["weather"]
        result["carbon_level"] = spec["carbon"]
        result["ai_power_kw"] = _profile_value(workload, "ai_power_kw", len(base), 0)
        result["request_rate"] = _profile_value(workload, "request_rate", len(base), 0)
        result["temperature"] = _profile_value(weather, "temperature", len(base), 20)
        result["humidity"] = _profile_value(weather, "humidity", len(base), 50)
        result["cooling_degree"] = _profile_value(weather, "cooling_degree", len(base), 0)
        result["carbon_intensity"] = _profile_value(carbon, "carbon_intensity", len(base), 0)
        result["carbon_interpolated"] = _profile_value(carbon, "carbon_interpolated", len(base), 0).astype(int)
        result["sla_min_workload_ratio"] = spec["sla_min"]
        cooling_delta = np.maximum(result["cooling_degree"].to_numpy() - normal_cooling, 0)
        result["pue"] = result["base_pue"] * (1 + _MODEL["pue_per_cooling_degree"] * cooling_delta)
        result["P_IT_kw"] = result["base_it_power_kw"] + result["ai_power_kw"]
        result["P_facility_kw"] = result["P_IT_kw"] * result["pue"]
        result["P_cooling_kw"] = result["P_facility_kw"] - result["P_IT_kw"]
        result["carbon_rate_kg_per_h"] = result["P_facility_kw"] * result["carbon_intensity"]
        result["carbon_emission_kg_per_interval"] = result["carbon_rate_kg_per_h"] * 0.25
        result["ai_power_source"] = workload["source_file"].iloc[0] \
            if "source_file" in workload else "test_profile"
        result["weather_source"] = weather["weather_source"].iloc[0] \
            if "weather_source" in weather else "test_profile"
        result["weather_profile_year"] = _profile_value(weather, "weather_profile_year", len(base), 0)
        result["carbon_profile_date"] = _profile_value(carbon, "carbon_profile_date", len(base), "test_profile")
        frames.append(result)

    return pd.concat(frames, ignore_index=True)


def _load_workload_profiles():
    requests = pd.read_csv(REQUEST_FILE)
    offline = pd.read_csv(OFFLINE_FILE)
    online_normal = requests[requests["request_category"].eq("normal")].copy()
    online_peak = requests[requests["request_category"].eq("peak")].copy()
    if online_normal.empty or online_peak.empty or offline.empty:
        raise ValueError("Required measured workload profiles are missing")
    return {"online_normal": online_normal, "online_peak": online_peak, "offline_batch": offline}


def build_scenarios():
    base = pd.read_csv(FEATURE_FILE)
    base["timestamp"] = pd.to_datetime(base["timestamp"])
    weather = pd.read_csv(WEATHER_FILE)
    carbon = pd.read_csv(CARBON_POOL_FILE)
    workloads = _load_workload_profiles()
    weather_profiles = {
        level: select_weather_profile(weather, level, len(base)) for level in ("normal", "hot", "extreme")
    }
    carbon_profiles = {
        level: select_carbon_profile(carbon, level, len(base)) for level in ("normal", "high")
    }
    result = create_scenarios(
        base,
        workloads,
        weather_profiles,
        carbon_profiles
    )
    result = result.sort_values(["scenario_id", "timestamp"]).reset_index(drop=True)
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    for name, scenario in result.groupby("scenario", sort=False):
        scenario.to_csv(SCENARIO_DIR / f"{name}.csv", index=False)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {result['scenario'].nunique()} scenarios: {OUTPUT_FILE} ({len(result)} rows)")
    return result
