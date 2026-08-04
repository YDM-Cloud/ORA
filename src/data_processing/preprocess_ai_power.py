from pathlib import Path
import numpy as np
import pandas as pd
from src.config import load_config, project_path

_CONFIG = load_config("energy_case")
_PATHS = _CONFIG["paths"]
PROFILE_STEPS = _CONFIG["scenario_model"]["profile_steps"]
DATASET_DIR = project_path(_PATHS["ai_dataset"])
OUTPUT_FILE = project_path(_PATHS["ai_power_profiles"])


def classify_power_level(power):
    low = power.quantile(0.33)
    high = power.quantile(0.66)
    return np.select([power <= low, power <= high], ["low", "medium"], default="high")


def _power_profile(row):
    source_file = Path(row["path_run"]).name
    power = pd.read_parquet(DATASET_DIR / "results" / source_file, columns=["power[W]"])["power[W]"]
    power = pd.to_numeric(power, errors="coerce").dropna().clip(lower=0)
    if power.empty:
        raise ValueError(f"Empty power trace: {source_file}")

    values = np.interp(np.linspace(0, len(power) - 1, PROFILE_STEPS), np.arange(len(power)), power.to_numpy())
    return pd.DataFrame({
        "profile_step": np.arange(PROFILE_STEPS),
        "ai_power_kw": values / 1000,
        "request_rate": row["request_rate"],
        "power_level": row["power_level"],
        "workload_type": "online",
        "source_file": source_file
    })


def preprocess_ai_power():
    metadata = pd.read_csv(DATASET_DIR / "metadata.csv")
    metadata = metadata[metadata["run_completed"].eq(True)].copy()
    metadata["request_rate"] = metadata["request_rate_y"].fillna(metadata["request_rate_x"])
    metadata = metadata.dropna(subset=["mean_power[W]", "request_rate", "path_run"])
    metadata["power_level"] = classify_power_level(metadata["mean_power[W]"])
    profiles = []
    for level in ("low", "medium", "high"):
        candidates = metadata[metadata["power_level"].eq(level)]
        target = candidates["mean_power[W]"].median()
        row = candidates.loc[(candidates["mean_power[W]"] - target).abs().idxmin()]
        profiles.append(_power_profile(row))

    result = pd.concat(profiles, ignore_index=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved AI power profiles: {OUTPUT_FILE} ({len(result)} rows)")
    return result
