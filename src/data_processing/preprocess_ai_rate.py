from pathlib import Path
import numpy as np
import pandas as pd
from src.config import load_config, project_path

_CONFIG = load_config("energy_case")
PROFILE_STEPS = _CONFIG["scenario_model"]["profile_steps"]
DATASET_DIR = project_path("data/raw/llama3/inference_online_rate_llama3_70b")
OUTPUT_FILE = project_path(_CONFIG["paths"]["request_profiles"])


def classify_request_category(request_rate):
    normal_limit = request_rate.quantile(0.50)
    peak_limit = request_rate.quantile(0.90)
    return np.select(
        [request_rate <= normal_limit, request_rate <= peak_limit],
        ["normal", "burst"],
        default="peak"
    )


def _request_profile(row):
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
        "request_category": row["request_category"],
        "workload_type": "online",
        "source_file": source_file
    })


def preprocess_ai_rate():
    metadata = pd.read_csv(DATASET_DIR / "metadata.csv")
    metadata = metadata[metadata["run_completed"].eq(True)].copy()
    metadata = metadata.dropna(subset=["request_rate", "mean_power[W]", "path_run"])
    metadata["request_category"] = classify_request_category(metadata["request_rate"])
    profiles = []
    for category in ("normal", "burst", "peak"):
        candidates = metadata[metadata["request_category"].eq(category)]
        target = candidates["request_rate"].median()
        row = candidates.loc[(candidates["request_rate"] - target).abs().idxmin()]
        profiles.append(_request_profile(row))

    result = pd.concat(profiles, ignore_index=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved request profiles: {OUTPUT_FILE} ({len(result)} rows)")
    return result
