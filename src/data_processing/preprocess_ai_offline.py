from pathlib import Path
import numpy as np
import pandas as pd
from src.config import load_config, project_path

_CONFIG = load_config("energy_case")
PROFILE_STEPS = _CONFIG["scenario_model"]["profile_steps"]
DATASET_DIR = project_path("data/raw/llama3/inference_offline_llama3_70b")
OUTPUT_FILE = project_path(_CONFIG["paths"]["offline_profiles"])


def preprocess_ai_offline():
    metadata = pd.read_csv(DATASET_DIR / "metadata.csv")
    metadata = metadata.dropna(subset=[
        "batch_size",
        "elapsed",
        "max_output_tokens",
        "mean_power[W]",
        "path_run"
    ]).copy()
    metadata["compute_demand"] = metadata["batch_size"] * \
                                 metadata["max_output_tokens"] / \
                                 metadata["elapsed"].clip(lower=1e-6)
    target = metadata["compute_demand"].median()
    row = metadata.loc[(metadata["compute_demand"] - target).abs().idxmin()]
    source_file = Path(row["path_run"]).name
    power = pd.read_parquet(DATASET_DIR / "results" / source_file, columns=["power[W]"])["power[W]"]
    power = pd.to_numeric(power, errors="coerce").dropna().clip(lower=0)
    if power.empty:
        raise ValueError(f"Empty power trace: {source_file}")

    values = np.interp(np.linspace(0, len(power) - 1, PROFILE_STEPS), np.arange(len(power)), power.to_numpy())
    result = pd.DataFrame({
        "profile_step": np.arange(PROFILE_STEPS),
        "ai_power_kw": values / 1000,
        "request_rate": row["batch_size"] / row["elapsed"],
        "sequence_length": row["max_output_tokens"],
        "compute_demand": row["compute_demand"],
        "request_category": "batch",
        "workload_type": "offline_batch",
        "source_file": source_file
    })
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved offline profile: {OUTPUT_FILE} ({len(result)} rows)")
    return result
