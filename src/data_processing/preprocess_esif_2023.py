import pandas as pd
from src.config import load_config, project_path

_CONFIG = load_config("energy_case")
_PATHS = _CONFIG["paths"]
_WINDOW = _CONFIG["data_window"]
WINDOW_START = pd.Timestamp(_WINDOW["start"])
WINDOW_END = WINDOW_START + pd.Timedelta(days=_WINDOW["days"])
INPUT_FILE = project_path(_PATHS["esif_raw"])
OUTPUT_FILE = project_path(_PATHS["esif_15min"])


def preprocess_esif():
    data = pd.read_parquet(
        INPUT_FILE, filters=[("ts", ">=", WINDOW_START.to_pydatetime()), ("ts", "<", WINDOW_END.to_pydatetime())]
    )
    data["ts"] = pd.to_datetime(data["ts"])
    columns = [
        "cooling_kw",
        "hvac_kw",
        "it_power_kw",
        "plug_and_light_kw",
        "pump_kw",
        "pue",
        "ere"
    ]
    full_time = pd.date_range(WINDOW_START, WINDOW_END - pd.Timedelta(minutes=15), freq="15min")
    result = data.set_index("ts")[columns].resample("15min").mean().reindex(full_time)
    interpolated = result.isna().any(axis=1)
    result = result.interpolate(method="time", limit_direction="both")
    if result[columns].isna().any().any():
        raise ValueError("Missing ESIF values remain after interpolation")

    result["esif_interpolated"] = interpolated.astype(int)
    result["facility_kw"] = result[[
        "it_power_kw",
        "cooling_kw",
        "hvac_kw",
        "pump_kw",
        "plug_and_light_kw"
    ]].sum(axis=1)
    result.index.name = "timestamp"
    result = result.reset_index()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved ESIF features: {OUTPUT_FILE} ({len(result)} rows)")
    return result
