from pathlib import Path
import numpy as np
import pandas as pd
from src.config import load_config, project_path

# ============================================================
# Paths
# ============================================================

_CONFIG = load_config("energy_case")
_PATHS = _CONFIG["paths"]
_CARBON = _CONFIG["carbon"]
_WINDOW = _CONFIG["data_window"]
YEAR = pd.Timestamp(_WINDOW["start"]).year
YEAR_START = pd.Timestamp(year=YEAR, month=1, day=1)
YEAR_END = YEAR_START + pd.DateOffset(years=1)
JAN_JUN_FILE = project_path(_PATHS["eia_jan_jun"])
JUL_DEC_FILE = project_path(_PATHS["eia_jul_dec"])
POOL_FILE = project_path(_PATHS["carbon_pool"])

# ============================================================
# Configuration
# ============================================================

BALANCING_AUTHORITY = _CARBON["balancing_authority"]
LOCAL_TIMEZONE = _CARBON["local_timezone"]

# EIA 2023 direct CO2 emission rates:
# coal:       2.31 lb CO2/kWh
# natural gas: 0.96 lb CO2/kWh
# petroleum:   2.46 lb CO2/kWh
LB_TO_KG = _CARBON["lb_to_kg"]
EF_KG_PER_KWH = {fuel: factor * LB_TO_KG for fuel, factor in _CARBON["emission_factors_lb_per_kwh"].items()}

# Minimum fraction of reported generation represented by the seven
# explicitly modelled fuel categories.
MIN_KNOWN_GENERATION_SHARE = _CARBON["min_known_generation_share"]

# A production-mix intensity above this value is inconsistent with
# the selected direct fuel emission factors and is treated as invalid.
MAX_VALID_CARBON_INTENSITY = _CARBON["max_valid_carbon_intensity"]

FUEL_COLUMNS = {
    "coal": "Net Generation (MW) from Coal",
    "gas": "Net Generation (MW) from Natural Gas",
    "nuclear": "Net Generation (MW) from Nuclear",
    "oil": "Net Generation (MW) from All Petroleum Products",
    "hydro": ("Net Generation (MW) from Hydropower and Pumped Storage"),
    "solar": "Net Generation (MW) from Solar",
    "wind": "Net Generation (MW) from Wind",
    "other": "Net Generation (MW) from Other Fuel Sources",
    "unknown": "Net Generation (MW) from Unknown Fuel Sources"
}


# ============================================================
# Helper functions
# ============================================================

def read_eia_file(path: Path) -> pd.DataFrame:
    """Read one EIA-930 half-year file."""
    if not path.exists():
        raise FileNotFoundError(f"EIA file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def coalesce_eia_series(df: pd.DataFrame, base_column: str) -> pd.Series:
    """
    Construct one EIA series using this priority:

    1. Adjusted value
    2. Imputed value
    3. Raw reported value

    Negative generation values are clipped to zero because the
    production-mix calculation requires non-negative fuel shares.
    """
    candidates = [f"{base_column} (Adjusted)", f"{base_column} (Imputed)", base_column]
    result = pd.Series(np.nan, index=df.index, dtype=float)
    available = []
    for column in candidates:
        if column in df.columns:
            available.append(column)
            values = pd.to_numeric(df[column], errors="coerce")
            result = result.fillna(values)
    if not available:
        raise KeyError(f"No usable EIA columns were found for '{base_column}'.")
    return result.clip(lower=0.0)


def longest_missing_run(series: pd.Series) -> int:
    """Return the longest consecutive missing-value run."""
    missing = series.isna()
    if not missing.any():
        return 0
    groups = missing.ne(missing.shift()).cumsum()
    runs = missing.groupby(groups).sum()
    return int(runs.max())


# ============================================================
# Main processing
# ============================================================

def preprocess_eia_carbon() -> None:
    print("=" * 68)
    print("Rebuilding PSCO hourly production-based carbon intensity")
    print("=" * 68)
    first_half = read_eia_file(JAN_JUN_FILE)
    second_half = read_eia_file(JUL_DEC_FILE)
    df = pd.concat([first_half, second_half], ignore_index=True)
    print(f"Combined EIA rows: {len(df):,}")
    df = df[df["Balancing Authority"].astype(str).str.strip() == BALANCING_AUTHORITY].copy()
    if df.empty:
        raise ValueError(f"No rows found for balancing authority {BALANCING_AUTHORITY}.")
    print(f"{BALANCING_AUTHORITY} rows: {len(df):,}")

    # --------------------------------------------------------
    # Time conversion
    # --------------------------------------------------------
    # EIA reports UTC time at the END of the hour.
    # Convert to Colorado local time and subtract one hour so
    # timestamps represent the START of each optimization interval.
    interval_end_utc = pd.to_datetime(df["UTC Time at End of Hour"], errors="coerce", utc=True)
    invalid_time_count = int(interval_end_utc.isna().sum())
    if invalid_time_count:
        print(f"Rows removed because of invalid timestamps: {invalid_time_count}")

    df = df.loc[interval_end_utc.notna()].copy()
    interval_end_utc = interval_end_utc.loc[interval_end_utc.notna()]
    local_interval_start = interval_end_utc.dt.tz_convert(LOCAL_TIMEZONE) - pd.Timedelta(hours=1)
    df["timestamp"] = local_interval_start.dt.tz_localize(None)

    # Keep the full year so scenarios can select real low/high-carbon days.
    df = df[(df["timestamp"] >= YEAR_START) & (df["timestamp"] < YEAR_END)].copy()

    # --------------------------------------------------------
    # Fuel generation
    # --------------------------------------------------------
    for fuel_name, base_column in FUEL_COLUMNS.items():
        df[f"{fuel_name}_mw"] = coalesce_eia_series(df, base_column)

    known_fuels = [
        "coal_mw",
        "gas_mw",
        "nuclear_mw",
        "oil_mw",
        "hydro_mw",
        "solar_mw",
        "wind_mw"
    ]
    df["known_generation_mw"] = df[known_fuels].sum(axis=1, min_count=1)
    df["reported_generation_mw"] = df["known_generation_mw"] + \
                                   df["other_mw"].fillna(0.0) + \
                                   df["unknown_mw"].fillna(0.0)
    df["known_generation_share"] = np.where(
        df["reported_generation_mw"] > 0,
        df["known_generation_mw"]
        / df["reported_generation_mw"],
        np.nan
    )

    # --------------------------------------------------------
    # Demand is retained only as an auxiliary diagnostic.
    # It is NOT used as the carbon-intensity denominator.
    # --------------------------------------------------------
    df["Demand (MW)"] = coalesce_eia_series(df, "Demand (MW)")

    # --------------------------------------------------------
    # Production-based direct operational carbon intensity
    # --------------------------------------------------------
    emission_rate_kg_per_h = df["coal_mw"] * 1000.0 * EF_KG_PER_KWH["coal"] + \
                             df["gas_mw"] * 1000.0 * EF_KG_PER_KWH["gas"] + \
                             df["oil_mw"] * 1000.0 * EF_KG_PER_KWH["oil"]
    generation_energy_kwh_per_h = df["known_generation_mw"] * 1000.0
    df["carbon_intensity"] = np.where(
        generation_energy_kwh_per_h > 0,
        emission_rate_kg_per_h / generation_energy_kwh_per_h,
        np.nan
    )

    # --------------------------------------------------------
    # Quality filters
    # --------------------------------------------------------
    invalid_quality = (df["known_generation_mw"] <= 0) \
                      | (df["known_generation_share"] < MIN_KNOWN_GENERATION_SHARE) \
                      | (df["carbon_intensity"] < 0) \
                      | (df["carbon_intensity"] > MAX_VALID_CARBON_INTENSITY)
    print("Rows rejected by generation-quality checks:", int(invalid_quality.sum()))
    df.loc[invalid_quality, [
        "carbon_intensity",
        "known_generation_mw",
        "known_generation_share"
    ]] = np.nan

    # --------------------------------------------------------
    # Resolve duplicate local-clock timestamps caused by DST.
    # Mean aggregation avoids arbitrarily selecting one repeated hour.
    # --------------------------------------------------------
    hourly = df.groupby("timestamp", as_index=False).agg({
        "Demand (MW)": "mean",
        "known_generation_mw": "mean",
        "known_generation_share": "mean",
        "carbon_intensity": "mean"
    }).sort_values("timestamp")

    # --------------------------------------------------------
    # Create a regular hourly sample axis.
    # --------------------------------------------------------
    full_time = pd.DataFrame({
        "timestamp": pd.date_range(start=YEAR_START, end=YEAR_END - pd.Timedelta(hours=1), freq="1h")
    })
    result = full_time.merge(hourly, on="timestamp", how="left")
    result["carbon_interpolated"] = result["carbon_intensity"].isna().astype(int)
    missing_before = int(result["carbon_intensity"].isna().sum())
    longest_gap = longest_missing_run(result["carbon_intensity"])
    print("Missing/invalid hours before interpolation:", missing_before)
    print("Longest missing run:", longest_gap, "hour(s)")

    # Time-based interpolation for continuous hourly signals.
    result = result.set_index("timestamp")
    interpolation_columns = [
        "Demand (MW)",
        "known_generation_mw",
        "known_generation_share",
        "carbon_intensity"
    ]
    result[interpolation_columns] = result[interpolation_columns].interpolate(
        method="time", limit_direction="both")
    result = result.reset_index()
    result.insert(1, "Balancing Authority", BALANCING_AUTHORITY)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------
    expected_hours = int((YEAR_END - YEAR_START) / pd.Timedelta(hours=1))
    if len(result) != expected_hours:
        raise ValueError(f"Expected {expected_hours} hourly rows, found {len(result)}.")
    if result["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps remain in the final carbon series.")

    required_output = [
        "timestamp",
        "Balancing Authority",
        "Demand (MW)",
        "known_generation_mw",
        "known_generation_share",
        "carbon_intensity",
        "carbon_interpolated"
    ]
    missing_final = result[required_output].isna().sum()
    if int(missing_final.sum()) > 0:
        raise ValueError("Missing values remain after interpolation:\n{missing_final}")

    ci = result["carbon_intensity"]
    if ci.min() < 0 or ci.max() > MAX_VALID_CARBON_INTENSITY:
        raise ValueError("Final carbon intensity is outside the accepted physical range.")

    print("\nFinal carbon-intensity diagnostics:")
    print(ci.describe(percentiles=[0.01, 0.50, 0.95, 0.99]))
    print("Interpolated-hour percentage:", f"{100.0 * result['carbon_interpolated'].mean():.3f}%")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    POOL_FILE.parent.mkdir(parents=True, exist_ok=True)
    result[required_output].to_csv(POOL_FILE, index=False)
    print("\nSaved:")
    print(POOL_FILE)
    print("=" * 68)
