import numpy as np
import pandas as pd
from src.config import load_config, project_path

_PATHS = load_config("energy_case")["paths"]
RAW_FILE = project_path(_PATHS["electricity_price_raw"])
OUTPUT_FILE = project_path(_PATHS["electricity_price_profile"])

ON_PEAK_PRICE = 0.18824
OFF_PEAK_PRICE = 0.06527
ON_PEAK_START_HOUR = 16
ON_PEAK_END_HOUR = 21
TARIFF = "Secondary Voltaic Time-of-Use Service Section B (SPVTOU-B)"
EFFECTIVE_DATE = "2023-01-01"


def build_price_profile(year=2023):
    timestamps = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:00:00", freq="1h")
    on_peak = timestamps.hour.to_series(index=timestamps).between(ON_PEAK_START_HOUR, ON_PEAK_END_HOUR - 1)
    prices = np.where(on_peak, ON_PEAK_PRICE, OFF_PEAK_PRICE)

    # ponytail: energy charges only; add monthly demand/fixed charges for
    # bill-grade accounting.
    return pd.DataFrame({
        "timestamp": timestamps,
        "electricity_price": prices,
        "price_source": "Public Service Company of Colorado",
        "price_tariff": TARIFF,
        "tariff_effective_date": EFFECTIVE_DATE,
        "price_source_file": RAW_FILE.name
    })


def preprocess_electricity():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Missing PSCo rate sheet: {RAW_FILE}")
    result = build_price_profile()
    if len(result) != 8760:
        raise ValueError("PSCo price profile must contain 8760 hourly rows")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)
    print(
        f"Saved PSCo prices: {OUTPUT_FILE} "
        f"({result['electricity_price'].min():.5f} to "
        f"{result['electricity_price'].max():.5f} USD/kWh)"
    )
    return result
