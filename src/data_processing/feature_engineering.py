import numpy as np
import pandas as pd
from src.config import load_config, project_path

_CONFIG = load_config("energy_case")
_PATHS = _CONFIG["paths"]
ESIF_FILE = project_path(_PATHS["esif_15min"])
WEATHER_FILE = project_path(_PATHS["weather_features"])
CARBON_POOL_FILE = project_path(_PATHS["carbon_pool"])
PRICE_FILE = project_path(_PATHS["electricity_price_profile"])
OUTPUT_FILE = project_path(_PATHS["feature_dataset"])


def repeat_profile(profile, length):
    if profile.empty:
        raise ValueError("Cannot repeat an empty profile")
    return pd.DataFrame({
        column: np.resize(profile[column].to_numpy(), length) for column in profile.columns
    })


def select_weather_profile(weather, level, target_length):
    weather = weather.copy()
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    weather["year"] = weather["timestamp"].dt.year
    annual = weather.groupby("year")["temperature"].agg(
        count="count",
        mean="mean",
        extreme=lambda values: values.quantile(0.99)
    )
    annual = annual[annual["count"].ge(8500)]
    if level == "normal":
        year = (annual["mean"] - annual["mean"].median()).abs().idxmin()
    elif level == "hot":
        year = annual["mean"].idxmax()
    elif level == "extreme":
        year = annual["extreme"].idxmax()
    else:
        raise ValueError(f"Unknown weather level: {level}")

    year_start = pd.Timestamp(year=year, month=1, day=1)
    year_end = year_start + pd.DateOffset(years=1)
    full_time = pd.date_range(year_start, year_end - pd.Timedelta(hours=1), freq="1h")
    profile = (
        weather[weather["year"].eq(year)]
        .set_index("timestamp")[["temperature", "humidity", "cooling_degree"]]
        .resample("1h")
        .mean()
        .reindex(full_time)
        .interpolate(method="time", limit_direction="both")
    )
    profile = profile[~((profile.index.month == 2) & (profile.index.day == 29))]
    profile["weather_source"] = weather["weather_source"].iloc[0]
    profile["weather_profile_year"] = int(year)
    profile = profile.loc[profile.index.repeat(4)].reset_index(drop=True)
    return repeat_profile(profile, target_length)


def select_carbon_profile(carbon, level, target_length):
    carbon = carbon.copy()
    carbon["timestamp"] = pd.to_datetime(carbon["timestamp"])
    carbon["date"] = carbon["timestamp"].dt.normalize()
    daily_stats = carbon.groupby("date")["carbon_intensity"].agg(["mean", "count"])
    daily = daily_stats[daily_stats["count"].ge(24)]["mean"]

    if level == "normal":
        selected_dates = daily.index
    elif level == "high":
        selected_dates = daily[daily.ge(daily.quantile(0.75))].index
    else:
        raise ValueError(f"Unknown carbon level: {level}")

    profile = carbon[carbon["date"].isin(selected_dates)].sort_values("timestamp")[
        ["timestamp", "carbon_intensity", "carbon_interpolated"]]
    profile["carbon_profile_date"] = profile["timestamp"].dt.date.astype(str)
    profile = profile.drop(columns="timestamp")
    profile = profile.loc[profile.index.repeat(4)].reset_index(drop=True)
    return repeat_profile(profile, target_length)


def select_price_profile(price, target_length):
    price = price.copy()
    price["timestamp"] = pd.to_datetime(price["timestamp"])
    price = price.sort_values("timestamp")
    if len(price) != target_length // 4:
        raise ValueError("Electricity-price profile does not cover the experiment window")
    intervals = price["timestamp"].diff().dropna()
    if not intervals.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Electricity-price timestamps must be hourly")
    columns = [
        "electricity_price",
        "price_source",
        "price_tariff",
        "tariff_effective_date",
        "price_source_file",
    ]
    return price[columns].loc[price.index.repeat(4)].reset_index(drop=True)


def build_dataset():
    esif = pd.read_csv(ESIF_FILE)
    esif = esif.rename(columns={"ts": "timestamp"})
    esif["timestamp"] = pd.to_datetime(esif["timestamp"])
    esif = esif.sort_values("timestamp").reset_index(drop=True)
    weather = pd.read_csv(WEATHER_FILE)
    carbon = pd.read_csv(CARBON_POOL_FILE)
    price = pd.read_csv(PRICE_FILE)
    normal_weather = select_weather_profile(weather, "normal", len(esif))
    normal_carbon = select_carbon_profile(carbon, "normal", len(esif))
    real_price = select_price_profile(price, len(esif))
    result = pd.DataFrame({
        "timestamp": esif["timestamp"],
        "scenario_id": 0,
        "scenario": "Base_Features",
        "scenario_name": "Base_Features",
        "base_it_power_kw": esif["it_power_kw"],
        "base_pue": esif["pue"],
        "esif_interpolated": esif["esif_interpolated"].astype(int),
    })
    result = pd.concat([
        result,
        normal_weather[[
            "temperature",
            "humidity",
            "cooling_degree",
            "weather_source",
            "weather_profile_year",
        ]],
        normal_carbon[[
            "carbon_intensity",
            "carbon_interpolated",
            "carbon_profile_date",
        ]],
        real_price[[
            "electricity_price",
            "price_source",
            "price_tariff",
            "tariff_effective_date",
            "price_source_file",
        ]],
    ], axis=1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved feature dataset: {OUTPUT_FILE} ({len(result)} rows)")
    return result
