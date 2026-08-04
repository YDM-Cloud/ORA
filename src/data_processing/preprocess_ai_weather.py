import numpy as np
import pandas as pd
from src.config import load_config, project_path

_CONFIG = load_config("energy_case")
INPUT_FILE = project_path("data/raw/nlr_outside/esif.influx.buildingData.outside.combined.parquet")
OUTPUT_FILE = project_path(_CONFIG["paths"]["weather_features"])


def classify_temperature(temperature):
    return np.select([temperature < 25, temperature <= 35], ["normal", "hot"], default="extreme")


def preprocess_ai_weather():
    weather = pd.read_parquet(INPUT_FILE, columns=["ts", "outdoor_air_humidity", "outdoor_air_temp"])
    weather = weather.rename(columns={
        "ts": "timestamp",
        "outdoor_air_humidity": "humidity",
        "outdoor_air_temp": "temperature_f"
    })
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], errors="coerce")
    weather["temperature_f"] = pd.to_numeric(weather["temperature_f"], errors="coerce")
    weather["humidity"] = pd.to_numeric(weather["humidity"], errors="coerce")
    weather = weather[
        weather["timestamp"].ge("2010-01-01")
        & weather["temperature_f"].between(-40, 140)
        & weather["humidity"].between(0, 100)
        ].copy()
    weather["temperature"] = (weather["temperature_f"] - 32) * 5 / 9
    weather = (
        weather.set_index("timestamp")[["temperature", "humidity"]]
        .resample("1h")
        .mean()
        .interpolate(method="time", limit=6)
        .dropna()
        .reset_index()
    )
    weather["cooling_degree"] = np.maximum(weather["temperature"] - 18, 0) * \
                                (1 + 0.01 * (weather["humidity"] - 50)).clip(lower=0.5)
    weather["temperature_level"] = classify_temperature(weather["temperature"])
    weather["weather_source"] = INPUT_FILE.name

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved weather features: {OUTPUT_FILE} ({len(weather)} rows)")
    return weather
