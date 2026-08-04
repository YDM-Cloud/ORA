import numpy as np
import pandas as pd
from src.config import load_config

_CONFIG = load_config("default")
_REQUIRED_COLUMNS = {
    "timestamp",
    "base_it_power_kw",
    "ai_power_kw",
    "request_rate",
    "pue",
    "electricity_price",
    "carbon_intensity",
    "sla_min_workload_ratio"
}


class EnergySchedulingObjective:
    def __init__(self, day_data, weights=None, baseline_reference=None):
        if not isinstance(day_data, pd.DataFrame):
            raise TypeError("day_data must be a pandas DataFrame")

        missing_columns = sorted(_REQUIRED_COLUMNS - set(day_data.columns))
        if missing_columns:
            raise ValueError(f"Missing objective columns: {', '.join(missing_columns)}")
        if day_data.empty:
            raise ValueError("day_data must not be empty")

        self.data = day_data.sort_values("timestamp").reset_index(drop=True)
        self.time_steps = len(self.data)
        self.dimension = self.time_steps
        self.weights = _CONFIG["objective"]["weights"] if weights is None else weights
        self.hard_penalty = _CONFIG["sla"]["hard_penalty"]
        self.baseline_reference = baseline_reference
        self.base_power = self.data["base_it_power_kw"].to_numpy(dtype=float)
        self.ai_power = self.data["ai_power_kw"].to_numpy(dtype=float)
        self.request_rate = self.data["request_rate"].to_numpy(dtype=float)
        self.pue = self.data["pue"].to_numpy(dtype=float)
        self.electricity_price = self.data["electricity_price"].to_numpy(dtype=float)
        self.carbon = self.data["carbon_intensity"].to_numpy(dtype=float)
        self.sla_min = self.data["sla_min_workload_ratio"].to_numpy(dtype=float)
        self.peak_limit = np.percentile((self.base_power + self.ai_power) * self.pue, 95)

        if self.baseline_reference is None:
            baseline = self.evaluate_solution(np.ones(self.dimension))
            self.baseline_energy = max(baseline["energy_kWh"], 1e-12)
            self.baseline_carbon = max(baseline["carbon_emission"], 1e-12)
            self.baseline_cost = max(baseline["electricity_cost"], 1e-12)
        else:
            self.baseline_energy = max(self.baseline_reference["energy"], 1e-12)
            self.baseline_carbon = max(self.baseline_reference["carbon"], 1e-12)
            self.baseline_cost = max(self.baseline_reference["cost"], 1e-12)

    def __call__(self, x):
        metrics = self.evaluate_solution(x)
        fitness = (
                self.weights["energy"] * metrics["energy_kWh"] / self.baseline_energy
                + self.weights["carbon"]
                * metrics["carbon_emission"]
                / self.baseline_carbon
                + self.weights["cost"]
                * metrics["electricity_cost"]
                / self.baseline_cost
                + self.weights["sla"] * metrics["sla_violation"] / self.time_steps
                + self.weights["peak"] * metrics["peak_penalty"] / self.time_steps
                + self.weights["pue"] * metrics["pue_penalty"] / self.time_steps
        )
        if metrics["sla_violation"] > 0:
            fitness += (self.hard_penalty * metrics["sla_violation"] / self.time_steps)
        return float(fitness)

    def evaluate_solution(self, x):
        schedule = np.asarray(x, dtype=float)
        if schedule.size != self.dimension:
            raise ValueError(f"Decision dimension must be {self.dimension}")
        schedule = np.clip(schedule, 0, 1)
        facility_power = (self.base_power + schedule * self.ai_power) * self.pue
        request_scale = self.request_rate / max(self.request_rate.mean(), 1e-12)
        sla_violation = np.sum(np.maximum(self.sla_min - schedule, 0) * request_scale)
        return {
            "energy_kWh": np.sum(facility_power),
            "electricity_cost": np.sum(facility_power * self.electricity_price),
            "carbon_emission": np.sum(facility_power * self.carbon),
            "peak_power": np.max(facility_power),
            "peak_penalty": np.sum(np.maximum(facility_power - self.peak_limit, 0)),
            "sla_violation": sla_violation,
            "served_requests": np.sum(self.request_rate * schedule),
            "unserved_requests": np.sum(self.request_rate * (1 - schedule)),
            "pue_penalty": np.sum(np.maximum(self.pue - 1.1, 0)),
            "average_pue": np.mean(self.pue)
        }
