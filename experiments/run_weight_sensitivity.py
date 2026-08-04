import math
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_energy_case import run_experiment
from src.config import load_config, project_path


# =====================================================
# Weight validation
# =====================================================

def validate_weight_sets(weight_sets):
    expected = {
        "energy",
        "carbon",
        "cost",
        "sla",
        "peak",
        "pue"
    }

    for name, weights in weight_sets.items():
        if set(weights) != expected:
            raise ValueError(f"{name} weight keys error")
        if any(v < 0 for v in weights.values()):
            raise ValueError(f"{name} contains negative weight")
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12):
            raise ValueError(f"{name} weights must sum to 1")


# =====================================================
# Summary
# =====================================================

def build_summary(df):
    return (
        df.groupby(["weight_set", "scenario"])
        .agg(
            Mean_Objective=("fitness", "mean"),
            Std_Objective=("fitness", "std"),
            Energy_Cost=("electricity_cost", "mean"),
            Carbon_Emission=("carbon_emission", "mean"),
            SLA_Violation=("sla_violation", "mean"),
            Runtime=("runtime_seconds", "mean")
        )
        .reset_index()
        .rename(columns={"weight_set": "Weight_Set", "scenario": "Scenario"})
    )


# =====================================================
# Post analysis
# =====================================================

def build_weight_analysis(summary):
    base = summary[summary["Weight_Set"] == "balanced"]
    base = base.mean(numeric_only=True)
    metrics = [
        "Mean_Objective",
        "Energy_Cost",
        "Carbon_Emission",
        "SLA_Violation",
        "Runtime"
    ]
    results = []
    grouped = summary.groupby("Weight_Set").mean(numeric_only=True).reset_index()

    for _, row in grouped.iterrows():
        item = {"Weight_Set": row["Weight_Set"]}
        for metric in metrics:
            base_value = base[metric]
            value = row[metric]
            if abs(base_value) < 1e-12:
                change = value - base_value
                item[metric + "_Change"] = change
            else:
                change = (value - base_value) / base_value * 100
                item[metric + "_Change_%"] = change
        results.append(item)

    return pd.DataFrame(results)


def build_rank(summary):
    result = summary.groupby("Weight_Set").mean(numeric_only=True).reset_index()
    result["Cost_Rank"] = result["Energy_Cost"].rank()
    result["Carbon_Rank"] = result["Carbon_Emission"].rank()
    result["Objective_Rank"] = result["Mean_Objective"].rank()
    return result


def build_tradeoff(summary):
    return (
        summary
        .groupby("Weight_Set")
        .mean(numeric_only=True)
        .reset_index()[[
            "Weight_Set",
            "Energy_Cost",
            "Carbon_Emission",
            "SLA_Violation",
            "Mean_Objective"
        ]]
    )


# =====================================================
# Main
# =====================================================

def main(experiment=None, result_dir=None):
    config = load_config("energy_case")
    experiment = config["weight_sensitivity"] if experiment is None else experiment
    result_dir = project_path(config["paths"]["weight_sensitivity_result_dir"]) \
        if result_dir is None else Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    validate_weight_sets(experiment["weight_sets"])
    all_results = []
    weight_items = list(experiment["weight_sets"].items())
    total = len(weight_items)
    print("=" * 80)
    print("ORA Weight Sensitivity Experiment")
    print("=" * 80)
    for idx, (name, weights) in enumerate(weight_items, 1):
        print("\n")
        print("-" * 80)
        print(f"[{idx}/{total}] Running: {name}")
        print("Weights:", weights)
        print("-" * 80)
        weight_experiment = {**experiment, "algorithms": [experiment["algorithms"]], "objective_weights": weights}
        weight_dir = result_dir / name
        run_experiment(experiment=weight_experiment, result_dir=weight_dir, enable_statistics=False)
        result = pd.read_csv(weight_dir / "optimization_results.csv")
        result.insert(0, "weight_set", name)
        all_results.append(result)
        print(f"Finished {name}, samples={len(result)}")
    print("\nCombining results...")
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(result_dir / "optimization_results.csv", index=False)
    summary = build_summary(combined)
    summary.to_csv(result_dir / "weight_sensitivity_summary.csv", index=False)
    analysis = build_weight_analysis(summary)
    analysis.to_csv(result_dir / "weight_analysis_summary.csv", index=False)
    rank = build_rank(summary)
    rank.to_csv(result_dir / "weight_rank.csv", index=False)
    tradeoff = build_tradeoff(summary)
    tradeoff.to_csv(result_dir / "tradeoff_metrics.csv", index=False)
    print("\nFinal summary:")
    print(summary.to_string(index=False))
    print("\nGenerated files:")
    print("weight_sensitivity_summary.csv")
    print("weight_analysis_summary.csv")
    print("weight_rank.csv")
    print("tradeoff_metrics.csv")


if __name__ == "__main__":
    main()
