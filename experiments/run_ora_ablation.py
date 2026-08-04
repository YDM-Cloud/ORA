import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_energy_case import run_experiment
from src.config import load_config, project_path
from src.evaluation.statistics import run_statistics


# =====================================================
# Summary
# =====================================================


def build_ablation_summary(results):
    return (
        results.groupby(["scenario", "algorithm"])
        .agg(
            Mean_Objective=("fitness", "mean"),
            Std_Objective=("fitness", "std"),
            Energy_Cost=("electricity_cost", "mean"),
            Carbon_Emission=("carbon_emission", "mean"),
            SLA_Violation=("sla_violation", "mean")
        )
        .reset_index()
        .rename(columns={"scenario": "Scenario", "algorithm": "Algorithm"})
    )


# =====================================================
# Main
# =====================================================


def main(experiment=None, result_dir=None):
    config = load_config("energy_case")
    experiment = config["ablation"] if experiment is None else experiment
    experiment = {
        **experiment,
        "day_selection_seed": experiment.get("day_selection_seed", config["optimization"]["day_selection_seed"])
    }
    result_dir = project_path(config["paths"]["ablation_result_dir"]) if result_dir is None else Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # ============================
    # Run ablation experiment
    # ============================

    run_experiment(experiment=experiment, result_dir=result_dir)
    results_file = result_dir / "optimization_results.csv"
    results = pd.read_csv(results_file)
    summary = build_ablation_summary(results)
    summary.to_csv(result_dir / "ablation_results.csv", index=False)
    print("\nAblation summary:")
    print(summary.to_string(index=False))

    # ============================
    # Statistical analysis
    # ============================

    statistics_dir = result_dir / "statistics"
    run_statistics(results_file, statistics_dir)
    print("\nStatistics saved:")
    print(statistics_dir)


if __name__ == "__main__":
    main()
