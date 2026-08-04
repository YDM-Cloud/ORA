# ORA Energy Scheduling

Reproducible research code for the manuscript *Beyond Scalar Optimization: A Feasibility-Audited Framework for Reliability Assessment of Energy-Carbon Scheduling in Intelligent Computing Infrastructures*.

The repository implements the ORA archive-resonance scheduling engine, five comparison approaches (DE, PSO, GTO, MGO, and MPC), an SLSQP continuous-optimization reference, and the independent feasibility-auditing and repair workflow used in the paper.

## Repository layout

- `configs/`: experiment settings and data/result paths.
- `data/raw/`: source traces used by the preprocessing pipeline.
- `data/processed/`: generated profiles and six scheduling scenarios.
- `src/algorithms/`: ORA and comparison algorithms.
- `src/data_processing/`: preprocessing, feature engineering, and scenario construction.
- `src/optimization/`: normalized energy-carbon scheduling objective.
- `src/evaluation/`: statistics and post-processing utilities.
- `experiments/`: executable experiment entry points.
- `scripts/`: paper figure and table reproduction.
- `results/`: experiment outputs and submission-ready figure/table data.

## Installation

Python 3.10 is recommended.

Using Conda:

```powershell
conda env create -f environment.yml
conda activate ora-energy
```

Using pip:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

The main configuration is `configs/energy_case.yaml`. It defines data paths, the six operating scenarios, objective weights, algorithm lists, formal run counts, population sizes, iteration budgets, and evaluation settings. `configs/default.yaml` contains the base objective and SLA penalty settings.

Some experiments can take a long time to complete. To verify that the code runs correctly, reduce parameters such as `runs`, `max_days`, `population`, and `max_iterations` in a local copy of the configuration. Use the published settings for full reproduction.

## Reproduce the study

Run commands from the repository root.

Prepare processed data and baseline schedules:

```powershell
python -m experiments.generate_baseline_results
```

Run the main comparison and reliability checks:

```powershell
python -m experiments.run_energy_case
python -m experiments.run_feasibility_audit
python -m experiments.run_repair_test
python -m experiments.run_slsqp_baseline
```

Run sensitivity, ablation, scalability, and robustness experiments:

```powershell
python -m experiments.run_penalty_sensitivity
python -m experiments.run_weight_sensitivity
python -m experiments.run_ora_ablation
python -m experiments.run_ora_scalability
python -m experiments.run_ora_multinode_scalability
python -m experiments.run_stress_test
python -m experiments.run_uncertainty_test
```

Regenerate paper figures and tables from the result files:

```powershell
python -m experiments.run_plots
```

Generated outputs are written beneath `results/`; paper-ready assets are collected under `results/paper/`.

## Quick validation

```powershell
python -m compileall -q src experiments scripts
python -c "from src.algorithms.ORA import ORA; from src.optimization.objective import EnergySchedulingObjective; print('ORA imports OK')"
```

## Data and reproducibility

The raw and processed data under `data/` and the generated outputs under `results/` are too large to distribute with this repository. Contact the corresponding author to request these files. Third-party traces remain subject to their original providers' terms; cite the corresponding datasets and sources when reusing them.

## License

The source code is released under the MIT License. See `LICENSE`.
