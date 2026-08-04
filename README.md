#  energy scheduling

Reproducible code for carbon-aware AI data-center energy scheduling.

## Layout

- `configs/`: experiment settings and paths
- `data/`: immutable inputs and generated processed data
- `src/`: algorithms, benchmark code, data processing, evaluation
- `experiments/`: executable experiment entry points
- `scripts/`: paper figure and table reproduction
- `results/`: raw, processed, and figure outputs

## Setup

```powershell
conda env create -f environment.yml
conda activate ora-energy
```

## Checks

```powershell
python -m unittest discover -s tests
python -m compileall -q src experiments scripts data/raw/llama3
```

## Run

Energy experiments, from the project root:

```powershell
python -m experiments.run_feature_engineering
python -m experiments.generate_baseline_results
python -m experiments.run_energy_case
python -m experiments.run_reliability_case
```

Processed datasets are written to `data/processed`. Main and reliability
experiment results are written to `results/energy_case` and
`results/reliability`.

Post-processing:

```powershell
python -m experiments.generate_baseline_results
python -m src.evaluation.metrics
python -m src.evaluation.statistics
python -m src.evaluation.calculate_reduction
python -m src.evaluation.pareto_analysis
python -m scripts.reproduce_figures
python -m scripts.reproduce_tables
```

CEC benchmark:

```powershell
python -m experiments.run_cec2026
```

CEC2026 results are written to `results/cec2026`.
