import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import os


# =====================================================
# Load raw results
# =====================================================

def load_results():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing file: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    df.columns = [c.strip() for c in df.columns]
    print("Available columns:")
    print(df.columns.tolist())
    return df


# =====================================================
# Prepare independent run-level samples
# =====================================================

def prepare_run_level_data(df):
    required = [
        "algorithm",
        "run",
        "fitness"
    ]

    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")

    # Aggregate internal samples into
    # independent optimization runs
    run_results = df.groupby(["algorithm", "run"], as_index=False)["fitness"].mean()
    print("\nRun-level samples:")
    print(run_results)
    run_results.to_csv(f"{OUTPUT_DIR}/run_level_fitness.csv", index=False)
    return run_results


# =====================================================
# Friedman test
# =====================================================

def friedman_test(run_results):
    pivot = run_results.pivot(index="run", columns="algorithm", values="fitness")
    algorithms = pivot.columns.tolist()
    data = [pivot[a].values for a in algorithms]
    statistic, p_value = friedmanchisquare(*data)
    result = pd.DataFrame({
        "test": ["Friedman"],
        "chi_square": [statistic],
        "p_value": [p_value],
        "algorithms": [",".join(algorithms)],
        "sample_size": [len(pivot)]
    })
    result.to_csv(f"{OUTPUT_DIR}/friedman_test.csv", index=False)
    print("\nFriedman test:")
    print(result)


# =====================================================
# Wilcoxon signed-rank test
# =====================================================

def wilcoxon_test(run_results):
    algorithms = run_results["algorithm"].unique().tolist()
    records = []
    reference = "ORA"
    if reference not in algorithms:
        raise ValueError("ORA not found")

    ora = run_results[run_results.algorithm == reference].sort_values("run")["fitness"].values
    for alg in algorithms:
        if alg == reference:
            continue

        baseline = run_results[run_results.algorithm == alg].sort_values("run")["fitness"].values
        if len(ora) != len(baseline):
            raise ValueError(f"Sample mismatch ORA vs {alg}")

        stat, p = wilcoxon(ora, baseline, alternative="two-sided")
        records.append({
            "Comparison": f"ORA vs {alg}",
            "Wilcoxon statistic": stat,
            "Wilcoxon p-value": p,
            "n": len(ora)
        })
    result = pd.DataFrame(records)
    result.to_csv(f"{OUTPUT_DIR}/wilcoxon_test.csv", index=False)
    print("\nWilcoxon test:")
    print(result)


# =====================================================
# Cliff's delta
# =====================================================

def cliffs_delta(x, y):
    greater = 0
    less = 0
    for a in x:
        for b in y:
            if a > b:
                greater += 1
            elif a < b:
                less += 1
    return (greater - less) / (len(x) * len(y))


def effect_size(run_results):
    algorithms = run_results["algorithm"].unique().tolist()
    reference = "ORA"
    ora = run_results[run_results.algorithm == reference].sort_values("run")["fitness"].values
    records = []

    for alg in algorithms:
        if alg == reference:
            continue
        baseline = run_results[run_results.algorithm == alg].sort_values("run")["fitness"].values
        delta = cliffs_delta(ora, baseline)

        if abs(delta) < 0.147:
            level = "Negligible"
        elif abs(delta) < 0.33:
            level = "Small"
        elif abs(delta) < 0.474:
            level = "Medium"
        else:
            level = "Large"

        records.append({
            "Comparison": f"ORA vs {alg}",
            "Cliffs delta": delta,
            "Effect level": level,
            "n": len(ora)
        })

    result = pd.DataFrame(records)
    result.to_csv(f"{OUTPUT_DIR}/effect_size.csv", index=False)
    print("\nEffect size:")
    print(result)


# =====================================================
# Main entry
# =====================================================

def run_statistics_scheduling_performance(input_file):
    global ENERGY_DIR
    global OUTPUT_DIR
    global INPUT_FILE

    ENERGY_DIR = input_file
    OUTPUT_DIR = f"{ENERGY_DIR}/statistics_scheduling_performance"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    INPUT_FILE = f"{input_file}/optimization_results.csv"
    df = load_results()
    run_results = prepare_run_level_data(df)
    friedman_test(run_results)
    wilcoxon_test(run_results)
    effect_size(run_results)
    print("\nStatistics completed.")
