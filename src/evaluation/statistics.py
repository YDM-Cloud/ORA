import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import friedmanchisquare, wilcoxon


# =====================================================
# Load
# =====================================================

def load_results(input_file):
    df = pd.read_csv(input_file)
    print("Loaded:", df.shape)
    return df


# =====================================================
# Prepare matrix
# =====================================================

def prepare_matrix(df, mode="standard"):
    if mode == "standard":
        index = [
            "scenario",
            "day",
            "run"
        ]
    elif mode == "scalability":
        index = [
            "scale",
            "scenario",
            "run"
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    matrix = df.pivot_table(index=index, columns="algorithm", values="fitness", aggfunc="mean")
    return matrix.dropna()


# =====================================================
# Friedman
# =====================================================

def friedman_test(matrix, output_dir):
    values = [matrix[col].values for col in matrix.columns]
    stat, p = friedmanchisquare(*values)
    result = pd.DataFrame({
        "test": ["Friedman"],
        "statistic": [stat],
        "p_value": [p]
    })
    result.to_csv(output_dir / "friedman_test.csv", index=False)
    return result


# =====================================================
# Wilcoxon
# =====================================================

def wilcoxon_test(matrix, output_dir):
    if "ORA" not in matrix.columns:
        raise ValueError("ORA not found")
    results = []
    ora = matrix["ORA"]

    for alg in matrix.columns:
        if alg == "ORA":
            continue
        stat, p = wilcoxon(ora, matrix[alg], alternative="less")
        results.append({
            "comparison": f"ORA vs {alg}",
            "statistic": stat,
            "p_value": p,
            "significant": p < 0.05
        })

    result = pd.DataFrame(results)
    result.to_csv(output_dir / "wilcoxon_test.csv", index=False)
    return result


# =====================================================
# Cliff delta
# =====================================================

def cliffs_delta(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    more = 0
    less = 0

    for xi in x:
        more += np.sum(xi > y)
        less += np.sum(xi < y)
    return (more - less) / (len(x) * len(y))


def effect_size(matrix, output_dir):
    ora = matrix["ORA"]
    results = []

    for alg in matrix.columns:
        if alg == "ORA":
            continue

        delta = cliffs_delta(matrix[alg], ora)
        if abs(delta) < 0.147:
            level = "negligible"
        elif abs(delta) < 0.33:
            level = "small"
        elif abs(delta) < 0.474:
            level = "medium"
        else:
            level = "large"

        results.append({
            "comparison": f"ORA vs {alg}",
            "cliffs_delta": delta,
            "effect_level": level
        })

    result = pd.DataFrame(results)
    result.to_csv(output_dir / "effect_size.csv", index=False)
    return result


# =====================================================
# Win rate
# =====================================================

def win_rate(matrix, output_dir):
    ora = matrix["ORA"]
    results = []

    for alg in matrix.columns:
        if alg == "ORA":
            continue

        baseline = matrix[alg]
        win = np.sum(ora < baseline)
        loss = np.sum(ora > baseline)
        tie = np.sum(ora == baseline)
        total = len(ora)
        results.append({
            "comparison": f"ORA vs {alg}",
            "samples": total,
            "ORA_win": win,
            "ORA_loss": loss,
            "tie": tie,
            "win_rate": win / total,
            "win_rate_percent": win / total * 100
        })

    result = pd.DataFrame(results)
    result.to_csv(output_dir / "win_rate.csv", index=False)
    return result


# =====================================================
# API
# =====================================================

def run_statistics(input_file, output_dir, mode="standard"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(input_file)
    matrix = prepare_matrix(df, mode)
    print("\nMatrix:")
    print(matrix.head())
    results = {
        "friedman": friedman_test(matrix, output_dir),
        "wilcoxon": wilcoxon_test(matrix, output_dir),
        "effect_size": effect_size(matrix, output_dir),
        "win_rate": win_rate(matrix, output_dir)
    }
    return results
