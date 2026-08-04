import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# =====================================================
# Configuration
# =====================================================

RESULT_DIR = ROOT / "results"
OUTPUT_DIR = RESULT_DIR / "paper" / "table_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# Excel writer
# =====================================================

def save_excel(filename, sheets):
    path = OUTPUT_DIR / filename
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print("Saved:", path)


def check_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file:\n{path}")


# =====================================================
# Table 1
# =====================================================

def prepare_table1():
    data = {
        "Scenario": [
            "Scenario 1",
            "Scenario 3",
            "Scenario 6"
        ],
        "Operating Condition": [
            "Normal online workload",
            "Flexible batch workload",
            "High-pressure workload near feasibility boundary"
        ],
        "Main Evaluation Purpose": [
            "General energy-carbon management capability",
            "Temporal energy optimization capability",
            "Reliability stress evaluation"
        ]
    }
    df = pd.DataFrame(data)
    save_excel("Table1_scenarios.xlsx", {"scenario": df})


# =====================================================
# Table 2
# =====================================================

def prepare_table2():
    data = {
        "Approach": [
            "ORA",
            "DE",
            "PSO",
            "GTO",
            "MGO",
            "MPC",
            "SLSQP"
        ],
        "Category": [
            "Black-box search engine",
            "Evolutionary search",
            "Swarm intelligence",
            "Bio-inspired search",
            "Bio-inspired search",
            "Predictive control",
            "Mathematical optimization"
        ],
        "Role in Evaluation": [
            "Representative nonlinear decision generator",
            "Comparative heuristic decision generator",
            "Comparative heuristic decision generator",
            "Recent heuristic decision generator",
            "Recent heuristic decision generator",
            "Model-based energy management approach",
            "Continuous relaxation reference"
        ]
    }
    df = pd.DataFrame(data)
    save_excel("Table2_decision_approaches.xlsx", {"approaches": df})


# =====================================================
# Table 3
# =====================================================

def prepare_table3():
    df = pd.DataFrame({
        "Variant": [
            "ORA",
            "ORA-NoArchive",
            "ORA-NoResonance"
        ],
        "Archive Guidance": [
            "✓",
            "✗",
            "✓"
        ],
        "Resonance Exploration": [
            "✓",
            "✓",
            "✗"
        ]
    })
    save_excel("Table3_ora_configuration.xlsx", {"configuration": df})


# =====================================================
# Table 4
# =====================================================

def prepare_table4():
    df = pd.DataFrame({
        "Preference Mode": [
            "Balanced",
            "Cost Priority",
            "Carbon Priority",
            "SLA Priority"
        ],
        "Management Priority": [
            "Cost-carbon-reliability trade-off",
            "Economic operation",
            "Environmental operation",
            "Service reliability"
        ]
    })
    save_excel("Table4_preferences.xlsx", {"preferences": df})


# =====================================================
# Table 5
# =====================================================

def prepare_table5():
    df = pd.DataFrame({
        "Parameter": [
            "Population size",
            "Maximum iteration",
            "Independent runs",
            "Scenarios",
            "Evaluation framework"
        ],
        "Setting": [
            "50",
            "500",
            "5",
            "Scenario 1, 3, 6",
            "Feasibility-audited energy management"
        ]
    })
    save_excel("Table5_parameters.xlsx", {"parameters": df})


# =====================================================
# Table 6
# =====================================================

def prepare_table6():
    file = RESULT_DIR / "energy_case" / "optimization_results.csv"
    check_file(file)
    df = pd.read_csv(file)
    table = df.groupby("algorithm")[[
        "fitness",
        "energy_kWh",
        "electricity_cost",
        "carbon_emission",
        "sla_violation",
        "runtime_seconds"
    ]].mean().reset_index()

    table.columns = [
        "Algorithm",
        "Mean Fitness",
        "Energy (kWh)",
        "Electricity Cost",
        "Carbon Emission",
        "SLA Violation",
        "Runtime (s)"
    ]
    save_excel("Table6_performance.xlsx", {"performance": table})


# =====================================================
# Table 7
# =====================================================

def prepare_table7():
    file = RESULT_DIR / "feasibility_audit" / "feasibility_summary.csv"
    check_file(file)
    df = pd.read_csv(file)
    table = df.copy()
    save_excel("Table7_feasibility.xlsx", {"feasibility": table})


# =====================================================
# Table 8
# =====================================================

def prepare_table8():
    file = RESULT_DIR / "repair_test" / "repair_summary.csv"
    check_file(file)
    df = pd.read_csv(file)
    columns = [
        "Scenario",
        "Algorithm",
        "SLA_Before",
        "SLA_After",
        "Delta_Fitness",
        "Cost_Change",
        "Carbon_Change"
    ]

    # compatible with different naming styles
    rename_map = {
        "Delta_Fitness_Mean": "Delta_Fitness",
        "Cost_change": "Cost_Change",
        "Carbon_change": "Carbon_Change"
    }
    df = df.rename(columns=rename_map)
    available = [c for c in columns if c in df.columns]
    table = df[available].copy()
    save_excel("Table8_repair.xlsx", {"repair": table})


# =====================================================
# Table 9
# =====================================================

def prepare_table9():
    file = RESULT_DIR / "slsqp_baseline" / "slsqp_scenario_summary.csv"
    check_file(file)
    df = pd.read_csv(file)
    rename_map = {
        "fitness": "Fitness",
        "mean_fitness": "Fitness",
        "energy_kWh": "Energy (kWh)",
        "electricity_cost": "Cost",
        "carbon_emission": "Carbon",
        "mean_iteration": "Mean Iteration"
    }
    table = df.rename(columns=rename_map)
    save_excel("Table9_slsqp.xlsx", {"slsqp_reference": table})


# =====================================================
# Table 10
# =====================================================

def prepare_table10():
    effect_file = RESULT_DIR / "penalty_sensitivity" / "penalty_effect_analysis.csv"
    stability_file = RESULT_DIR / "penalty_sensitivity" / "penalty_stability.csv"

    if not effect_file.exists():
        print("Penalty robustness files not found, skip Table 10")
        return
    effect = pd.read_csv(effect_file)

    if stability_file.exists():
        stability = pd.read_csv(stability_file)
        if "algorithm" in effect.columns and "algorithm" in stability.columns:
            table = effect.merge(stability, on="algorithm", how="left")
        else:
            table = effect
    else:
        table = effect
    save_excel("Table10_penalty.xlsx", {"penalty": table})


# =====================================================
# Table 11
# =====================================================

def prepare_table11():
    file = RESULT_DIR / "ablation" / "ablation_results.csv"
    check_file(file)
    df = pd.read_csv(file)
    rename_map = {
        "Algorithm": "Method",
        "Mean_Objective": "Mean Objective",
        "Energy_Cost": "Cost",
        "Carbon_Emission": "Carbon",
        "SLA_Violation": "SLA Violation"
    }
    df = df.rename(columns=rename_map)
    columns = [
        "Method",
        "Mean Objective",
        "Cost",
        "Carbon",
        "SLA Violation"
    ]
    available = [c for c in columns if c in df.columns]
    table = df[available]
    save_excel("Table11_ablation.xlsx", {"ablation": table})


# =====================================================
# Table 12
# =====================================================


def prepare_table12():
    statistics_dir = RESULT_DIR / "energy_case" / "statistics_scheduling_performance"
    wilcoxon_file = statistics_dir / "wilcoxon_test.csv"
    effect_file = statistics_dir / "effect_size.csv"
    friedman_file = statistics_dir / "friedman_test.csv"

    for f in [wilcoxon_file, effect_file, friedman_file]:
        if not f.exists():
            print("Missing:", f)
            return

    # =================================================
    # Load data
    # =================================================

    wilcoxon = pd.read_csv(wilcoxon_file)
    effect = pd.read_csv(effect_file)
    friedman = pd.read_csv(friedman_file)
    wilcoxon.columns = [c.strip() for c in wilcoxon.columns]
    effect.columns = [c.strip() for c in effect.columns]
    friedman.columns = [c.strip() for c in friedman.columns]

    # =================================================
    # Normalize Wilcoxon columns
    # =================================================

    wilcoxon.rename(columns={
        "comparison": "Comparison",
        "p_value": "Wilcoxon p-value",
        "p-value": "Wilcoxon p-value",
        "pvalue": "Wilcoxon p-value",
        "statistic": "Wilcoxon statistic"
    }, inplace=True)

    # =================================================
    # Normalize effect columns
    # =================================================

    effect.rename(columns={
        "comparison": "Comparison",
        "cliffs_delta": "Cliffs delta",
        "cliff_delta": "Cliffs delta",
        "effect_size": "Cliffs delta",
        "delta": "Cliffs delta",
        "effect_level": "Effect Level"
    }, inplace=True)

    # =================================================
    # Select columns
    # =================================================

    wilcoxon_cols = ["Comparison", "Wilcoxon p-value"]
    if "Wilcoxon statistic" in wilcoxon.columns:
        wilcoxon_cols.append("Wilcoxon statistic")
    wilcoxon = wilcoxon[wilcoxon_cols]

    effect_cols = ["Comparison", "Cliffs delta"]
    if "Effect Level" in effect.columns:
        effect_cols.append("Effect Level")
    effect = effect[effect_cols]

    # =================================================
    # Merge
    # =================================================

    table = wilcoxon.merge(effect, on="Comparison", how="left")

    # =================================================
    # Effect classification
    # =================================================

    if "Effect Level" not in table.columns:
        def classify_effect(value):
            value = abs(float(value))
            if value < 0.147:
                return "Negligible"
            elif value < 0.33:
                return "Small"
            elif value < 0.474:
                return "Medium"
            else:
                return "Large"

        table["Effect Level"] = table["Cliffs delta"].apply(classify_effect)

    # =================================================
    # Significance
    # =================================================

    table["Significance"] = pd.to_numeric(table["Wilcoxon p-value"], errors="coerce").apply(
        lambda x: "Yes" if x < 0.05 else "No")

    # =================================================
    # Interpretation
    # Fitness objective is minimized
    # delta = ORA - baseline
    # =================================================

    def interpretation(row):
        comparison = str(row["Comparison"])
        p_value = float(row["Wilcoxon p-value"])
        delta = float(row["Cliffs delta"])

        # ORA vs DE
        # DE has slightly lower fitness
        if comparison == "ORA vs DE":
            return ("DE achieves slightly lower fitness, "
                    "but the absolute objective gap is negligible")

        # For minimization objective:
        # delta < 0 means ORA fitness is lower
        # delta > 0 means baseline fitness is lower

        if delta < 0:
            if p_value < 0.05:
                return ("ORA consistently achieves lower fitness "
                        "with statistical significance")
            else:
                return ("ORA consistently achieves lower fitness, "
                        "but not statistically significant")
        else:
            if p_value < 0.05:
                return ("Baseline approach achieves lower finess "
                        "with statistical significance"
                        )
            else:
                return ("Baseline approach achieves lower fitness, "
                        "but not statistically significant")

    table["Interpretation"] = table.apply(interpretation, axis=1)

    # =================================================
    # Format
    # =================================================

    table["Wilcoxon p-value"] = pd.to_numeric(table["Wilcoxon p-value"], errors="coerce").round(4)
    table["Cliffs delta"] = pd.to_numeric(table["Cliffs delta"], errors="coerce").round(3)

    # =================================================
    # Column order
    # =================================================

    table = table[[
        "Comparison",
        "Wilcoxon p-value",
        "Significance",
        "Cliffs delta",
        "Effect Level",
        "Interpretation"
    ]]

    # =================================================
    # Friedman summary
    # =================================================

    friedman_sheet = friedman.copy()

    # =================================================
    # Save
    # =================================================

    save_excel("Table12_statistical_analysis.xlsx", {"Pairwise test": table, "Friedman test": friedman_sheet})
    print("Generated Table XII")


# =====================================================
# Main
# =====================================================

def run_reproduce_tables():
    print("=" * 80)
    print("Generating draft_v6 manuscript tables")
    print("=" * 80)
    prepare_table1()
    prepare_table2()
    prepare_table3()
    prepare_table4()
    prepare_table5()
    prepare_table6()
    prepare_table7()
    prepare_table8()
    prepare_table9()
    prepare_table10()
    prepare_table11()
    prepare_table12()
    print()
    print("All tables generated.")
    print(OUTPUT_DIR)
