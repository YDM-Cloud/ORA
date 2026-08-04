import sys
import json
import pandas as pd
import numpy as np
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator, FixedLocator
from pathlib import Path
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# =====================================================
# Configuration
# =====================================================


RESULT_DIR = ROOT / "results"
OUTPUT_DIR = RESULT_DIR / "paper" / "figure_data"
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


# =====================================================
# Fig.6
# =====================================================

def prepare_fig6():
    file = RESULT_DIR / "energy_case" / "convergence_results.csv"
    if not file.exists():
        print("Missing:", file)
        return

    df = pd.read_csv(file)
    curves = {}

    for _, row in df.iterrows():
        alg = row["algorithm"]
        if alg not in ["ORA", "DE", "PSO", "MGO"]:
            continue
        if alg in curves:
            continue
        curves[alg] = np.array(json.loads(row["curve"]))

    full = pd.DataFrame({"iteration": np.arange(1, max(len(v) for v in curves.values()) + 1)})

    for alg, curve in curves.items():
        full[alg] = pd.Series(curve)

    early = full[full["iteration"] <= 40].copy()
    save_excel("Fig6_convergence.xlsx", {"full_convergence": full, "early_stage": early})


def plot_fig6():
    output_tif = OUTPUT_DIR / "Fig6_convergence.tif"
    df_full = pd.read_excel(OUTPUT_DIR / "Fig6_convergence.xlsx", sheet_name="full_convergence")
    df_full.columns = [str(c).strip() for c in df_full.columns]
    required_cols = [
        "iteration",
        "ORA",
        "DE",
        "PSO",
        "MGO"
    ]
    missing = [c for c in required_cols if c not in df_full.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df_zoom = df_full[df_full["iteration"] <= 50].copy()
    algorithms = [
        "ORA",
        "DE",
        "PSO",
        "MGO"
    ]

    # =========================
    # 3. Nature style
    # =========================

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 8.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })
    colors = {
        "ORA": "#1f77b4",
        "DE": "#ff7f0e",
        "PSO": "#2ca02c",
        "MGO": "#d65db1"
    }
    final_vals = {alg: df_full[alg].iloc[-1] for alg in algorithms}

    # =========================
    # 4. plotting function
    # =========================

    def plot_panel(ax, data, title, xlim, ylim, zoom=False):
        x = data["iteration"].values

        for alg in algorithms:
            y = data[alg].values
            ax.plot(
                x,
                y,
                color=colors[alg],
                linewidth=1.8,
                marker="s",
                markersize=3,
                markevery=20 if zoom else 60,
                markeredgewidth=0,
                label=f"{alg} (final: {final_vals[alg]:.3f})"
            )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(title, pad=6, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best objective (normalized)")
        # remove upper/right border
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if zoom:
            ax.xaxis.set_major_locator(MultipleLocator(10))
            ax.xaxis.set_minor_locator(MultipleLocator(5))
            ax.yaxis.set_major_locator(MultipleLocator(0.01))
            ax.yaxis.set_minor_locator(MultipleLocator(0.005))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.xaxis.set_major_locator(MultipleLocator(100))
            ax.xaxis.set_minor_locator(MultipleLocator(50))

        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            framealpha=1,
            edgecolor="#cccccc",
            borderpad=0.5,
            handlelength=2.2
        )

    # =========================
    # 5. create figure
    # =========================

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), dpi=300, constrained_layout=True)

    # ---------- left ----------

    ymin = df_full[algorithms].min().min()
    ymax = df_full[algorithms].max().max()
    ypad = (ymax - ymin) * 0.03
    plot_panel(
        axes[0],
        df_full,
        "(a) Full convergence",
        (0, 500),
        (ymin - ypad, ymax + ypad),
        zoom=False
    )

    # ---------- right ----------

    plot_panel(
        axes[1],
        df_zoom,
        "(b) Early stage (0–50 iterations)",
        (0, 50),
        (0.668, 0.700),
        zoom=True
    )

    # =========================
    # 6. save only TIFF
    # =========================

    fig.savefig(output_tif, dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close()
    print(f"Saved: {output_tif.resolve()}")


# =====================================================
# Fig.7
# Feasibility audit
# =====================================================

def prepare_fig7():
    file = RESULT_DIR / "feasibility_audit" / "feasibility_summary.csv"
    if not file.exists():
        print("Missing:", file)
        return

    df = pd.read_csv(file)
    cols = [
        "algorithm",
        "objective",
        "mean_violation",
        "runtime_seconds",
        "fully_feasible_rate"
    ]
    result = df[cols].copy()
    result.columns = [
        "algorithm",
        "objective",
        "feasibility_violation",
        "runtime",
        "fully_feasible_rate"
    ]
    save_excel("Fig7_feasibility_audit.xlsx", {"tradeoff": result})


def plot_fig7():
    # =========================================================
    # 1. file path
    # =========================================================

    input_file = RESULT_DIR / "paper/figure_data" / "Fig7_feasibility_audit.xlsx"
    output_file = OUTPUT_DIR / "Fig7_feasibility_audit.tif"

    # =========================================================
    # 2. read data
    # =========================================================
    xls = pd.ExcelFile(input_file)
    sheet_name = "tradeoff" if "tradeoff" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(input_file, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = ["algorithm", "objective", "runtime", "fully_feasible_rate"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    algorithms = ["ORA", "DE", "MGO", "PSO", "MPC", "GTO"]
    df = df[df["algorithm"].isin(algorithms)].copy()
    if df.empty:
        raise ValueError("No target algorithms found in the input file.")

    order = {a: i for i, a in enumerate(algorithms)}
    df["order"] = df["algorithm"].map(order)
    df = df.sort_values("order").reset_index(drop=True)

    # =========================================================
    # 3. Nature style (aligned with Fig.6/Fig.7)
    # =========================================================
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

    # =========================================================
    # 4. colors
    # =========================================================
    colors = {
        "ORA": "#4472C4",
        "DE": "#A78BFA",
        "MGO": "#9E77ED",
        "PSO": "#D6B656",
        "MPC": "#8BC9A8",
        "GTO": "#F28E8E"
    }

    # =========================================================
    # 5. bubble size scaling
    # =========================================================
    runtime = df["runtime"].to_numpy(dtype=float)
    rmin, rmax = runtime.min(), runtime.max()
    size_min, size_max = 180, 950

    if np.isclose(rmax, rmin):
        bubble_sizes = np.full_like(runtime, (size_min + size_max) / 2.0, dtype=float)
    else:
        bubble_sizes = size_min + (runtime - rmin) / (rmax - rmin) * (size_max - size_min)

    def bubble_size(v):
        if np.isclose(rmax, rmin):
            return (size_min + size_max) / 2.0
        return size_min + (v - rmin) / (rmax - rmin) * (size_max - size_min)

    # =========================================================
    # 6. figure and axes
    # =========================================================
    fig, ax = plt.subplots(figsize=(6.4, 4.95), dpi=300)

    x_min, x_max = 0.655, 0.825
    y_min, y_max = 0.10, 1.15
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(FixedLocator([0.66, 0.70, 0.74, 0.78, 0.82]))
    ax.xaxis.set_minor_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25, color="#BFBFBF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # =========================================================
    # 7. plot bubbles
    # =========================================================
    draw_order = ["DE", "MGO", "PSO", "MPC", "GTO", "ORA"]
    for alg in draw_order:
        row = df[df["algorithm"] == alg].iloc[0]
        ax.scatter(
            row["objective"],
            row["fully_feasible_rate"],
            s=bubble_size(row["runtime"]),
            color=colors[alg],
            alpha=0.65,
            edgecolor="#404040",
            linewidth=1.2,
            zorder=4 if alg == "ORA" else 3
        )

    # =========================================================
    # 8. annotations for non-ORA algorithms
    # =========================================================
    label_cfg = {
        "DE": (22, 18),  # right-upper
        "MGO": (8, -30),  # lower
        "PSO": (18, -12),
        "MPC": (15, -10),
        "GTO": (0, 18)  # top
    }

    for _, row in df.iterrows():
        alg = row["algorithm"]
        if alg == "ORA":
            continue
        dx, dy = label_cfg[alg]
        ha = "center" if alg == "GTO" else "left"
        va = "bottom" if alg == "GTO" else "center"
        ax.annotate(
            f"{alg}\n({row.objective:.2f}, {row.fully_feasible_rate:.2f})",
            xy=(row.objective, row.fully_feasible_rate),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            ha=ha,
            va=va,
            zorder=5
        )

    # =========================================================
    # 9. runtime legend
    # =========================================================
    legend_values = np.quantile(runtime, [0.0, 0.33, 0.67, 1.0])
    legend_values = np.unique(np.round(legend_values, 2))
    if len(legend_values) < 4:
        legend_values = np.unique(np.round(np.linspace(rmin, rmax, 4), 2))

    handles = [
        ax.scatter([], [], s=bubble_size(v), color="#A6A6A6", alpha=0.65, edgecolor="#555555", linewidth=0.9)
        for v in legend_values
    ]
    legend = ax.legend(
        handles,
        [f"{v:.2f}" for v in legend_values],
        title="Runtime (s)",
        loc="upper right",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        borderpad=1.9,
        labelspacing=1.7,
        handletextpad=1.2,
        borderaxespad=0.8
    )
    legend.get_title().set_fontsize(11)

    # =========================================================
    # 10. axis labels
    # =========================================================
    ax.set_xlabel("Objective score (lower is better)", labelpad=10)
    ax.set_ylabel("Feasibility violation Φ(x) (lower is better)", labelpad=12)

    # =========================================================
    # 11. helper text + blue arrow (moved to first vertical gridline)
    # =========================================================
    first_grid_x = (0.66 - x_min) / (x_max - x_min)
    ax.text(
        0.02, 1.025,
        "High reliability\n(low violation)",
        transform=ax.transAxes,
        color="#1565C0",
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom"
    )
    ax.annotate(
        "",
        xy=(first_grid_x, 0.965),
        xytext=(first_grid_x, 0.80),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#1565C0"),
        annotation_clip=False
    )

    # =========================================================
    # 12. helper text + red arrow (slightly moved upward)
    # =========================================================
    red_arrow_y = -0.085
    red_text_y = -0.19
    ax.annotate(
        "",
        xy=(0.985, red_arrow_y),
        xytext=(0.80, red_arrow_y),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="red", lw=1.4),
        annotation_clip=False
    )
    ax.text(
        0.905, red_text_y,
        "Lower objective\nbetter",
        transform=ax.transAxes,
        color="red",
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="bottom"
    )

    # =========================================================
    # 13. ORA annotation LAST
    # =========================================================
    ora_row = df[df["algorithm"] == "ORA"].iloc[0]
    ax.annotate(
        f"ORA\n({ora_row.objective:.2f}, {ora_row.fully_feasible_rate:.2f})",
        xy=(ora_row.objective, ora_row.fully_feasible_rate),
        xytext=(0.665, 0.82),
        textcoords="data",
        fontsize=9,
        ha="left",
        va="center",
        arrowprops=dict(
            arrowstyle="-|>",
            lw=0.8,
            color="black",
            shrinkA=2,
            shrinkB=4,
            mutation_scale=10
        ), zorder=10)

    # =========================================================
    # 14. layout and save
    # =========================================================
    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.28)
    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.close()
    print(f"Saved: {output_file.resolve()}")


# =====================================================
# Fig.8
# =====================================================

def prepare_fig8():
    file = RESULT_DIR / "repair_test" / "repair_summary.csv"
    if not file.exists():
        print("Missing:", file)
        return

    df = pd.read_csv(file)
    sla = df[[
        "Algorithm",
        "Scenario",
        "SLA_Before",
        "SLA_After"
    ]].copy()
    sla.columns = [
        "algorithm",
        "scenario",
        "before",
        "after"
    ]
    cost = df[[
        "Algorithm",
        "Scenario",
        "Cost_Change"
    ]].copy()
    cost.columns = [
        "algorithm",
        "scenario",
        "cost_change"
    ]
    carbon = df[[
        "Algorithm",
        "Scenario",
        "Carbon_Change"
    ]].copy()
    carbon.columns = [
        "algorithm",
        "scenario",
        "carbon_change"
    ]
    save_excel("Fig8_repair_verification.xlsx", {
        "SLA": sla,
        "Cost": cost,
        "Carbon": carbon
    })


def plot_fig8():
    # =========================================================
    # 1. File path
    # =========================================================

    input_file = RESULT_DIR / "paper/figure_data" / "Fig8_repair_verification.xlsx"
    output_file = OUTPUT_DIR / "Fig8_repair_verification.tif"

    # =========================================================
    # 2. Nature-like style
    # =========================================================
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

    # =========================================================
    # 3. Read data
    # =========================================================
    scenario = "Scenario_6_Stress_Test"
    sla = pd.read_excel(input_file, sheet_name="SLA")
    cost = pd.read_excel(input_file, sheet_name="Cost")
    carbon = pd.read_excel(input_file, sheet_name="Carbon")
    sla = sla[sla["scenario"] == scenario].copy()
    cost = cost[cost["scenario"] == scenario].copy()
    carbon = carbon[carbon["scenario"] == scenario].copy()
    algorithms = ["ORA", "DE", "PSO"]
    order_map = {"ORA": 0, "DE": 1, "PSO": 2}
    sla["order"] = sla["algorithm"].map(order_map)
    cost["order"] = cost["algorithm"].map(order_map)
    carbon["order"] = carbon["algorithm"].map(order_map)
    sla = sla.sort_values("order").reset_index(drop=True)
    cost = cost.sort_values("order").reset_index(drop=True)
    carbon = carbon.sort_values("order").reset_index(drop=True)

    # =========================================================
    # 4. Colors
    # =========================================================
    blue = "#4472C4"
    orange = "#ED7D31"
    label_blue = "#2F5597"
    label_orange = "#C55A11"
    highlight_red = "#E74C3C"

    # =========================================================
    # 5. Figure
    # =========================================================
    fig, axes = plt.subplots(1, 3, figsize=(9.1, 3.25), dpi=300)

    # =========================================================
    # 6. Helper functions
    # =========================================================
    def style_axis(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3, color="#BFBFBF")
        ax.set_axisbelow(True)

    def add_value_labels(ax, bars, color="#333333", dy_ratio=0.015):
        ymin, ymax = ax.get_ylim()
        dy = (ymax - ymin) * dy_ratio
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + dy,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                fontweight="bold"
            )

    def add_pso_highlight(ax, x_center=2.0, width=0.95):
        ymin, ymax = ax.get_ylim()
        rect = Rectangle(
            (x_center - width / 2, ymin),
            width,
            (ymax - ymin) * 0.995,
            fill=False,
            edgecolor=highlight_red,
            linewidth=1.1,
            linestyle="--",
            alpha=0.85,
            clip_on=False
        )
        ax.add_patch(rect)

    # =========================================================
    # 7. (a) SLA violation
    # =========================================================
    ax = axes[0]
    x = np.arange(len(algorithms))
    width = 0.32
    before = [sla.loc[sla["algorithm"] == a, "before"].iloc[0] for a in algorithms]
    after = [sla.loc[sla["algorithm"] == a, "after"].iloc[0] for a in algorithms]
    bars_before = ax.bar(x - width / 2, before, width, label="Before repair", color=blue, edgecolor=blue)
    bars_after = ax.bar(x + width / 2, after, width, label="After repair", color=orange, edgecolor=orange)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=10, fontweight="bold")
    ax.set_ylabel("SLA violation (%)")
    ax.set_title("(a) SLA violation (%)", fontweight="bold")
    ax.set_ylim(0, max(before) * 1.38 + 0.05)
    ax.set_xlim(-0.5, 2.6)
    style_axis(ax)
    ax.legend(frameon=True, fontsize=8, loc="upper left")
    add_value_labels(ax, bars_before, color=label_blue)
    add_value_labels(ax, bars_after, color=label_orange)
    add_pso_highlight(ax)

    # =========================================================
    # 8. (b) Cost change after repair
    # =========================================================
    ax = axes[1]
    cost_values = [cost.loc[cost["algorithm"] == a, "cost_change"].iloc[0] * 100 for a in algorithms]
    bars = ax.bar(x, cost_values, width=0.45, color=blue, edgecolor=blue)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=10, fontweight="bold")
    ax.set_ylabel("Δ Cost after repair (%)")
    ax.set_title("(b) Cost change after repair (%)", fontweight="bold")
    ax.set_ylim(0, max(cost_values) * 1.42 + 0.15)
    ax.set_xlim(-0.5, 2.6)
    style_axis(ax)
    add_value_labels(ax, bars, color=label_blue)
    add_pso_highlight(ax)

    # =========================================================
    # 9. (c) Carbon change after repair
    # =========================================================
    ax = axes[2]
    carbon_values = [carbon.loc[carbon["algorithm"] == a, "carbon_change"].iloc[0] * 100 for a in algorithms]
    bars = ax.bar(x, carbon_values, width=0.45, color=blue, edgecolor=blue)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=10, fontweight="bold")
    ax.set_ylabel("Δ Carbon after repair (%)")
    ax.set_title("(c) Carbon change after repair (%)", fontweight="bold")
    ax.set_ylim(0, max(carbon_values) * 1.42 + 0.08)
    ax.set_xlim(-0.5, 2.6)
    style_axis(ax)
    add_value_labels(ax, bars, color=label_blue)
    add_pso_highlight(ax)

    # =========================================================
    # 10. Bottom annotation box (moved upward)
    # =========================================================
    fig.text(
        0.5,
        0.105,
        "PSO delivers deceptively low objective but triggers large hidden penalties under stress conditions.",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", edgecolor="#C00000", facecolor="white", linewidth=1.2)
    )

    # =========================================================
    # 11. Layout and save
    # =========================================================
    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.86,
        bottom=0.24,
        wspace=0.38
    )
    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.close()
    print(f"Saved: {output_file.resolve()}")


# =====================================================
# Fig.9
# SLSQP benchmark
# =====================================================

def prepare_fig9():
    file = RESULT_DIR / "slsqp_baseline" / "slsqp_scenario_summary.csv"
    if not file.exists():
        print("Missing:", file)
        return

    df = pd.read_csv(file)
    cols = [
        "scenario",
        "fitness",
        "sla_violation",
        "iterations",
        "energy_kWh",
        "electricity_cost",
        "carbon_emission"
    ]
    result = df[cols].copy()
    result.columns = [
        "scenario",
        "objective",
        "sla_violation",
        "iterations",
        "energy",
        "cost",
        "carbon"
    ]
    save_excel("Fig9_slsqp_reference.xlsx", {"benchmark": result})


def plot_fig9():
    # =========================================================
    # 1. File path
    # =========================================================

    input_file = RESULT_DIR / "paper/figure_data" / "Fig9_slsqp_reference.xlsx"
    output_file = OUTPUT_DIR / "Fig9_slsqp_reference.tif"

    # =========================================================
    # 2. Nature style
    # =========================================================
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

    # =========================================================
    # 3. Read data
    # =========================================================
    xls = pd.ExcelFile(input_file)
    sheet_name = "benchmark" if "benchmark" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(input_file, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]
    required_cols = ["scenario", "objective", "sla_violation"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    scenario_order = {
        "Scenario_1_Online_Normal": 0,
        "Scenario_3_Offline_Batch": 1,
        "Scenario_6_Stress_Test": 2
    }
    df["order"] = df["scenario"].map(scenario_order)
    df = df.sort_values("order").reset_index(drop=True)
    scenario_labels = [
        "Scenario 1\n(Online_Normal)",
        "Scenario 3\n(Offline_Batch)",
        "Scenario 6\n(Stress_Test)"
    ]
    objective = df["objective"].to_numpy(dtype=float)
    sla = df["sla_violation"].to_numpy(dtype=float)

    # =========================================================
    # 4. Figure
    # =========================================================
    fig, ax1 = plt.subplots(figsize=(7.2, 3.6), dpi=300)
    ax2 = ax1.twinx()
    x = np.arange(len(df))
    width = 0.35

    # =========================================================
    # 5. Bars
    # =========================================================
    bar1 = ax1.bar(
        x - width / 2,
        objective,
        width,
        color="#4472C4",
        edgecolor="#365F91",
        linewidth=0.8,
        label=r"Final objective $F(x)$",
        zorder=3
    )
    bar2 = ax2.bar(
        x + width / 2,
        sla,
        width,
        color="#F77F7F",
        edgecolor="#C05050",
        linewidth=0.8,
        label="SLA violation (%)",
        zorder=3
    )

    # =========================================================
    # 6. Left axis - objective
    # =========================================================
    ax1.set_ylabel(r"Final objective $F(x)$" + "\n(lower is better)", color="#4472C4")
    ax1.tick_params(axis="y", labelcolor="#4472C4", colors="#4472C4")
    obj_min, obj_max = objective.min(), objective.max()
    ax1.set_ylim(obj_min - 0.07, obj_max + 0.11)

    # =========================================================
    # 7. Right axis - SLA
    # =========================================================
    ax2.set_ylabel("SLA violation (%)", color="#E74C3C")
    ax2.tick_params(axis="y", labelcolor="#E74C3C", colors="#E74C3C")
    ax2.spines["right"].set_color("#E74C3C")
    ax2.yaxis.label.set_color("#E74C3C")
    ax2.set_ylim(0, max(3.0, sla.max() * 1.9 if sla.max() > 0 else 3.0))

    # =========================================================
    # 8. X axis
    # =========================================================
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_labels, fontsize=10, fontweight="bold")
    ax1.set_xlim(-0.6, 2.6)

    # =========================================================
    # 9. Grid and spines
    # =========================================================
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25, color="#BFBFBF")
    ax1.set_axisbelow(True)

    # =========================================================
    # 10. Vertical dashed separators (light blue)
    # =========================================================
    for xpos in [0.5, 1.5]:
        ax1.axvline(
            x=xpos,
            ymin=0.0,
            ymax=1.0,
            color="#9DC3E6",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
            zorder=1
        )

    # =========================================================
    # 11. Value labels
    # =========================================================
    for bar in bar1:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#C00000",
            fontweight="bold",
            zorder=5
        )

    for bar in bar2:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.05,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#C00000",
            fontweight="bold",
            zorder=5
        )

    # =========================================================
    # 12. Legend
    # =========================================================
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC"
    )

    # =========================================================
    # 13. Extra cue text ("Iterations")
    # =========================================================
    ax1.text(
        -0.07, 1.02,
        "Iterations",
        transform=ax1.transAxes,
        color="#5B9BD5",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom"
    )

    # =========================================================
    # 14. Bottom explanation box (moved upward)
    # =========================================================
    fig.text(
        0.5,
        0.075,
        "SLSQP satisfies linearized constraints but fails to guarantee physical reliability,\nespecially under stress conditions.",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", edgecolor="#5B9BD5", facecolor="white", linewidth=1.2)
    )

    # =========================================================
    # 15. Layout
    # =========================================================
    fig.subplots_adjust(
        left=0.11,
        right=0.89,
        top=0.90,
        bottom=0.26
    )

    # =========================================================
    # 16. Save
    # =========================================================
    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.close()
    print(f"Saved: {output_file.resolve()}")


# =====================================================
# Fig.10
# =====================================================

def prepare_fig10():
    file = RESULT_DIR / "ablation" / "ablation_results.csv"
    if not file.exists():
        print("Missing:", file)
        return

    df = pd.read_csv(file)
    result = df.groupby("Algorithm").agg({
        "Mean_Objective": "mean",
        "Energy_Cost": "mean",
        "Carbon_Emission": "mean",
        "SLA_Violation": "mean"
    }).reset_index()
    result.columns = [
        "variant",
        "objective",
        "cost",
        "carbon",
        "sla"
    ]
    save_excel("Fig10_ablation.xlsx", {"ablation": result})


def plot_fig10():
    # =========================================================
    # 1. File path
    # =========================================================

    input_file = RESULT_DIR / "paper/figure_data" / "Fig10_ablation.xlsx"
    output_file = OUTPUT_DIR / "Fig10_ablation.tif"

    # =========================================================
    # 2. Nature style
    # =========================================================

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 9,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

    # =========================================================
    # 3. Read data
    # =========================================================

    df = pd.read_excel(input_file, sheet_name="ablation")
    df.columns = [str(c).strip() for c in df.columns]

    # =========================================================
    # 4. Calculate degradation
    # =========================================================

    baseline = df.iloc[0]
    objective_change = (df["objective"] - baseline["objective"]) / baseline["objective"] * 100
    cost_change = (df["cost"] - baseline["cost"]) / baseline["cost"] * 100
    carbon_change = (df["carbon"] - baseline["carbon"]) / baseline["carbon"] * 100
    sla_change = df["sla"] * 100

    # =========================================================
    # 5. Figure layout
    # =========================================================

    fig = plt.figure(figsize=(9.0, 3.8), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.45)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # =========================================================
    # 6. Left panel bar chart
    # =========================================================

    colors = [
        "#4472C4",
        "#ED7D31",
        "#70AD47"
    ]
    labels = [
        "Full ORA\n(Archive + Resonance)",
        "w/o Archive\nguidance",
        "w/o Resonance\nterm"
    ]
    objective = df["objective"].values
    y = np.arange(len(labels))
    bars = ax1.barh(y, objective, height=0.48, color=colors)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel(r"Mean objective $F(x)$")
    ax1.set_title("Ablation study\n(mean objective $F(x)$)", fontweight="bold")
    ax1.set_xlim(objective.min() - 0.02, objective.max() + 0.05)

    for i, b in enumerate(bars):
        value = objective[i]
        if i == 0:
            text = f"{value:.3f}"
        else:
            text = (f"{value:.3f} (+{objective_change[i]:.1f}%)")
        ax1.text(
            b.get_width() + 0.002,
            b.get_y() + b.get_height() / 2, text,
            va="center",
            fontsize=9,
            color="#333333"
        )

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="x", linestyle="--", alpha=0.25)

    # =========================================================
    # 7. Right panel heatmap
    # =========================================================

    impact = np.vstack([cost_change.values, carbon_change.values, sla_change.values]).T
    heat_data = np.array([impact[0], impact[1], impact[2]])
    im = ax2.imshow(heat_data, cmap="Reds", aspect="auto")
    ax2.set_title("Impact on key metrics\n(Δ vs. Full ORA, %)", fontweight="bold", pad=12)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels([
        "Cost\n(%)",
        "Carbon\n(%)",
        "SLA\n(%)"
    ], fontsize=9)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels([
        "Full ORA",
        "No Archive",
        "No Resonance"
    ], fontsize=9)

    # annotation
    for i in range(heat_data.shape[0]):
        for j in range(heat_data.shape[1]):
            value = heat_data[i, j]
            ax2.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=9, color="black")

    # remove frame
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # =========================================================
    # 8. Bottom explanation
    # =========================================================

    fig.text(
        0.5,
        0.035,
        "Removing adaptive components degrades objective quality and increases economic and environmental penalties.",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", edgecolor="#4472C4", facecolor="white", linewidth=1.2)
    )

    # =========================================================
    # 9. Save
    # =========================================================

    fig.subplots_adjust(
        left=0.18,
        right=0.95,
        top=0.85,
        bottom=0.20
    )
    plt.savefig(output_file, dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close()
    print(f"Saved: {output_file.resolve()}")


# =====================================================
# Fig.11
# =====================================================

def prepare_fig11():
    file = RESULT_DIR / "scalability" / "scalability_results.csv"

    if not file.exists():
        print("Missing:", file)
        return

    df = pd.read_csv(file)

    # Objective data
    objective = df.groupby(["dimension", "algorithm"]).agg({"fitness": ["mean", "std"]}).reset_index()
    objective.columns = [
        "dimension",
        "algorithm",
        "objective_mean",
        "objective_std"
    ]

    # Runtime data
    if "runtime_seconds" in df.columns:
        runtime = df.groupby(["dimension", "algorithm"])["runtime_seconds"].mean().reset_index()
    else:
        runtime = pd.DataFrame()

    save_excel("Fig11_scalability.xlsx", {"objective": objective, "runtime": runtime})


def plot_fig11():
    # =========================================================
    # 1. File path
    # =========================================================
    input_file = RESULT_DIR / "paper/figure_data" / "Fig11_scalability.xlsx"
    output_file = OUTPUT_DIR / "Fig11_scalability.tif"

    # =========================================================
    # 2. Nature style
    # =========================================================
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

    # =========================================================
    # 3. Read data
    # =========================================================
    objective_df = pd.read_excel(input_file, sheet_name="objective")
    runtime_df = pd.read_excel(input_file, sheet_name="runtime")
    objective_df.columns = [str(c).strip() for c in objective_df.columns]
    runtime_df.columns = [str(c).strip() for c in runtime_df.columns]
    required_obj = ["dimension", "algorithm", "objective_mean"]
    required_run = ["dimension", "algorithm", "runtime_seconds"]
    missing_obj = [c for c in required_obj if c not in objective_df.columns]
    missing_run = [c for c in required_run if c not in runtime_df.columns]

    if missing_obj:
        raise ValueError(f"Missing columns in objective sheet: {missing_obj}")
    if missing_run:
        raise ValueError(f"Missing columns in runtime sheet: {missing_run}")

    # =========================================================
    # 4. Select algorithms
    # =========================================================
    algorithms = ["ORA", "DE"]
    dimensions = sorted(objective_df["dimension"].unique())
    objective_data = {}
    runtime_data = {}

    for alg in algorithms:
        temp_obj = objective_df[objective_df["algorithm"] == alg].sort_values("dimension")
        temp_run = runtime_df[runtime_df["algorithm"] == alg].sort_values("dimension")
        objective_data[alg] = temp_obj["objective_mean"].to_numpy(dtype=float)
        runtime_data[alg] = temp_run["runtime_seconds"].to_numpy(dtype=float)

    x = np.array(dimensions, dtype=float)

    # =========================================================
    # 5. Figure
    # =========================================================
    fig, ax1 = plt.subplots(figsize=(5.2, 5.2), dpi=300)
    ax2 = ax1.twinx()

    # =========================================================
    # 6. Colors
    # =========================================================
    colors = {
        "ORA": "#4472C4",
        "DE": "#ED7D31"
    }

    # =========================================================
    # 7. Plot objective curves
    # =========================================================
    for alg in algorithms:
        ax1.plot(
            x,
            objective_data[alg],
            color=colors[alg],
            linewidth=2.0,
            marker="o",
            markersize=5.5,
            markeredgewidth=0.8,
            label=f"Objective ({alg})"
        )

    # =========================================================
    # 8. Plot runtime curves
    # =========================================================
    for alg in algorithms:
        ax2.plot(
            x,
            runtime_data[alg],
            color=colors[alg],
            linewidth=1.8,
            linestyle="--",
            marker="s",
            markersize=5.5,
            markeredgewidth=0.8,
            label=f"Runtime ({alg})"
        )

    # =========================================================
    # 9. Axis labels
    # =========================================================
    ax1.set_xlabel("Decision dimension")
    ax1.set_ylabel(r"Mean objective $F(x)$" + "\n(lower is better)", color="#4472C4")
    ax2.set_ylabel("Runtime (s)", color="#C00000")
    ax1.tick_params(axis="y", labelcolor="#4472C4", colors="#4472C4")
    ax2.tick_params(axis="y", labelcolor="#C00000", colors="#C00000")
    ax1.spines["left"].set_color("#4472C4")
    ax2.spines["right"].set_color("#C00000")
    ax1.yaxis.label.set_color("#4472C4")
    ax2.yaxis.label.set_color("#C00000")

    # =========================================================
    # 10. Axis limits and ticks
    # =========================================================
    ax1.set_xticks(x)
    obj_all = np.concatenate([objective_data[a] for a in algorithms])
    run_all = np.concatenate([runtime_data[a] for a in algorithms])
    left_ymin = 0.64
    left_ymax = max(0.755, obj_all.max() + 0.008)
    right_ymin = 4
    right_ymax = max(14.0, run_all.max() + 0.8)
    ax1.set_ylim(left_ymin, left_ymax)
    ax2.set_ylim(right_ymin, right_ymax)
    left_ticks = [0.64, 0.68, 0.72, 0.76]
    left_ticks = [v for v in left_ticks if left_ymin <= v <= left_ymax + 1e-9]
    ax1.set_yticks(left_ticks)
    right_ticks = [4, 8, 12, 16]
    right_ticks = [v for v in right_ticks if right_ymin <= v <= right_ymax + 1e-9]
    ax2.set_yticks(right_ticks)

    # =========================================================
    # 11. Grid and spines
    # =========================================================
    ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25, color="#BFBFBF")
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # =========================================================
    # 12. Legend (moved into upper-left blank area)
    # =========================================================
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        borderpad=0.7,
        labelspacing=0.6,
        handlelength=2.2
    )

    # =========================================================
    # 13. Bottom explanation box (moved slightly downward)
    # =========================================================
    fig.text(
        0.5,
        0.085,
        "ORA maintains stable solution quality with moderate\ncomputational growth as dimension increases.",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", edgecolor="#4472C4", facecolor="white", linewidth=1.2)
    )

    # =========================================================
    # 14. Layout
    # =========================================================
    fig.subplots_adjust(
        left=0.16,
        right=0.84,
        top=0.93,
        bottom=0.21
    )

    # =========================================================
    # 15. Save
    # =========================================================
    plt.savefig(output_file, dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close()
    print(f"Saved: {output_file.resolve()}")


# =====================================================
# Fig.12
# =====================================================

def prepare_fig12():
    file = RESULT_DIR / "multinode_scalability" / "multinode_results.csv"

    if not file.exists():
        print("Missing:", file)
        return

    df = pd.read_csv(file)
    runtime = df.groupby(["scale", "algorithm"]).agg({"runtime_seconds": "mean"}).reset_index()
    runtime.columns = [
        "nodes",
        "algorithm",
        "runtime"
    ]
    objective = pd.DataFrame()

    if "fitness" in df.columns:
        objective = df.groupby(["scale", "algorithm"])["fitness"].mean().reset_index()
        objective.columns = [
            "nodes",
            "algorithm",
            "objective"
        ]
    feasible = pd.DataFrame()

    if "fully_feasible_rate" in df.columns:
        feasible = df.groupby(["scale", "algorithm"])["fully_feasible_rate"].mean().reset_index()
        feasible.columns = [
            "nodes",
            "algorithm",
            "feasible_rate"
        ]

    save_excel("Fig12_multinode_scalability.xlsx", {
        "runtime": runtime,
        "objective": objective,
        "feasible": feasible
    })


def plot_fig12():
    # =========================================================
    # 1. File path
    # =========================================================
    input_file = RESULT_DIR / "paper/figure_data" / "Fig12_multinode_scalability.xlsx"
    output_file = OUTPUT_DIR / "Fig12_multinode_scalability.tif"

    # =========================================================
    # 2. Nature style
    # =========================================================
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

    # =========================================================
    # 3. Read data
    # =========================================================
    runtime_df = pd.read_excel(input_file, sheet_name="runtime")
    objective_df = pd.read_excel(input_file, sheet_name="objective")
    runtime_df.columns = [str(c).strip() for c in runtime_df.columns]
    objective_df.columns = [str(c).strip() for c in objective_df.columns]
    required_runtime = ["nodes", "algorithm", "runtime"]
    required_objective = ["nodes", "algorithm", "objective"]
    missing_runtime = [c for c in required_runtime if c not in runtime_df.columns]
    missing_objective = [c for c in required_objective if c not in objective_df.columns]

    if missing_runtime:
        raise ValueError(f"Missing columns in runtime sheet: {missing_runtime}")
    if missing_objective:
        raise ValueError(f"Missing columns in objective sheet: {missing_objective}")

    # =========================================================
    # 4. Select algorithms
    # =========================================================
    algorithms = ["ORA", "DE"]
    nodes = sorted(runtime_df["nodes"].unique())
    runtime_data = {}
    objective_data = {}

    for alg in algorithms:
        runtime_data[alg] = runtime_df[runtime_df["algorithm"] == alg] \
            .sort_values("nodes")["runtime"].to_numpy(dtype=float)
        objective_data[alg] = objective_df[objective_df["algorithm"] == alg] \
            .sort_values("nodes")["objective"].to_numpy(dtype=float)

    x = np.array(nodes, dtype=float)

    # =========================================================
    # 5. Figure
    # =========================================================
    fig, ax1 = plt.subplots(figsize=(5.2, 5.2), dpi=300)
    ax2 = ax1.twinx()

    # =========================================================
    # 6. Colors
    # =========================================================
    runtime_colors = {"ORA": "#4472C4", "DE": "#ED7D31"}
    objective_colors = {"ORA": "#70AD47", "DE": "#A9D18E"}

    # =========================================================
    # 7. Runtime curves
    # =========================================================
    for alg in algorithms:
        ax1.plot(
            x,
            runtime_data[alg],
            color=runtime_colors[alg],
            linewidth=2.0,
            linestyle="-" if alg == "ORA" else "--",
            marker="o",
            markersize=5.5,
            label=f"{alg} runtime"
        )

    # =========================================================
    # 8. Objective curves
    # =========================================================
    for alg in algorithms:
        ax2.plot(
            x,
            objective_data[alg],
            color=objective_colors[alg],
            linewidth=1.8,
            linestyle="-" if alg == "ORA" else "--",
            marker="^",
            markersize=6,
            label=f"{alg} objective"
        )

    # =========================================================
    # 9. Axis labels
    # =========================================================
    ax1.set_xlabel("Number of nodes")
    ax1.set_ylabel("Mean runtime (s)", color="#4472C4")
    ax2.set_ylabel(r"Mean objective $F(x)$", color="#70AD47")
    ax1.tick_params(axis="y", labelcolor="#4472C4", colors="#4472C4")
    ax2.tick_params(axis="y", labelcolor="#70AD47", colors="#70AD47")
    ax1.spines["left"].set_color("#4472C4")
    ax2.spines["right"].set_color("#70AD47")
    ax1.yaxis.label.set_color("#4472C4")
    ax2.yaxis.label.set_color("#70AD47")

    # =========================================================
    # 10. Axis range
    # =========================================================
    runtime_all = np.concatenate(list(runtime_data.values()))
    objective_all = np.concatenate(list(objective_data.values()))
    ax1.set_ylim(6, max(runtime_all.max() + 1.2, 12.5))
    ax2.set_ylim(objective_all.min() - 0.02, objective_all.max() + 0.04)
    ax1.set_xticks(x)
    top_runtime = ax1.get_ylim()[1]
    if top_runtime <= 12.5:
        ax1.set_yticks([6, 8, 10, 12])
    else:
        ax1.set_yticks([6, 8, 10, 12, 14])

    # =========================================================
    # 11. Grid and border
    # =========================================================
    ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25, color="#BFBFBF")
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # =========================================================
    # 12. Legend (slightly upward and to the right)
    # =========================================================
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left",
        bbox_to_anchor=(0.42, 0.995),
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        borderpad=0.7,
        labelspacing=0.6
    )

    # =========================================================
    # 13. Bottom explanation box
    # =========================================================
    fig.text(
        0.5,
        0.085,
        "ORA exhibits lower runtime and more consistent solution\nquality across increasing number of nodes.",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", edgecolor="#4472C4", facecolor="white", linewidth=1.2)
    )

    # =========================================================
    # 14. Layout
    # =========================================================
    fig.subplots_adjust(left=0.16, right=0.84, top=0.93, bottom=0.22)

    # =========================================================
    # 15. Save TIFF
    # =========================================================
    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.close()
    print(f"Saved: {output_file.resolve()}")


# =====================================================
# Fig.13
# =====================================================

def prepare_fig13():
    candidates = [
        RESULT_DIR / "weight_sensitivity" / "tradeoff_metrics.csv",
        RESULT_DIR / "preference_analysis" / "preference_results.csv"
    ]
    file = None

    for f in candidates:
        if f.exists():
            file = f
            break

    if file is None:
        print("Missing preference result file")
        return

    df = pd.read_csv(file)
    columns = df.columns.tolist()
    rename = {}

    if "electricity_cost" in columns:
        rename["electricity_cost"] = "cost"
    if "mean_electricity_cost" in columns:
        rename["mean_electricity_cost"] = "cost"
    if "carbon_emission" in columns:
        rename["carbon_emission"] = "carbon"
    if "mean_carbon_emission" in columns:
        rename["mean_carbon_emission"] = "carbon"
    if "sla_violation" in columns:
        rename["sla_violation"] = "sla"

    result = df.rename(columns=rename)
    save_excel("Fig13_preference_tradeoff.xlsx", {"preference": result})


def plot_fig13():
    # =========================================================
    # 1. file path
    # =========================================================
    input_file = RESULT_DIR / "paper/figure_data" / "Fig13_preference_tradeoff.xlsx"
    output_file = OUTPUT_DIR / "Fig13_preference_tradeoff.tif"

    # =========================================================
    # 2. nature-like style
    # =========================================================
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

    # =========================================================
    # 3. read data
    # =========================================================
    xls = pd.ExcelFile(input_file)
    sheet_name = "preference" if "preference" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(input_file, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]

    def find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(f"Cannot find any of these columns: {candidates}")

    col_weight = find_col(["Weight_Set", "weight_set", "Preference", "preference", "Mode", "mode"])
    col_cost = find_col(["Energy_Cost", "energy_cost", "Cost", "cost"])
    col_carbon = find_col(["Carbon_Emission", "carbon_emission", "Carbon", "carbon"])
    col_sla = find_col(["SLA_Violation", "sla_violation", "SLA", "sla"])
    col_obj = find_col(["Mean_Objective", "mean_objective", "Objective", "objective"])

    # =========================================================
    # 4. normalize names
    # =========================================================
    name_map = {
        "balanced": "Balanced",
        "balance": "Balanced",
        "carbon_priority": "Carbon priority",
        "carbon-priority": "Carbon priority",
        "carbon priority": "Carbon priority",
        "cost_priority": "Cost priority",
        "cost-priority": "Cost priority",
        "cost priority": "Cost priority",
        "energy_priority": "Energy priority",
        "energy-priority": "Energy priority",
        "energy priority": "Energy priority",
        "sla_priority": "SLA priority",
        "sla-priority": "SLA priority",
        "sla priority": "SLA priority"
    }

    df["label"] = df[col_weight].astype(str).str.strip().str.lower().map(name_map).fillna(df[col_weight].astype(str))
    preferred_order = ["Balanced", "Carbon priority", "Cost priority", "Energy priority", "SLA priority"]
    order_map = {k: i for i, k in enumerate(preferred_order)}
    df["order"] = df["label"].map(order_map).fillna(999)
    df = df.sort_values("order").reset_index(drop=True)

    # =========================================================
    # 5. compute absolute trade-off relative to balanced
    # =========================================================
    if "Balanced" not in df["label"].values:
        raise ValueError("Balanced setting is required in the preference sheet.")

    base = df[df["label"] == "Balanced"].iloc[0]
    base_cost = float(base[col_cost])
    base_carbon = float(base[col_carbon])
    df["delta_cost"] = df[col_cost].astype(float) - base_cost
    df["delta_carbon_reduction"] = base_carbon - df[col_carbon].astype(float)
    df["objective_val"] = df[col_obj].astype(float)
    df["sla_val"] = df[col_sla].astype(float)

    # =========================================================
    # 6. colors
    # =========================================================
    colors = {
        "Balanced": "#1565C0",
        "Carbon priority": "#2E7D32",
        "Cost priority": "#E67E22",
        "Energy priority": "#7B1FA2",
        "SLA priority": "#D83A34"
    }

    # =========================================================
    # 7. bubble size mapping
    # =========================================================
    obj = df["objective_val"].to_numpy(dtype=float)
    obj_min, obj_max = obj.min(), obj.max()
    size_min, size_max = 350, 1300

    if np.isclose(obj_min, obj_max):
        df["bubble_size"] = (size_min + size_max) / 2
    else:
        df["bubble_size"] = size_min + (obj - obj_min) / (obj_max - obj_min) * (size_max - size_min)

    def map_size(v):
        if np.isclose(obj_min, obj_max):
            return (size_min + size_max) / 2
        return size_min + (v - obj_min) / (obj_max - obj_min) * (size_max - size_min)

    # =========================================================
    # 8. display coordinates
    # =========================================================
    df["plot_x"] = df["delta_carbon_reduction"]
    df["plot_y"] = df["delta_cost"]
    display_shift = {
        "Balanced": (0.000, 0.000),
        "Carbon priority": (0.000, 0.000),
        "Cost priority": (-0.010, -0.0035),
        "Energy priority": (0.000, 0.000),
        "SLA priority": (0.000, 0.000)
    }

    for idx, row in df.iterrows():
        dx, dy = display_shift.get(row["label"], (0.0, 0.0))
        df.loc[idx, "plot_x"] = row["delta_carbon_reduction"] + dx
        df.loc[idx, "plot_y"] = row["delta_cost"] + dy

    # =========================================================
    # 9. figure
    # =========================================================
    fig, ax = plt.subplots(figsize=(8.0, 7.6), dpi=300)
    x = df["plot_x"].to_numpy(dtype=float)
    y = df["plot_y"].to_numpy(dtype=float)
    x_min = min(x.min() - 0.03, -0.20)
    x_max = max(x.max() + 0.14, 0.28)
    y_min = min(y.min() - 0.012, -0.047)
    y_max = max(y.max() + 0.018, 0.040)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # =========================================================
    # 10. grid + reference lines
    # =========================================================
    ax.grid(True, linestyle="--", linewidth=0.6, color="#D9D9D9", alpha=0.7, zorder=0)
    ax.axhline(0, color="black", linewidth=1.0, zorder=1)
    ax.axvline(0, color="black", linewidth=1.0, zorder=1)

    # =========================================================
    # 11. leader lines
    # =========================================================
    for _, row in df.iterrows():
        if abs(row["plot_x"] - row["delta_carbon_reduction"]) > 1e-12 or abs(row["plot_y"] - row["delta_cost"]) > 1e-12:
            ax.plot(
                [row["delta_carbon_reduction"], row["plot_x"]],
                [row["delta_cost"], row["plot_y"]],
                color="#8A8A8A",
                linewidth=0.9,
                linestyle=":",
                zorder=2
            )

    # =========================================================
    # 12. scatter
    # =========================================================
    draw_order = ["Balanced", "Cost priority", "SLA priority", "Energy priority", "Carbon priority"]
    for name in draw_order:
        row = df[df["label"] == name].iloc[0]
        ax.scatter(
            row["plot_x"],
            row["plot_y"],
            s=row["bubble_size"],
            color=colors[name],
            edgecolor="white",
            linewidth=1.5,
            alpha=0.82,
            zorder=4
        )

    # =========================================================
    # 13. draw once to compute d
    # =========================================================
    fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.20)
    fig.canvas.draw()

    yticks = ax.get_yticks()
    if len(yticks) >= 2:
        d_data = abs(yticks[1] - yticks[0])
    else:
        d_data = 0.01

    ax_bbox_in = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax_height_in = ax_bbox_in.height
    ax_pos = ax.get_position()  # in figure fraction
    d_axes = d_data / (y_max - y_min)
    d_points = d_axes * ax_height_in * 72.0
    d_fig = d_axes * ax_pos.height

    # =========================================================
    # 14. labels
    # =========================================================
    label_offsets = {
        "Balanced": (14, 12),
        "Carbon priority": (10, 12),
        "Cost priority": (-70, -12),
        "Energy priority": (10, -38 - d_points / 8.0),
        "SLA priority": (8, 10)
    }

    for _, row in df.iterrows():
        name = row["label"]
        dx, dy = label_offsets.get(name, (8, 8))
        ax.annotate(
            f"{name}\nF={row['objective_val']:.3f}",
            xy=(row["plot_x"], row["plot_y"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9,
            color=colors[name],
            fontweight="bold" if name == "Balanced" else None,
            zorder=6
        )

    # =========================================================
    # 15. axes labels
    # =========================================================
    ax.set_xlabel("Carbon reduction relative to balanced", color="#2E7D32", labelpad=10)
    ax.set_ylabel("Energy cost change relative to balanced", color="#1565C0", labelpad=10)

    # =========================================================
    # 16. preference legend in C area
    # =========================================================
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=9,
               markerfacecolor=colors[name], markeredgecolor=colors[name], label=name)
        for name in preferred_order if name in df["label"].values
    ]
    leg1 = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.00, 0.53 - d_axes / 8.0),
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        borderpad=0.7,
        labelspacing=0.7,
        handlelength=1.0,
        handletextpad=0.6,
        columnspacing=1.2
    )
    leg1.get_frame().set_edgecolor("#D9D9D9")
    ax.add_artist(leg1)

    # =========================================================
    # 17. bubble-size legend
    # =========================================================
    size_vals = np.linspace(obj_min, obj_max, 4)
    size_x = [0.68, 0.76, 0.84, 0.92]
    bubble_y = 0.89
    line_y = 0.875 - d_axes / 2.0
    label_y = 0.847 - d_axes / 2.0
    ax.text(0.68, 0.93, r"Mean objective $F(x)$", transform=ax.transAxes, fontsize=11, ha="left", va="bottom")

    for sx, sv in zip(size_x, size_vals):
        ax.scatter(
            sx, bubble_y,
            s=map_size(sv),
            transform=ax.transAxes,
            color="#7A7A7A",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.9,
            zorder=4
        )

    ax.annotate(
        "", xy=(0.95, line_y), xytext=(0.69, line_y),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", lw=0.9, color="#555555")
    )

    for sx, sv in zip(size_x, size_vals):
        ax.text(
            sx, label_y,
            f"{sv:.2f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9
        )

    # =========================================================
    # 18. quadrant texts
    # =========================================================
    quad_kw = dict(color="#555555", fontsize=9, fontstyle="italic")
    ax.text(0.02, 0.68, "Higher cost\nLower carbon", transform=ax.transAxes, ha="left", va="center", **quad_kw)
    ax.text(0.98, 0.68, "Higher cost\nHigher carbon reduction", transform=ax.transAxes, ha="right", va="center",
            **quad_kw)
    ax.text(0.025, 0.17, "Lower cost\nLower carbon", transform=ax.transAxes, ha="left", va="center", **quad_kw)
    ax.text(0.98, 0.17, "Lower cost\nHigher carbon reduction", transform=ax.transAxes, ha="right", va="center",
            **quad_kw)

    # =========================================================
    # 19. style
    # =========================================================
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # =========================================================
    # 20. bottom explanation box
    # =========================================================

    fig.text(
        0.5,
        0.085,
        r"$\bf{Operational\ preference\ shifts\ the\ cost\!-\!carbon\ trade\!-\!off.}$"
        + "\n"
          "Using the balanced setting as reference, each preference yields a distinct operating point\n"
          "under the same feasibility-audited framework.",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", edgecolor="#1565C0", facecolor="white", linewidth=1.2)
    )

    # =========================================================
    # 21. save
    # =========================================================
    plt.savefig(output_file, dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close()
    print(f"Saved: {output_file.resolve()}")


# =====================================================
# Main
# =====================================================

def run_reproduce_figures():
    print("=" * 80)
    print("Generating figure data for manuscript")
    print("=" * 80)
    prepare_fig6()
    plot_fig6()
    prepare_fig7()
    plot_fig7()
    prepare_fig8()
    plot_fig8()
    prepare_fig9()
    plot_fig9()
    prepare_fig10()
    plot_fig10()
    prepare_fig11()
    plot_fig11()
    prepare_fig12()
    plot_fig12()
    prepare_fig13()
    plot_fig13()
    print()
    print("All figure data generated.")
    print(OUTPUT_DIR)
