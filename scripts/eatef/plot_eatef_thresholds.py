#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text

plt.rcParams["figure.dpi"] = 150

# ==========================================================
# Load burden results
# ==========================================================

results = pd.read_csv(
    "results/eatef_burden_results_complex.tsv",
    sep="\t",
)

misfit = results[
    results["Method"] == "MisFit_S"
].copy()

alpha = results[
    results["Method"] == "AlphaMissense"
].copy()

revel = results[
    results["Method"] == "REVEL"
].copy()

cadd = results[
    results["Method"] == "CADD"
].copy()

# ==========================================================
# Colors
# ==========================================================

colors = {
    "CADD": "#1f77b4",
    "REVEL": "#ff7f0e",
    "AlphaMissense": "#2ca02c",
    "MisFit-S": "#d62728",
}

# ==========================================================
# Figure
# ==========================================================

fig, ax = plt.subplots(figsize=(9, 6.5))

texts = []

# ==========================================================
# Plot function
# ==========================================================

def plot_method(df, color, label):

    if df.empty:
        return

    df = df.dropna(
        subset=[
            "EstimatedRiskVariants",
            "Enrichment",
            "Observed",
        ]
    )

    if len(df) == 0:
        return

    sizes = (
        120
        + 280
        * df["Observed"]
        / df["Observed"].max()
    )

    ax.scatter(
        df["EstimatedRiskVariants"],
        df["Enrichment"],
        s=sizes,
        color=color,
        edgecolors="white",
        linewidths=1.5,
        alpha=0.9,
        label=label,
        zorder=3,
    )

    for _, row in df.iterrows():

        if label == "CADD":
            label_text = f"≥{int(row['Threshold'])}"

        else:
            label_text = f"≥{row['Threshold']:g}"

        texts.append(

            ax.text(
                row["EstimatedRiskVariants"],
                row["Enrichment"],
                label_text,
                fontsize=10,
                zorder=4,
            )

        )

# ==========================================================
# Plot methods
# ==========================================================

plot_method(
    cadd,
    colors["CADD"],
    "CADD",
)

plot_method(
    revel,
    colors["REVEL"],
    "REVEL",
)

plot_method(
    alpha,
    colors["AlphaMissense"],
    "AlphaMissense",
)

plot_method(
    misfit,
    colors["MisFit-S"],
    "MisFit-S",
)

# ==========================================================
# Adjust labels
# ==========================================================

adjust_text(
    texts,
    ax=ax,
    expand_points=(2.0, 2.0),
    expand_text=(1.7, 1.7),
    force_points=1.0,
    force_text=1.2,
    only_move={
        "text": "xy",
        "static": "xy",
        "explode": "xy",
        "pull": "xy",
    },
    lim=500,
)

# ==========================================================
# Formatting
# ==========================================================

ax.set_xlabel(
    "Estimated number of risk variants",
    fontsize=15,
)

ax.set_ylabel(
    "Enrichment rate",
    fontsize=15,
)

ax.set_title(
    "Complex EA/TEF",
    fontsize=24,
    fontweight="bold",
)

ax.grid(
    True,
    linestyle="--",
    alpha=0.35,
)

# ==========================================================
# Dynamic axis limits
# ==========================================================

xmin = results["EstimatedRiskVariants"].min()
xmax = results["EstimatedRiskVariants"].max()

xpad = (xmax - xmin) * 0.10

ax.set_xlim(
    xmin - xpad,
    xmax + xpad,
)

ymin = max(
    0,
    results["Enrichment"].min() - 0.08,
)

ymax = results["Enrichment"].max() + 0.10

ax.set_ylim(
    ymin,
    ymax,
)

# ==========================================================
# Legend
# ==========================================================

ax.legend(
    title="Methods",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False,
    fontsize=12,
    title_fontsize=14,
)

plt.tight_layout()

# ==========================================================
# Save
# ==========================================================

plt.savefig(
    "results/eatef_threshold_optimization.png",
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    "results/eatef_threshold_optimization.pdf",
    bbox_inches="tight",
)

plt.show()