
#!/usr/bin/env python3

"""
Apply ontology-derived pathway clusters to the existing significant-pathway
comparison table and regenerate the LoF-versus-Dmis scatter plot.

Run this after pathway_similarity_network_ontology.py.
"""

import os
import textwrap

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator


# ==========================================================
# Files
# ==========================================================

INPUT_TABLE = "results/significant_pathway_union_lof_vs_dmis.tsv"
ONTOLOGY_CLUSTER_FILE = "results/pathway_network_clusters.tsv"

OUTPUT_TABLE = "results/significant_pathway_union_lof_vs_dmis.tsv"
OUTPUT_PDF = "results/significant_pathway_union_lof_vs_dmis.pdf"
OUTPUT_PNG = "results/significant_pathway_union_lof_vs_dmis.png"


# ==========================================================
# Parameters
# ==========================================================

FIGURE_WIDTH = 20
FIGURE_HEIGHT = 11.5
PLOT_DPI = 600

MARKER_SIZE = 350
MARKER_EDGE_WIDTH = 1.6
NUMBER_FONT_SIZE = 11.5
KEY_WRAP_WIDTH = 38

CI_AXIS_EXPANSION = 1.18
CI_ARROW_SIZE = 9
CI_LINE_WIDTH = 1.35
CI_ALPHA = 0.25

KEY_HEADER_FONT_SIZE = 13.2
KEY_ITEM_FONT_SIZE = 11.0
KEY_BOX_ALPHA = 0.055

HIGH_CONTRAST_CLUSTER_COLORS = [
    "#0047AB",
    "#FF3B00",
    "#00A651",
    "#9C27B0",
    "#FFC000",
    "#00B8D9",
    "#E91E63",
    "#222222",
]


# ==========================================================
# Helpers
# ==========================================================

def readable_pathway_name(pathway_name):
    readable = str(pathway_name).strip()

    for prefix in ("GOBP_", "GOCC_", "GOMF_"):
        if readable.startswith(prefix):
            readable = readable[len(prefix):]
            break

    return readable.replace("_", " ").lower()


def wrap_pathway_name(pathway_name):
    return "\n".join(
        textwrap.wrap(
            readable_pathway_name(pathway_name),
            width=KEY_WRAP_WIDTH,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def format_tick(value, position):
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def draw_clipped_interval(
    axis,
    point_x,
    point_y,
    lower,
    upper,
    horizontal,
    limit,
    color,
):
    """Draw a light, cluster-colored 95% confidence interval."""
    visible_lower = max(float(lower), 0.0)
    visible_upper = min(float(upper), limit)

    if horizontal:
        axis.plot(
            [visible_lower, visible_upper],
            [point_y, point_y],
            linewidth=CI_LINE_WIDTH,
            color=color,
            alpha=CI_ALPHA,
            solid_capstyle="round",
            zorder=2,
        )

        if upper > limit:
            axis.annotate(
                "",
                xy=(limit, point_y),
                xytext=(limit - 0.30, point_y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "linewidth": CI_LINE_WIDTH,
                    "color": color,
                    "alpha": CI_ALPHA,
                    "mutation_scale": CI_ARROW_SIZE,
                },
                zorder=2,
            )
    else:
        axis.plot(
            [point_x, point_x],
            [visible_lower, visible_upper],
            linewidth=CI_LINE_WIDTH,
            color=color,
            alpha=CI_ALPHA,
            solid_capstyle="round",
            zorder=2,
        )

        if upper > limit:
            axis.annotate(
                "",
                xy=(point_x, limit),
                xytext=(point_x, limit - 0.30),
                arrowprops={
                    "arrowstyle": "-|>",
                    "linewidth": CI_LINE_WIDTH,
                    "color": color,
                    "alpha": CI_ALPHA,
                    "mutation_scale": CI_ARROW_SIZE,
                },
                zorder=2,
            )



def display_title(text):
    small = {"and","or","of","the","in","on","to","for","with"}
    words = str(text).replace("_"," ").strip().lower().split()
    out=[]
    for i,w in enumerate(words):
        if i>0 and w in small:
            out.append(w)
        else:
            out.append(w.capitalize())
    return " ".join(out)

def cluster_color(cluster):
    """Return the exact same cluster color used by the network script."""
    return HIGH_CONTRAST_CLUSTER_COLORS[
        (int(cluster) - 1) % len(HIGH_CONTRAST_CLUSTER_COLORS)
    ]


# ==========================================================
# Load and merge ontology assignments
# ==========================================================

if not os.path.exists(INPUT_TABLE):
    raise FileNotFoundError(
        f"Comparison table not found: {INPUT_TABLE}"
    )

if not os.path.exists(ONTOLOGY_CLUSTER_FILE):
    raise FileNotFoundError(
        "Ontology cluster file not found. Run "
        "pathway_similarity_network_ontology.py first: "
        f"{ONTOLOGY_CLUSTER_FILE}"
    )

data = pd.read_csv(
    INPUT_TABLE,
    sep="\t",
    low_memory=False,
)

clusters = pd.read_csv(
    ONTOLOGY_CLUSTER_FILE,
    sep="\t",
    low_memory=False,
)

required_data = {
    "Pathway",
    "LoF_Enrichment",
    "Dmis_Enrichment",
    "LoF_CI_Lower",
    "LoF_CI_Upper",
    "Dmis_CI_Lower",
    "Dmis_CI_Upper",
    "Minimum_FWER",
}

missing = required_data - set(data.columns)

if missing:
    raise ValueError(
        "Comparison table is missing columns: "
        f"{sorted(missing)}"
    )

required_clusters = {
    "Pathway",
    "Cluster",
    "Ontology_Cluster_Name",
    "Ontology_Ancestor_GO_ID",
    "GO_ID",
}

missing = required_clusters - set(clusters.columns)

if missing:
    raise ValueError(
        "Ontology cluster table is missing columns: "
        f"{sorted(missing)}"
    )

data["Pathway"] = data["Pathway"].astype(str).str.strip()
clusters["Pathway"] = clusters["Pathway"].astype(str).str.strip()

# Remove old clustering fields before replacing them.
columns_to_replace = [
    "Cluster",
    "Ontology_Cluster_Name",
    "Ontology_Ancestor_GO_ID",
    "GO_ID",
]

data = data.drop(
    columns=[
        column
        for column in columns_to_replace
        if column in data.columns
    ]
)

data = data.merge(
    clusters[
        [
            "Pathway",
            "Cluster",
            "Ontology_Cluster_Name",
            "Ontology_Ancestor_GO_ID",
            "GO_ID",
        ]
    ],
    on="Pathway",
    how="left",
    validate="one_to_one",
)

if data["Cluster"].isna().any():
    missing_pathways = data.loc[
        data["Cluster"].isna(),
        "Pathway",
    ].tolist()

    raise ValueError(
        "Missing ontology cluster assignments for: "
        f"{missing_pathways[:20]}"
    )

data["Cluster"] = data["Cluster"].astype(int)

data = data.sort_values(
    ["Cluster", "Minimum_FWER", "Pathway"],
    ascending=[True, True, True],
).reset_index(drop=True)

# Renumber plot labels after ontology-based sorting.
if "Plot_Number" in data.columns:
    data["Plot_Number"] = np.arange(1, len(data) + 1)
else:
    data.insert(
        0,
        "Plot_Number",
        np.arange(1, len(data) + 1),
    )

front_columns = [
    "Plot_Number",
    "Pathway",
    "Cluster",
    "Ontology_Cluster_Name",
    "Ontology_Ancestor_GO_ID",
    "GO_ID",
]

remaining_columns = [
    column
    for column in data.columns
    if column not in front_columns
]

data[
    front_columns + remaining_columns
].to_csv(
    OUTPUT_TABLE,
    sep="\t",
    index=False,
)


# ==========================================================
# Plot
# ==========================================================

x = pd.to_numeric(
    data["LoF_Enrichment"],
    errors="raise",
).to_numpy(dtype=float)

y = pd.to_numeric(
    data["Dmis_Enrichment"],
    errors="raise",
).to_numpy(dtype=float)

observed_max = max(
    float(np.nanmax(x)),
    float(np.nanmax(y)),
    1.0,
)

axis_limit = max(
    1.0,
    observed_max * CI_AXIS_EXPANSION,
)

figure = plt.figure(
    figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)
)

grid = figure.add_gridspec(
    1,
    2,
    width_ratios=[1.28, 1.0],
    wspace=0.12,
)

axis = figure.add_subplot(grid[0, 0])
key_axis = figure.add_subplot(grid[0, 1])
key_axis.set_xlim(0, 1)
key_axis.set_ylim(0, 1)
key_axis.axis("off")

axis.plot(
    [0, axis_limit],
    [0, axis_limit],
    linestyle="--",
    linewidth=1.25,
    color="#8A8A8A",
    alpha=0.72,
    zorder=1,
)

# Draw confidence intervals first so points remain crisp and prominent.
for _, row in data.iterrows():
    point_x = float(row["LoF_Enrichment"])
    point_y = float(row["Dmis_Enrichment"])
    color = cluster_color(row["Cluster"])

    draw_clipped_interval(
        axis,
        point_x,
        point_y,
        row["LoF_CI_Lower"],
        row["LoF_CI_Upper"],
        horizontal=True,
        limit=axis_limit,
        color=color,
    )

    draw_clipped_interval(
        axis,
        point_x,
        point_y,
        row["Dmis_CI_Lower"],
        row["Dmis_CI_Upper"],
        horizontal=False,
        limit=axis_limit,
        color=color,
    )

for cluster in sorted(data["Cluster"].unique()):
    subset = data[data["Cluster"].eq(cluster)]
    color = cluster_color(cluster)

    axis.scatter(
        subset["LoF_Enrichment"],
        subset["Dmis_Enrichment"],
        s=MARKER_SIZE,
        facecolor=color,
        edgecolor="black",
        linewidth=MARKER_EDGE_WIDTH,
        zorder=3,
        clip_on=False,
    )

for _, row in data.iterrows():
    number_text = axis.text(
        row["LoF_Enrichment"],
        row["Dmis_Enrichment"],
        str(int(row["Plot_Number"])),
        ha="center",
        va="center",
        fontsize=NUMBER_FONT_SIZE,
        fontweight="bold",
        color="white",
        zorder=4,
        clip_on=False,
    )

    number_text.set_path_effects(
        [
            path_effects.withStroke(
                linewidth=1.35,
                foreground="black",
            )
        ]
    )

axis_padding = max(0.45, axis_limit * 0.035)
axis.set_xlim(-axis_padding, axis_limit)
axis.set_ylim(-axis_padding, axis_limit)
axis.set_aspect("equal", adjustable="box")

# Place the visible axes at the true zero coordinates. This keeps pathways with
# LoF enrichment = 0 centered directly on the y-axis while allowing full circles.
axis.spines["left"].set_position(("data", 0))
axis.spines["bottom"].set_position(("data", 0))
axis.spines["right"].set_visible(False)
axis.spines["top"].set_visible(False)

axis.set_xlabel(
    "LoF enrichment (observed / expected)",
    fontsize=14,
    fontweight="bold",
    labelpad=9,
)

axis.set_ylabel(
    "Dmis enrichment (observed / expected)",
    fontsize=14,
    fontweight="bold",
    labelpad=9,
)

axis.set_title(
    "LoF versus Dmis pathway enrichment",
    fontsize=17,
    fontweight="bold",
    pad=16,
)

axis.xaxis.set_major_locator(MaxNLocator(integer=True, prune=None))
axis.yaxis.set_major_locator(MaxNLocator(integer=True, prune=None))
axis.xaxis.set_major_formatter(FuncFormatter(format_tick))
axis.yaxis.set_major_formatter(FuncFormatter(format_tick))
# Keep the visual padding needed for complete circles while showing ticks from zero upward.
x_ticks = [tick for tick in axis.get_xticks() if tick >= 0 and tick <= axis_limit]
y_ticks = [tick for tick in axis.get_yticks() if tick >= 0 and tick <= axis_limit]
axis.set_xticks(x_ticks)
axis.set_yticks(y_ticks)
axis.tick_params(labelsize=12, width=1.1, length=5)

for spine in axis.spines.values():
    spine.set_linewidth(1.1)
    spine.set_color("#444444")

axis.grid(
    True,
    linestyle=":",
    linewidth=0.55,
    alpha=0.18,
    zorder=0,
)

# ------------------------------------------------------------------
# Publication-style cluster key with aligned panels and no overlap.
# ------------------------------------------------------------------

key_axis.text(
    0.02,
    0.985,
    "GO Ontology Clusters",
    fontsize=16.5,
    fontweight="bold",
    va="top",
)

cluster_groups = []
for cluster in sorted(data["Cluster"].unique()):
    subset = data[data["Cluster"].eq(cluster)].sort_values("Plot_Number")
    cluster_name = subset["Ontology_Cluster_Name"].iloc[0]
    cluster_groups.append((cluster, cluster_name, subset))

# Estimate required height for each box from the wrapped text.
group_line_counts = []
wrapped_group_content = {}

for cluster, cluster_name, subset in cluster_groups:
    cluster_title = f"Cluster {cluster}: {display_title(cluster_name)}"

    wrapped_title = "\n".join(
        textwrap.wrap(
            cluster_title,
            width=44,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )

    items = []
    total_lines = max(1, wrapped_title.count("\n") + 1)

    for _, row in subset.iterrows():
        wrapped_name = "\n".join(
            textwrap.wrap(
                display_title(readable_pathway_name(row["Pathway"])),
                width=39,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
        line_count = max(1, wrapped_name.count("\n") + 1)
        total_lines += line_count
        items.append((int(row["Plot_Number"]), wrapped_name, line_count))

    wrapped_group_content[cluster] = (wrapped_title, items)
    group_line_counts.append(total_lines)

available_height = 0.90
gap = 0.014
line_unit = (
    available_height - gap * (len(cluster_groups) - 1)
) / max(sum(group_line_counts) + 0.9 * len(cluster_groups), 1)

current_top = 0.93

for (cluster, cluster_name, subset), line_count_total in zip(
    cluster_groups,
    group_line_counts,
):
    color = cluster_color(cluster)
    wrapped_title, items = wrapped_group_content[cluster]

    box_height = line_unit * (line_count_total + 0.9)
    box_bottom = current_top - box_height

    panel = FancyBboxPatch(
        (0.015, box_bottom),
        0.965,
        box_height,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        linewidth=1.35,
        edgecolor=color,
        facecolor=color,
        alpha=KEY_BOX_ALPHA,
        transform=key_axis.transAxes,
        clip_on=False,
    )
    key_axis.add_patch(panel)

    title_y = current_top - 0.022

    key_axis.text(
        0.045,
        title_y,
        wrapped_title,
        transform=key_axis.transAxes,
        fontsize=KEY_HEADER_FONT_SIZE,
        fontweight="bold",
        color=color,
        va="top",
        linespacing=1.05,
    )

    title_lines = max(1, wrapped_title.count("\n") + 1)
    item_y = title_y - line_unit * title_lines - 0.006

    for number, wrapped_name, item_lines in items:
        key_axis.text(
            0.058,
            item_y,
            str(number),
            transform=key_axis.transAxes,
            fontsize=KEY_ITEM_FONT_SIZE - 0.2,
            fontweight="bold",
            color="white",
            ha="center",
            va="top",
            bbox={
                "boxstyle": "circle,pad=0.20",
                "facecolor": color,
                "edgecolor": "black",
                "linewidth": 0.75,
            },
        )

        key_axis.text(
            0.102,
            item_y,
            wrapped_name,
            transform=key_axis.transAxes,
            fontsize=KEY_ITEM_FONT_SIZE,
            color="#222222",
            va="top",
            linespacing=1.10,
        )

        item_y -= line_unit * item_lines

    current_top = box_bottom - gap

figure.suptitle(
    "Pathway enrichment colored by Gene Ontology hierarchy",
    fontsize=19,
    fontweight="bold",
    y=0.995,
)

figure.lines.append(
    plt.Line2D(
        [0.06, 0.96],
        [0.055, 0.055],
        transform=figure.transFigure,
        color="#9A9A9A",
        linewidth=0.8,
    )
)

figure.text(
    0.5,
    0.028,
    ("Points denote observed-to-expected pathway enrichment. "
     "Horizontal and vertical bars indicate exact 95% Poisson confidence intervals. "
     "The dashed diagonal represents equal LoF and Dmis enrichment."),
    ha="center",
    va="center",
    fontsize=10.8,
    color="#444444",
)
figure.subplots_adjust(
    left=0.07,
    right=0.985,
    top=0.91,
    bottom=0.11,
)

figure.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
)

figure.savefig(
    OUTPUT_PNG,
    dpi=PLOT_DPI,
    bbox_inches="tight",
)

plt.close(figure)

print(f"Saved ontology-clustered table: {OUTPUT_TABLE}")
print(f"Saved ontology-clustered scatter PDF: {OUTPUT_PDF}")
print(f"Saved ontology-clustered scatter PNG: {OUTPUT_PNG}")