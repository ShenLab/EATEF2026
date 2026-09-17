#!/usr/bin/env python3
from __future__ import annotations

import os
import numpy as np
import pandas as pd

FILES = {
    "LoF": {
        "Original": "results/pathway_enrichment_adjusted_lof.tsv",
        "SPARK": "results/control_background/pathway_enrichment_adjusted_lof_spark.tsv",
    },
    "Dmis": {
        "Original": "results/pathway_enrichment_adjusted_misfit.tsv",
        "SPARK": "results/control_background/pathway_enrichment_adjusted_misfit_spark.tsv",
    },
    "LoF_Dmis": {
        "Original": "results/pathway_enrichment_adjusted_lof_dmis.tsv",
        "SPARK": "results/control_background/pathway_enrichment_adjusted_lof_dmis_spark.tsv",
    },
}

DETAILED_OUTPUT = "results/control_background/pathway_model_vs_spark_comparison.tsv"
SUMMARY_OUTPUT = "results/control_background/pathway_model_vs_spark_significance_summary.tsv"

# Use 0.05 for per-analysis significance.
# Change to 0.025 if you want the manuscript's Bonferroni-adjusted
# selection threshold across Dmis and LoF+Dmis.
SIGNIFICANCE_THRESHOLD = 0.05


def load_result(path: str, analysis: str, background: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, sep="\t", low_memory=False)

    required = {
        "Pathway",
        "Observed",
        "Expected",
        "Enrichment",
        "P-Value",
        "FWER",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    if df["Pathway"].duplicated().any():
        raise ValueError(f"Duplicate pathways found in {path}")

    numeric = ["Observed", "Expected", "Enrichment", "P-Value", "FWER"]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[numeric].isna().any().any():
        raise ValueError(f"Invalid numeric values found in {path}")

    df["Analysis"] = analysis
    df["Background"] = background
    df["Significant"] = df["FWER"] <= SIGNIFICANCE_THRESHOLD

    return df[
        [
            "Pathway",
            "Analysis",
            "Background",
            "Observed",
            "Expected",
            "Enrichment",
            "P-Value",
            "FWER",
            "Significant",
        ]
    ]


detailed_tables = []
summary_tables = []

for analysis, paths in FILES.items():
    original = load_result(paths["Original"], analysis, "Original")
    spark = load_result(paths["SPARK"], analysis, "SPARK")

    original = original.drop(columns=["Analysis", "Background"]).rename(
        columns={
            "Observed": "Original_Observed",
            "Expected": "Original_Expected",
            "Enrichment": "Original_Enrichment",
            "P-Value": "Original_P_Value",
            "FWER": "Original_FWER",
            "Significant": "Original_Significant",
        }
    )

    spark = spark.drop(columns=["Analysis", "Background"]).rename(
        columns={
            "Observed": "SPARK_Observed",
            "Expected": "SPARK_Expected",
            "Enrichment": "SPARK_Enrichment",
            "P-Value": "SPARK_P_Value",
            "FWER": "SPARK_FWER",
            "Significant": "SPARK_Significant",
        }
    )

    comparison = original.merge(
        spark,
        on="Pathway",
        how="outer",
        validate="one_to_one",
    )
    comparison.insert(1, "Analysis", analysis)

    comparison["Expected_Difference_SPARK_minus_Original"] = (
        comparison["SPARK_Expected"] - comparison["Original_Expected"]
    )
    comparison["Enrichment_Difference_SPARK_minus_Original"] = (
        comparison["SPARK_Enrichment"] - comparison["Original_Enrichment"]
    )
    comparison["FWER_Difference_SPARK_minus_Original"] = (
        comparison["SPARK_FWER"] - comparison["Original_FWER"]
    )

    comparison["Significance_Change"] = np.select(
        [
            comparison["Original_Significant"].eq(True)
            & comparison["SPARK_Significant"].eq(True),

            comparison["Original_Significant"].eq(True)
            & comparison["SPARK_Significant"].eq(False),

            comparison["Original_Significant"].eq(False)
            & comparison["SPARK_Significant"].eq(True),

            comparison["Original_Significant"].eq(False)
            & comparison["SPARK_Significant"].eq(False),
        ],
        [
            "Significant in both",
            "Lost significance with SPARK",
            "Gained significance with SPARK",
            "Not significant in either",
        ],
        default="Not tested in one background",
    )

    detailed_tables.append(comparison)

    summary = comparison[
        [
            "Pathway",
            "Original_Significant",
            "SPARK_Significant",
            "Significance_Change",
        ]
    ].rename(
        columns={
            "Original_Significant": f"Original_{analysis}_Significant",
            "SPARK_Significant": f"SPARK_{analysis}_Significant",
            "Significance_Change": f"{analysis}_Significance_Change",
        }
    )
    summary_tables.append(summary)


detailed = pd.concat(detailed_tables, ignore_index=True)
detailed = detailed.sort_values(
    ["Analysis", "Original_FWER", "SPARK_FWER", "Pathway"],
    na_position="last",
).reset_index(drop=True)

summary = summary_tables[0]
for table in summary_tables[1:]:
    summary = summary.merge(table, on="Pathway", how="outer", validate="one_to_one")

for column in summary.columns:
    if column.endswith("_Significant"):
        summary[column] = summary[column].map(
            {True: "Yes", False: "No"}
        ).fillna("Not tested")

summary["Any_Original_Significant"] = summary[
    [
        "Original_LoF_Significant",
        "Original_Dmis_Significant",
        "Original_LoF_Dmis_Significant",
    ]
].eq("Yes").any(axis=1)

summary["Any_SPARK_Significant"] = summary[
    [
        "SPARK_LoF_Significant",
        "SPARK_Dmis_Significant",
        "SPARK_LoF_Dmis_Significant",
    ]
].eq("Yes").any(axis=1)

summary["Overall_Sensitivity_Result"] = np.select(
    [
        summary["Any_Original_Significant"]
        & summary["Any_SPARK_Significant"],

        summary["Any_Original_Significant"]
        & ~summary["Any_SPARK_Significant"],

        ~summary["Any_Original_Significant"]
        & summary["Any_SPARK_Significant"],
    ],
    [
        "Supported by both backgrounds",
        "Original-only signal",
        "SPARK-only signal",
    ],
    default="Not significant under either background",
)

summary = summary.sort_values(
    ["Any_Original_Significant", "Any_SPARK_Significant", "Pathway"],
    ascending=[False, False, True],
).reset_index(drop=True)

for path in [DETAILED_OUTPUT, SUMMARY_OUTPUT]:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

detailed.to_csv(DETAILED_OUTPUT, sep="\t", index=False)
summary.to_csv(SUMMARY_OUTPUT, sep="\t", index=False)

print(f"Saved: {DETAILED_OUTPUT}")
print(f"Saved: {SUMMARY_OUTPUT}")