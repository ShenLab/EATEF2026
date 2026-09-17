#!/usr/bin/env python3


import os


import numpy as np
import pandas as pd




# ==========================================================
# Files
# ==========================================================


REAL_RESULTS_FILE = (
   "results/control_background/pathway_enrichment_lof_spark.tsv"
)


PERMUTATION_FILE = (
   "results/control_background/permutation_best_pvalues_lof_spark.tsv"
)


COMMON_PATHWAY_FILE = "results/control_background/common_pathway_universe_spark.tsv"


OUTPUT_FILE = (
   "results/control_background/pathway_enrichment_adjusted_lof_spark.tsv"
)




# ==========================================================
# Load results
# ==========================================================


print("Loading real pathway results...")


real_results = pd.read_csv(
   REAL_RESULTS_FILE,
   sep="\t",
)


print("Loading permutation results...")


permutation_results = pd.read_csv(
   PERMUTATION_FILE,
   sep="\t",
)


print("Loading common pathway universe...")


common_pathways = pd.read_csv(
   COMMON_PATHWAY_FILE,
   sep="\t",
)




# ==========================================================
# Validate columns and pathway universe
# ==========================================================


if "Pathway" not in real_results.columns:
   raise ValueError(
       f"'Pathway' was not found in "
       f"{REAL_RESULTS_FILE}"
   )


if "P-Value" not in real_results.columns:
   raise ValueError(
       f"'P-Value' was not found in "
       f"{REAL_RESULTS_FILE}"
   )


if "BestP" not in permutation_results.columns:
   raise ValueError(
       f"'BestP' was not found in "
       f"{PERMUTATION_FILE}"
   )


if "Pathway" not in common_pathways.columns:
   raise ValueError(
       f"'Pathway' was not found in "
       f"{COMMON_PATHWAY_FILE}"
   )


real_pathways = set(
   real_results["Pathway"]
   .dropna()
   .astype(str)
)


common_pathway_set = set(
   common_pathways["Pathway"]
   .dropna()
   .astype(str)
)


if real_pathways != common_pathway_set:
   missing_from_real = sorted(
       common_pathway_set - real_pathways
   )
   extra_in_real = sorted(
       real_pathways - common_pathway_set
   )


   raise ValueError(
       "The real LoF results do not match the common "
       "pathway universe.\n"
       f"Missing from real results: {missing_from_real[:10]}\n"
       f"Extra in real results: {extra_in_real[:10]}"
   )




# ==========================================================
# Convert p-values to numeric
# ==========================================================


real_results["P-Value"] = pd.to_numeric(
   real_results["P-Value"],
   errors="coerce",
)


permutation_results["BestP"] = pd.to_numeric(
   permutation_results["BestP"],
   errors="coerce",
)


if real_results["P-Value"].isna().any():
   invalid_count = int(
       real_results["P-Value"]
       .isna()
       .sum()
   )


   raise ValueError(
       f"{invalid_count} real pathways have invalid p-values."
   )


permutation_results = permutation_results.dropna(
   subset=["BestP"]
).copy()


permutation_best_pvalues = (
   permutation_results["BestP"]
   .to_numpy(dtype=float)
)


number_of_permutations = len(
   permutation_best_pvalues
)


if number_of_permutations == 0:
   raise ValueError(
       "No valid permutation p-values were found."
   )


print(
   f"Usable permutations: "
   f"{number_of_permutations:,}"
)




# ==========================================================
# Calculate empirical FWER
# ==========================================================


sorted_best_pvalues = np.sort(
   permutation_best_pvalues
)


raw_pvalues = real_results[
   "P-Value"
].to_numpy(dtype=float)


counts = np.searchsorted(
   sorted_best_pvalues,
   raw_pvalues,
   side="right",
)


# Preserve the requested no-offset calculation.
fwer = (
   counts
   / number_of_permutations
)


real_results[
   "PermutationsBestP_LE_RawP"
] = counts.astype(int)


real_results["FWER"] = fwer




# ==========================================================
# Sort and save
# ==========================================================


real_results = real_results.sort_values(
   [
       "FWER",
       "P-Value",
   ]
).reset_index(drop=True)


output_directory = os.path.dirname(
   OUTPUT_FILE
)


if output_directory:
   os.makedirs(
       output_directory,
       exist_ok=True,
   )


real_results.to_csv(
   OUTPUT_FILE,
   sep="\t",
   index=False,
)


print()
print(
   real_results[
       [
           "Pathway",
           "Observed",
           "Expected",
           "Enrichment",
           "P-Value",
           "PermutationsBestP_LE_RawP",
           "FWER",
       ]
   ].head(20).to_string(index=False)
)
print()


print(
   f"Adjusted pathways: {len(real_results)}"
)


print(
   f"Saved to {OUTPUT_FILE}"
)