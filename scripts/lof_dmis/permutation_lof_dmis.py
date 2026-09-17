#!/usr/bin/env python3


import os
import re


import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import poisson




# ==========================================================
# Files
# ==========================================================


MUTATION_RATE_FILE = "data/gene_mutation_rate.new.txt"


BLACKLIST_FILE = "data/09GENCODEV19_blacklist.txt"


COMMON_PATHWAY_FILE = "results/common_pathway_universe.tsv"


OUTPUT_FILE = (
   "results/permutation_best_pvalues_lof_dmis.tsv"
)

SCALE_FACTOR_FILE = (
   "results/synonymous_scale_factor_lof_dmis.tsv"
)




# ==========================================================
# Parameters
# ==========================================================


N_CASES = 161


GENE_COLUMN = "Gene_name"
LOF_COLUMN = "LoF"
DMIS_COLUMN = "Dmis_MisFit_S_0.005"
SYNONYMOUS_COLUMN = "Synonymous"


N_PERMUTATIONS = 1_000_000
RANDOM_SEED = 42
PROGRESS_INTERVAL = 100




# ==========================================================
# Helper functions
# ==========================================================


def load_blacklisted_genes(path):
   """
   Load blacklisted gene symbols from an irregular text file.


   Blank lines and comment lines are ignored. The first
   whitespace-, tab-, or comma-separated field is treated
   as the gene symbol.
   """


   if not os.path.exists(path):
       raise FileNotFoundError(
           f"Blacklist file was not found: {path}"
       )


   blacklisted_genes = set()


   with open(
       path,
       "r",
       encoding="utf-8",
       errors="replace",
   ) as handle:
       for raw_line in handle:
           line = raw_line.strip()


           if not line or line.startswith("#"):
               continue


           gene = re.split(
               r"[\t, ]+",
               line,
               maxsplit=1,
           )[0].strip()


           if gene:
               blacklisted_genes.add(gene.upper())


   if not blacklisted_genes:
       raise ValueError(
           f"No genes were loaded from {path}"
       )


   return blacklisted_genes




# ==========================================================
# Load data
# ==========================================================


print("Loading blacklist...")


blacklisted_genes = load_blacklisted_genes(
   BLACKLIST_FILE
)


print(
   f"Blacklisted genes loaded: {len(blacklisted_genes)}"
)




print("Loading gene mutation rates...")


mutation_rate = pd.read_csv(
   MUTATION_RATE_FILE,
   sep="\t",
   low_memory=False,
)




print("Loading common pathway universe...")


common_pathways = pd.read_csv(
   COMMON_PATHWAY_FILE,
   sep="\t",
   low_memory=False,
)




# ==========================================================
# Validate required columns
# ==========================================================


required_rate_columns = {
   GENE_COLUMN,
   LOF_COLUMN,
   DMIS_COLUMN,
   SYNONYMOUS_COLUMN,
}


missing_rate_columns = (
   required_rate_columns
   - set(mutation_rate.columns)
)


if missing_rate_columns:
   raise ValueError(
       "Missing required mutation-rate columns: "
       f"{sorted(missing_rate_columns)}"
   )




required_pathway_columns = {
   "Pathway",
   "Genes",
}


missing_pathway_columns = (
   required_pathway_columns
   - set(common_pathways.columns)
)


if missing_pathway_columns:
   raise ValueError(
       "Missing required common-pathway columns: "
       f"{sorted(missing_pathway_columns)}"
   )




if common_pathways["Pathway"].duplicated().any():
   raise ValueError(
       "The common pathway universe contains duplicate pathways."
   )




# ==========================================================
# Clean mutation-rate data
# ==========================================================


mutation_rate[LOF_COLUMN] = pd.to_numeric(
   mutation_rate[LOF_COLUMN],
   errors="coerce",
).fillna(0)


mutation_rate[DMIS_COLUMN] = pd.to_numeric(
   mutation_rate[DMIS_COLUMN],
   errors="coerce",
).fillna(0)

mutation_rate[SYNONYMOUS_COLUMN] = pd.to_numeric(
   mutation_rate[SYNONYMOUS_COLUMN],
   errors="coerce",
).fillna(0)


mutation_rate = mutation_rate.dropna(
   subset=[GENE_COLUMN]
).copy()


mutation_rate[GENE_COLUMN] = (
   mutation_rate[GENE_COLUMN]
   .astype(str)
   .str.strip()
   .str.upper()
)


mutation_rate = mutation_rate[
   mutation_rate[GENE_COLUMN] != ""
].copy()


mutation_rate_before_blacklist = len(
   mutation_rate
)


mutation_rate = mutation_rate[
   ~mutation_rate[GENE_COLUMN].isin(
       blacklisted_genes
   )
].copy()


print(
   "Mutation-rate rows removed by blacklist: "
   f"{mutation_rate_before_blacklist - len(mutation_rate)}"
)


gene_rates = (
   mutation_rate
   .groupby(
       GENE_COLUMN,
       as_index=False,
   )[
       [
           LOF_COLUMN,
           DMIS_COLUMN,
           SYNONYMOUS_COLUMN,
       ]
   ]
   .sum()
)


if not os.path.exists(
   SCALE_FACTOR_FILE
):
   raise FileNotFoundError(
       "Synonymous scale-factor file was not found. Run the "
       "matching LoF+Dmis enrichment script first: "
       f"{SCALE_FACTOR_FILE}"
   )

scale_data = pd.read_csv(
   SCALE_FACTOR_FILE,
   sep="\t",
)

required_scale_columns = {
   "ObservedSynonymous",
   "ExpectedSynonymousUnscaled",
   "ScaleFactor",
   "N_CASES",
}

missing_scale_columns = (
   required_scale_columns
   - set(scale_data.columns)
)

if missing_scale_columns:
   raise ValueError(
       "The synonymous scale-factor file is missing columns: "
       f"{sorted(missing_scale_columns)}"
   )

if len(scale_data) != 1:
   raise ValueError(
       "The synonymous scale-factor file must contain exactly one row."
   )

scale_n_cases = int(
   scale_data.loc[0, "N_CASES"]
)

if scale_n_cases != N_CASES:
   raise ValueError(
       "N_CASES does not match between enrichment and permutation. "
       f"Scale file: {scale_n_cases}; permutation: {N_CASES}"
   )

observed_synonymous = int(
   scale_data.loc[0, "ObservedSynonymous"]
)

expected_synonymous_unscaled = float(
   scale_data.loc[0, "ExpectedSynonymousUnscaled"]
)

scale_factor = float(
   scale_data.loc[0, "ScaleFactor"]
)

if scale_factor <= 0:
   raise ValueError(
       f"Invalid synonymous scale factor: {scale_factor}"
   )

current_expected_synonymous = float(
   2
   * N_CASES
   * gene_rates[SYNONYMOUS_COLUMN].sum()
)

if not np.isclose(
   current_expected_synonymous,
   expected_synonymous_unscaled,
   rtol=1e-10,
   atol=1e-12,
):
   raise ValueError(
       "The mutation-rate table or blacklist differs from the one "
       "used by the enrichment script. "
       f"Enrichment expectation: {expected_synonymous_unscaled}; "
       f"current expectation: {current_expected_synonymous}"
   )

print(
   "Observed synonymous variants used for calibration: "
   f"{observed_synonymous}"
)

print(
   "Unscaled expected synonymous variants: "
   f"{expected_synonymous_unscaled:.6g}"
)

print(
   "Synonymous calibration scale factor: "
   f"{scale_factor:.6g}"
)

genes = gene_rates[
   GENE_COLUMN
].to_numpy()


lof_rates = gene_rates[
   LOF_COLUMN
].to_numpy(
   dtype=np.float64
)


dmis_rates = gene_rates[
   DMIS_COLUMN
].to_numpy(
   dtype=np.float64
)


if (
   lof_rates.sum()
   + dmis_rates.sum()
   <= 0
):
   raise ValueError(
       "The total LoF+Dmis mutation rate is zero."
   )


lof_gene_expected = (
   2
   * N_CASES
   * lof_rates
   * scale_factor
)


dmis_gene_expected = (
   2
   * N_CASES
   * dmis_rates
   * scale_factor
)


combined_gene_expected = (
   lof_gene_expected
   + dmis_gene_expected
)


print(
   f"Genes with mutation-rate rows: {len(genes)}"
)


print(
   "Expected total LoF variants: "
   f"{lof_gene_expected.sum():.6g}"
)


print(
   "Expected total Dmis variants: "
   f"{dmis_gene_expected.sum():.6g}"
)


print(
   "Expected total LoF+Dmis variants: "
   f"{combined_gene_expected.sum():.6g}"
)




# ==========================================================
# Build pathway-by-gene matrix from shared universe
# ==========================================================


gene_to_index = {
   gene: index
   for index, gene in enumerate(genes)
}


pathway_names = []
pathway_expected = []
pathway_gene_counts = []


matrix_rows = []
matrix_columns = []
matrix_values = []


for _, pathway_row in common_pathways.iterrows():


   pathway_name = str(
       pathway_row["Pathway"]
   )


   pathway_genes = {
       gene.strip().upper()
       for gene in str(pathway_row["Genes"]).split(";")
       if gene.strip()
       and gene.strip().upper() not in blacklisted_genes
   }


   indices = [
       gene_to_index[gene]
       for gene in pathway_genes
       if gene in gene_to_index
   ]


   if not indices:
       raise ValueError(
           f"{pathway_name} has no genes with mutation rates."
       )


   indices = np.asarray(
       sorted(set(indices)),
       dtype=np.int32,
   )


   expected = float(
       combined_gene_expected[
           indices
       ].sum()
   )


   if expected <= 0:
       raise ValueError(
           f"{pathway_name} has nonpositive combined "
           f"expected burden: {expected}"
       )


   pathway_index = len(
       pathway_names
   )


   pathway_names.append(
       pathway_name
   )


   pathway_expected.append(
       expected
   )


   pathway_gene_counts.append(
       len(indices)
   )


   matrix_rows.extend(
       [pathway_index] * len(indices)
   )


   matrix_columns.extend(
       indices.tolist()
   )


   matrix_values.extend(
       [1] * len(indices)
   )




pathway_matrix = csr_matrix(
   (
       np.asarray(
           matrix_values,
           dtype=np.int8,
       ),
       (
           np.asarray(
               matrix_rows,
               dtype=np.int32,
           ),
           np.asarray(
               matrix_columns,
               dtype=np.int32,
           ),
       ),
   ),
   shape=(
       len(pathway_names),
       len(genes),
   ),
)


pathway_names = np.asarray(
   pathway_names,
   dtype=object,
)


pathway_expected = np.asarray(
   pathway_expected,
   dtype=np.float64,
)


pathway_gene_counts = np.asarray(
   pathway_gene_counts,
   dtype=np.int32,
)


if len(pathway_names) != len(common_pathways):
   raise ValueError(
       "The LoF+Dmis permutation pathway count does not "
       "match the common pathway universe."
   )


print(
   "Pathways tested in every permutation: "
   f"{len(pathway_names)}"
)


print(
   f"Pathway matrix shape: {pathway_matrix.shape}"
)




# ==========================================================
# Allocate permutation outputs
# ==========================================================


rng = np.random.default_rng(
   RANDOM_SEED
)


permutation_numbers = np.arange(
   1,
   N_PERMUTATIONS + 1,
   dtype=np.int32,
)


best_pvalues = np.empty(
   N_PERMUTATIONS,
   dtype=np.float64,
)


best_pathways = np.empty(
   N_PERMUTATIONS,
   dtype=object,
)


best_observed_values = np.empty(
   N_PERMUTATIONS,
   dtype=np.int32,
)


best_expected_values = np.empty(
   N_PERMUTATIONS,
   dtype=np.float64,
)


best_enrichment_values = np.empty(
   N_PERMUTATIONS,
   dtype=np.float64,
)


best_gene_counts = np.empty(
   N_PERMUTATIONS,
   dtype=np.int32,
)


simulated_lof_totals = np.empty(
   N_PERMUTATIONS,
   dtype=np.int32,
)


simulated_dmis_totals = np.empty(
   N_PERMUTATIONS,
   dtype=np.int32,
)


simulated_combined_totals = np.empty(
   N_PERMUTATIONS,
   dtype=np.int32,
)




# ==========================================================
# Run permutations
# ==========================================================


print(
   f"Running {N_PERMUTATIONS:,} permutations..."
)


for permutation_index in range(
   N_PERMUTATIONS
):


   # Simulate LoF and Dmis independently using their own
   # gene-level expected mutation burdens, then combine them.
   simulated_lof_counts = rng.poisson(
       lof_gene_expected
   )


   simulated_dmis_counts = rng.poisson(
       dmis_gene_expected
   )


   simulated_gene_counts = (
       simulated_lof_counts
       + simulated_dmis_counts
   )


   simulated_lof_total = int(
       simulated_lof_counts.sum()
   )


   simulated_dmis_total = int(
       simulated_dmis_counts.sum()
   )


   simulated_combined_total = (
       simulated_lof_total
       + simulated_dmis_total
   )


   simulated_lof_totals[
       permutation_index
   ] = simulated_lof_total


   simulated_dmis_totals[
       permutation_index
   ] = simulated_dmis_total


   simulated_combined_totals[
       permutation_index
   ] = simulated_combined_total


   observed_values = np.asarray(
       pathway_matrix.dot(
           simulated_gene_counts
       )
   ).ravel()


   pvalues = poisson.sf(
       observed_values - 1,
       pathway_expected,
   )


   best_index = int(
       np.argmin(pvalues)
   )


   best_observed = int(
       observed_values[best_index]
   )


   best_expected = float(
       pathway_expected[best_index]
   )


   best_pvalue = float(
       pvalues[best_index]
   )


   best_enrichment = (
       best_observed
       / best_expected
   )


   best_pvalues[
       permutation_index
   ] = best_pvalue


   best_pathways[
       permutation_index
   ] = pathway_names[best_index]


   best_observed_values[
       permutation_index
   ] = best_observed


   best_expected_values[
       permutation_index
   ] = best_expected


   best_enrichment_values[
       permutation_index
   ] = best_enrichment


   best_gene_counts[
       permutation_index
   ] = pathway_gene_counts[best_index]


   permutation_number = (
       permutation_index + 1
   )


   if (
       permutation_number == 1
       or permutation_number % PROGRESS_INTERVAL == 0
       or permutation_number == N_PERMUTATIONS
   ):
       print(
           f"Completed permutation "
           f"{permutation_number:,} / "
           f"{N_PERMUTATIONS:,} "
           f"(LoF total: {simulated_lof_total}, "
           f"Dmis total: {simulated_dmis_total}, "
           f"combined: {simulated_combined_total})"
       )




# ==========================================================
# Save results
# ==========================================================


permutation_results = pd.DataFrame(
   {
       "Permutation": permutation_numbers,
       "BestP": best_pvalues,
       "BestPathway": best_pathways,
       "Observed": best_observed_values,
       "Expected": best_expected_values,
       "Enrichment": best_enrichment_values,
       "GenesWithMutationRate": best_gene_counts,
       "SimulatedLoFTotalVariants": simulated_lof_totals,
       "SimulatedDmisTotalVariants": simulated_dmis_totals,
       "SimulatedTotalVariants": simulated_combined_totals,
   }
)


output_directory = os.path.dirname(
   OUTPUT_FILE
)


if output_directory:
   os.makedirs(
       output_directory,
       exist_ok=True,
   )


permutation_results.to_csv(
   OUTPUT_FILE,
   sep="\t",
   index=False,
)


print()
print(
   permutation_results.head().to_string(
       index=False
   )
)
print()


print(
   permutation_results["BestP"].describe()
)
print()


print(
   "Simulated total LoF+Dmis variants per permutation:"
)


print(
   permutation_results[
       "SimulatedTotalVariants"
   ].describe()
)
print()


print(
   f"Saved to {OUTPUT_FILE}"
)