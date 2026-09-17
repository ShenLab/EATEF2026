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
MUTATION_RATE_FILE = "results/control_background/SPARK_gene_DNV_rates.tsv"

BLACKLIST_FILE = "data/09GENCODEV19_blacklist.txt"

# Must be the same pathway universe used by the LoF
# enrichment, Dmis enrichment, and LoF+Dmis analyses.
COMMON_PATHWAY_FILE = "results/control_background/common_pathway_universe_spark.tsv"

OUTPUT_FILE = "results/control_background/permutation_best_pvalues_lof_spark.tsv"

# ==========================================================
# Parameters
# ==========================================================

N_CASES = 161

GENE_COLUMN = "Gene"
LOF_COLUMN = "Control_LoF_Rate"

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

print("Loading SPARK empirical control rates...")

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
}

missing_rate_columns = (
   required_rate_columns
   - set(mutation_rate.columns)
)

if missing_rate_columns:
   raise ValueError(
       "Missing required SPARK control-rate columns: "
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
   "SPARK control-rate rows removed by blacklist: "
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
       ]
   ]
   .sum()
)

genes = gene_rates[
   GENE_COLUMN
].to_numpy()

rates = gene_rates[
   LOF_COLUMN
].to_numpy(
   dtype=np.float64
)

if rates.sum() <= 0:
   raise ValueError(
       "The total LoF mutation rate is zero."
   )

gene_expected = (
   N_CASES
   * rates
)

print(
   f"Genes with SPARK control-rate rows: {len(genes)}"
)

print(
   "Expected total LoF variants: "
   f"{gene_expected.sum():.6g}"
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
           f"{pathway_name} has no genes with SPARK LoF control rates."
       )

   # Remove any accidental duplicate indices and use a stable order.
   indices = np.asarray(
       sorted(set(indices)),
       dtype=np.int32,
   )

   expected = float(
       gene_expected[indices].sum()
   )

   # Every pathway in the shared universe must be tested,
   # provided its expected LoF burden is positive.
   if expected <= 0:
       raise ValueError(
           f"{pathway_name} has nonpositive LoF expected burden: "
           f"{expected}"
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
       "The LoF permutation pathway count does not match "
       "the common pathway universe."
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

simulated_totals = np.empty(
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
   simulated_gene_counts = rng.poisson(
       gene_expected
   )

   simulated_total = int(
       simulated_gene_counts.sum()
   )

   simulated_totals[
       permutation_index
   ] = simulated_total

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
           f"(simulated total: {simulated_total})"
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
       "GenesWithControlRate": best_gene_counts,
       "SimulatedTotalVariants": simulated_totals,
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
   "Simulated total LoF variants per permutation:"
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