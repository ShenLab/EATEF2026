#!/usr/bin/env python3

import os
import re

import numpy as np
import pandas as pd
from scipy.stats import poisson

# ==========================================================
# Files
# ==========================================================

CASE_FILE = "results/EA_TEF_cases_annotated.tsv"

MUTATION_RATE_FILE = "results/control_background/SPARK_gene_DNV_rates.tsv"

BLACKLIST_FILE = "data/09GENCODEV19_blacklist.txt"

# The exact same pathway file must be used by the LoF,
# Dmis, and LoF+Dmis analyses.
COMMON_PATHWAY_FILE = "results/control_background/common_pathway_universe_spark.tsv"

OUTPUT_FILE = "results/control_background/pathway_enrichment_lof_dmis_spark.tsv"

# =========================================================
# Parameters
# ==========================================================

N_CASES = 161

SCORE = "MisFit_S"
THRESHOLD = 0.005

LOF_COLUMN = "Control_LoF_Rate"
DMIS_COLUMN = "Control_Dmis_Rate"

CASE_GENE_COLUMN = "HGNC"
MUTATION_RATE_GENE_COLUMN = "Gene"
PHENOTYPE_COLUMN = "Syndormic or not"

LOF_VARIANT_TYPES = {
  "lof",
  "frameshift",
  "stop_gained",
  "splice_acceptor",
  "splice_donor",
  "start_lost",
}

MISSENSE_VARIANT_TYPES = {
  "mis",
  "missense",
}

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

print("Loading EA/TEF variants...")

cases = pd.read_csv(
  CASE_FILE,
  sep="\t",
  low_memory=False,
  encoding="latin1",
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

required_case_columns = {
  "Sample_id",
  "Var_type",
  CASE_GENE_COLUMN,
  SCORE,
  PHENOTYPE_COLUMN,
}

missing_case_columns = (
  required_case_columns
  - set(cases.columns)
)

if missing_case_columns:
  raise ValueError(
      "Missing required case columns: "
      f"{sorted(missing_case_columns)}"
  )

required_rate_columns = {
  MUTATION_RATE_GENE_COLUMN,
  LOF_COLUMN,
  DMIS_COLUMN,
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
  "GenesInPathway",
  "GenesWithControlRate",
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
# Keep complex cases
# ==========================================================

cases["Sample_id"] = (
  cases["Sample_id"]
  .astype(str)
  .str.strip()
)

cases[PHENOTYPE_COLUMN] = (
  cases[PHENOTYPE_COLUMN]
  .astype("string")
  .str.strip()
  .str.lower()
)

missing_phenotype_labels = int(
  cases[PHENOTYPE_COLUMN].isna().sum()
  + cases[PHENOTYPE_COLUMN].eq("").sum()
)

print(
  "Variants without a phenotype label: "
  f"{missing_phenotype_labels}"
)

cases = cases[
  cases[PHENOTYPE_COLUMN].eq("complex")
].copy()

print(
  "Unique complex samples in annotated file: "
  f"{cases['Sample_id'].nunique()}"
)

print(
  "Annotated variants from complex cases: "
  f"{len(cases)}"
)
# ==========================================================
# Prepare observed LoF + Dmis variants
# ==========================================================

cases[SCORE] = pd.to_numeric(
  cases[SCORE],
  errors="coerce",
)

# MisFit_S = -1 represents a missing score.
cases.loc[
  cases[SCORE] == -1,
  SCORE,
] = np.nan

cases["Var_type"] = (
  cases["Var_type"]
  .astype(str)
  .str.strip()
  .str.lower()
)

lof_mask = cases["Var_type"].isin(
  LOF_VARIANT_TYPES
)

dmis_mask = (
  cases["Var_type"].isin(
      MISSENSE_VARIANT_TYPES
  )
  & cases[SCORE].notna()
  & cases[SCORE].ge(THRESHOLD)
)

cases = cases[
  lof_mask | dmis_mask
].copy()

cases = cases.dropna(
  subset=[CASE_GENE_COLUMN]
).copy()

cases[CASE_GENE_COLUMN] = (
  cases[CASE_GENE_COLUMN]
  .astype(str)
  .str.strip()
  .str.upper()
)

cases = cases[
  cases[CASE_GENE_COLUMN] != ""
].copy()

cases_before_blacklist = len(cases)

cases = cases[
  ~cases[CASE_GENE_COLUMN].isin(
      blacklisted_genes
  )
].copy()

print(
  "Observed LoF+Dmis variant rows removed by blacklist: "
  f"{cases_before_blacklist - len(cases)}"
)

observed_lof = int(
  cases["Var_type"]
  .isin(LOF_VARIANT_TYPES)
  .sum()
)

observed_dmis = int(
  cases["Var_type"]
  .isin(MISSENSE_VARIANT_TYPES)
  .sum()
)

print(
  "Observed LoF + Dmis variants after blacklist: "
  f"{len(cases)}"
)

print(
  "Observed LoF variants after blacklist: "
  f"{observed_lof}"
)

print(
  "Observed MisFit-qualified missense variants "
  "after blacklist: "
  f"{observed_dmis}"
)

# ==========================================================
# Prepare background mutation rates
# ==========================================================

mutation_rate[LOF_COLUMN] = pd.to_numeric(
  mutation_rate[LOF_COLUMN],
  errors="coerce",
).fillna(0)

mutation_rate[DMIS_COLUMN] = pd.to_numeric(
  mutation_rate[DMIS_COLUMN],
  errors="coerce",
).fillna(0)

mutation_rate = mutation_rate.dropna(
  subset=[MUTATION_RATE_GENE_COLUMN]
).copy()

mutation_rate[MUTATION_RATE_GENE_COLUMN] = (
  mutation_rate[MUTATION_RATE_GENE_COLUMN]
  .astype(str)
  .str.strip()
  .str.upper()
)

mutation_rate = mutation_rate[
  mutation_rate[MUTATION_RATE_GENE_COLUMN] != ""
].copy()

mutation_rate_before_blacklist = len(
  mutation_rate
)

mutation_rate = mutation_rate[
  ~mutation_rate[MUTATION_RATE_GENE_COLUMN].isin(
      blacklisted_genes
  )
].copy()

print(
  "SPARK control-rate rows removed by blacklist: "
  f"{mutation_rate_before_blacklist - len(mutation_rate)}"
)

mutation_rate = (
  mutation_rate
  .groupby(
      MUTATION_RATE_GENE_COLUMN,
      as_index=False,
  )[
      [
          LOF_COLUMN,
          DMIS_COLUMN,
      ]
  ]
  .sum()
)

rate_lookup = mutation_rate.set_index(
  MUTATION_RATE_GENE_COLUMN
)

rate_gene_set = set(rate_lookup.index)

# ==========================================================
# Pathway enrichment analysis
# ==========================================================

results = []

for _, pathway_row in common_pathways.iterrows():

  pathway = str(pathway_row["Pathway"])

  gene_set = {
      gene.strip().upper()
      for gene in str(pathway_row["Genes"]).split(";")
      if gene.strip()
      and gene.strip().upper() not in blacklisted_genes
  }

  pathway_genes_with_rates = sorted(
      gene_set & rate_gene_set
  )

  if not pathway_genes_with_rates:
      raise ValueError(
          f"{pathway} has no genes with SPARK control rates."
      )

  pathway_rates = rate_lookup.loc[
      pathway_genes_with_rates
  ]

  lof_rate_sum = float(
      pathway_rates[LOF_COLUMN].sum()
  )

  dmis_rate_sum = float(
      pathway_rates[DMIS_COLUMN].sum()
  )

  lof_expected = (
      N_CASES
      * lof_rate_sum
  )

  dmis_expected = (
      N_CASES
      * dmis_rate_sum
  )

  expected = (
      lof_expected
      + dmis_expected
  )

  if expected <= 0:
      raise ValueError(
          f"{pathway} has nonpositive combined expected burden: "
          f"{expected}"
      )

  pathway_cases = cases[
      cases[CASE_GENE_COLUMN].isin(gene_set)
  ]

  observed = int(
      len(pathway_cases)
  )

  lof_observed = int(
      pathway_cases["Var_type"]
      .isin(LOF_VARIANT_TYPES)
      .sum()
  )

  dmis_observed = int(
      pathway_cases["Var_type"]
      .isin(MISSENSE_VARIANT_TYPES)
      .sum()
  )

  enrichment = observed / expected

  lof_enrichment = (
      lof_observed / lof_expected
      if lof_expected > 0
      else np.nan
  )

  dmis_enrichment = (
      dmis_observed / dmis_expected
      if dmis_expected > 0
      else np.nan
  )

  pvalue = float(
      poisson.sf(
          observed - 1,
          expected,
      )
  )

  lof_pvalue = (
      float(
          poisson.sf(
              lof_observed - 1,
              lof_expected,
          )
      )
      if lof_expected > 0
      else np.nan
  )

  dmis_pvalue = (
      float(
          poisson.sf(
              dmis_observed - 1,
              dmis_expected,
          )
      )
      if dmis_expected > 0
      else np.nan
  )

  results.append(
      {
          "Pathway": pathway,
          "GenesInPathway": int(
              pathway_row["GenesInPathway"]
          ),
          "GenesWithControlRate": len(
              pathway_genes_with_rates
          ),
          "Observed": observed,
          "Expected": expected,
          "Enrichment": enrichment,
          "P-Value": pvalue,
          "LoF_Observed": lof_observed,
          "LoF_Expected": lof_expected,
          "LoF_Enrichment": lof_enrichment,
          "LoF_P-Value": lof_pvalue,
          "Dmis_Observed": dmis_observed,
          "Dmis_Expected": dmis_expected,
          "Dmis_Enrichment": dmis_enrichment,
          "Dmis_P-Value": dmis_pvalue,
          "Background": "SPARK_unaffected_siblings",
      }
  )

# ==========================================================
# Save results
# ==========================================================

results = pd.DataFrame(results)

if results.empty:
  raise ValueError(
      "No pathways were tested."
  )

if len(results) != len(common_pathways):
  raise ValueError(
      "The number of tested LoF+Dmis pathways does not "
      "match the common pathway universe."
  )

results = results.sort_values(
  "P-Value"
).reset_index(drop=True)

output_directory = os.path.dirname(
  OUTPUT_FILE
)

if output_directory:
  os.makedirs(
      output_directory,
      exist_ok=True,
  )

results.to_csv(
  OUTPUT_FILE,
  sep="\t",
  index=False,
)

print()
print(results.head(20).to_string(index=False))
print()

print(
  f"Pathways tested: {len(results)}"
)

print(
  f"Saved to {OUTPUT_FILE}"
)