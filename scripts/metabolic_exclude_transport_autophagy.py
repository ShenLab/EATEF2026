
#!/usr/bin/env python3


"""
Test whether selected metabolic GO pathways remain enriched for de novo
predicted-damaging missense variants after excluding every gene belonging to:


   GOBP_INTRACELLULAR_TRANSPORT
   GOBP_REGULATION_OF_AUTOPHAGY


Outputs
-------
results/metabolic_enrichment_excluding_transport_autophagy.tsv
results/metabolic_overlap_excluded_genes.tsv


This is a post-analysis sensitivity test. It recalculates raw Poisson p-values
and enrichment, but it does not recompute empirical FWER.
"""


from __future__ import annotations


import os
import re
from pathlib import Path
from typing import Iterable


import numpy as np
import pandas as pd
from scipy.stats import poisson




# ==========================================================
# Configuration
# ==========================================================


N_COMPLEX_CASES = 161
MISFIT_THRESHOLD = 0.005


ANNOTATION_FILE = "results/EA_TEF_cases_annotated.tsv"
MUTATION_RATE_FILE = "data/gene_mutation_rate.new.txt"
BLACKLIST_CANDIDATES = [
   "data/blacklisted_genes.txt",
   "results/blacklisted_genes.txt",
   "scripts/data/blacklisted_genes.txt",
]
SELECTED_PATHWAY_FILE = "results/significant_pathway_union_lof_vs_dmis.tsv"


GO_GMT_CANDIDATES = [
   "/share/vault/Users/xl3126/EATEF/data/c5.all.v2024.1.Hs.symbols.gmt",
   "data/c5.all.v2024.1.Hs.symbols.gmt",
   "data/c5.go.bp.v2026.1.Hs.symbols.gmt",
]


EXCLUSION_PATHWAYS = [
   "GOBP_INTRACELLULAR_TRANSPORT",
   "GOBP_REGULATION_OF_AUTOPHAGY",
]


# Leave empty to automatically test selected pathway names containing METABOL.
# You may instead enter exact pathway names here.
METABOLIC_TERMS: list[str] = [
   "GOBP_CELLULAR_CATABOLIC_PROCESS",
   "GOBP_ORGANIC_ACID_METABOLIC_PROCESS",
   "GOBP_MONOCARBOXYLIC_ACID_METABOLIC_PROCESS",
   "GOBP_SMALL_MOLECULE_METABOLIC_PROCESS",
   "GOBP_SMALL_MOLECULE_CATABOLIC_PROCESS",
   "GOBP_ORGANONITROGEN_COMPOUND_CATABOLIC_PROCESS",
   "GOBP_LIPID_METABOLIC_PROCESS",
]


OUTPUT_RESULTS = (
   "results/metabolic_enrichment_excluding_transport_autophagy.tsv"
)
OUTPUT_OVERLAP = (
   "results/metabolic_overlap_excluded_genes.tsv"
)




# ==========================================================
# Helpers
# ==========================================================


def first_existing_path(paths: Iterable[str]) -> str:
   for path in paths:
       if os.path.exists(path):
           return path
   raise FileNotFoundError(
       "None of the candidate files exist:\n  " + "\n  ".join(paths)
   )




def normalize_gene(value: object) -> str:
   if pd.isna(value):
       return ""
   return str(value).strip().upper()




def normalize_pathway(value: object) -> str:
   if pd.isna(value):
       return ""
   return str(value).strip().upper()




def find_exact_column(columns: Iterable[str], candidates: Iterable[str]) -> str:
   columns = list(columns)
   mapping = {str(column).strip().lower(): column for column in columns}
   for candidate in candidates:
       key = candidate.strip().lower()
       if key in mapping:
           return mapping[key]
   raise ValueError(
       f"Could not identify a required column from {list(candidates)}. "
       f"Available columns: {columns}"
   )




def find_rate_column(columns: Iterable[str]) -> str:
   columns = list(columns)
   exact = [
       "Dmis", "dmis", "MisFit", "misfit", "missense",
       "mis_rate", "dmis_rate", "missense_rate",
   ]
   mapping = {str(column).strip().lower(): column for column in columns}
   for candidate in exact:
       if candidate.lower() in mapping:
           return mapping[candidate.lower()]


   for column in columns:
       lower = str(column).lower()
       if "mis" in lower and "rate" in lower:
           return column


   raise ValueError(
       "Could not identify the damaging-missense mutation-rate column. "
       f"Available columns: {columns}"
   )




def load_gmt(path: str) -> dict[str, set[str]]:
   pathway_to_genes: dict[str, set[str]] = {}
   with open(path, "r", encoding="utf-8", errors="replace") as handle:
       for raw_line in handle:
           fields = raw_line.rstrip("\n").split("\t")
           if len(fields) < 3:
               continue
           pathway = normalize_pathway(fields[0])
           genes = {normalize_gene(gene) for gene in fields[2:]}
           genes.discard("")
           if pathway:
               pathway_to_genes[pathway] = genes
   return pathway_to_genes




def load_blacklist(path: str) -> set[str]:
   if not os.path.exists(path):
       print(f"Warning: blacklist file not found: {path}")
       return set()


   genes: set[str] = set()
   with open(path, "r", encoding="utf-8", errors="replace") as handle:
       for raw_line in handle:
           value = raw_line.strip()
           if not value or value.startswith("#"):
               continue
           gene = normalize_gene(re.split(r"[\t,\s]+", value)[0])
           if gene:
               genes.add(gene)
   return genes




def poisson_stats(observed: int, expected: float) -> tuple[float, float]:
   if expected <= 0:
       if observed == 0:
           return np.nan, 1.0
       return np.inf, 0.0
   return observed / expected, float(poisson.sf(observed - 1, expected))




# ==========================================================
# Load pathway definitions
# ==========================================================


gmt_file = first_existing_path(GO_GMT_CANDIDATES)
for required_file in [ANNOTATION_FILE, MUTATION_RATE_FILE, SELECTED_PATHWAY_FILE]:
   if not os.path.exists(required_file):
       raise FileNotFoundError(f"Required file not found: {required_file}")


print(f"Using GO GMT file: {gmt_file}")
pathway_to_genes = load_gmt(gmt_file)
blacklist_file = next(
   (
       path
       for path in BLACKLIST_CANDIDATES
       if os.path.exists(path)
   ),
   None,
)


if blacklist_file is None:
   print(
       "Warning: no blacklist file was found. Checked: "
       + ", ".join(BLACKLIST_CANDIDATES)
   )
   blacklisted_genes = set()
else:
   print(f"Using blacklist file: {blacklist_file}")
   blacklisted_genes = load_blacklist(blacklist_file)


missing_exclusions = [
   pathway for pathway in EXCLUSION_PATHWAYS
   if pathway not in pathway_to_genes
]
if missing_exclusions:
   raise ValueError(
       "Exclusion pathway(s) missing from GMT: " + ", ".join(missing_exclusions)
   )


exclusion_gene_sets = {
   pathway: pathway_to_genes[pathway] - blacklisted_genes
   for pathway in EXCLUSION_PATHWAYS
}
genes_to_remove = set().union(*exclusion_gene_sets.values())


print(f"Blacklisted genes: {len(blacklisted_genes):,}")
print(f"Unique exclusion genes: {len(genes_to_remove):,}")




# ==========================================================
# Select metabolic pathways from the 15-pathway table
# ==========================================================


selected = pd.read_csv(SELECTED_PATHWAY_FILE, sep="\t", low_memory=False)
pathway_column = find_exact_column(
   selected.columns,
   ["Pathway", "pathway", "Term"],
)
selected[pathway_column] = selected[pathway_column].map(normalize_pathway)
selected_pathways = selected[pathway_column].drop_duplicates().tolist()


if METABOLIC_TERMS:
   metabolic_terms = [normalize_pathway(term) for term in METABOLIC_TERMS]
else:
   metabolic_terms = [
       pathway for pathway in selected_pathways
       if "METABOL" in pathway
   ]


if not metabolic_terms:
   raise ValueError(
       "No selected metabolic pathways were found. Populate METABOLIC_TERMS "
       "near the top of the script with exact GO pathway names."
   )


missing_metabolic = [
   pathway for pathway in metabolic_terms
   if pathway not in pathway_to_genes
]
if missing_metabolic:
   raise ValueError(
       "Metabolic pathway(s) missing from GMT: " + ", ".join(missing_metabolic)
   )


print(f"Metabolic pathways tested: {len(metabolic_terms)}")
for pathway in metabolic_terms:
   print(f"  - {pathway}")




# ==========================================================
# Load qualifying complex-case LoF and Dmis variants
# ==========================================================


variants = pd.read_csv(ANNOTATION_FILE, sep="\t", low_memory=False)
gene_column = find_exact_column(variants.columns, ["HGNC", "Gene", "SYMBOL"])
variant_type_column = find_exact_column(
   variants.columns,
   ["Var_type", "Variant_type", "Consequence"],
)
misfit_column = find_exact_column(
   variants.columns,
   ["MisFit_S", "MisFit1.5_S", "MisFit"],
)
phenotype_column = find_exact_column(
   variants.columns,
   ["Syndormic or not", "Syndromic or not", "Cohort", "Phenotype"],
)


variants[gene_column] = variants[gene_column].map(normalize_gene)
variants[misfit_column] = pd.to_numeric(
   variants[misfit_column],
   errors="coerce",
)


variant_type_lower = (
   variants[variant_type_column]
   .astype(str)
   .str.lower()
)

is_complex = (
   variants[phenotype_column]
   .astype(str)
   .str.lower()
   .str.contains("complex|syndrom", regex=True, na=False)
)

is_missense = variant_type_lower.str.contains(
   "missense|dmis",
   regex=True,
   na=False,
)

# This handles a direct "LoF" label as well as common LoF consequence labels.
is_lof = variant_type_lower.str.contains(
   (
       r"\blof\b|loss[_ -]?of[_ -]?function|frameshift|"
       r"stop[_ -]?gained|stop[_ -]?lost|start[_ -]?lost|"
       r"splice[_ -]?acceptor|splice[_ -]?donor"
   ),
   regex=True,
   na=False,
)

is_dmis = (
   is_missense
   & variants[misfit_column].ge(MISFIT_THRESHOLD)
   & variants[misfit_column].ne(-1)
)

valid_gene = (
   variants[gene_column].ne("")
   & ~variants[gene_column].isin(blacklisted_genes)
)

lof_variants = variants.loc[
   is_complex
   & is_lof
   & valid_gene
].copy()

dmis_variants = variants.loc[
   is_complex
   & is_dmis
   & valid_gene
].copy()

lof_dmis_variants = pd.concat(
   [lof_variants, dmis_variants],
   axis=0,
   ignore_index=True,
)


print(f"Qualifying LoF variants: {len(lof_variants):,}")
print(f"Unique LoF-carrying genes: {lof_variants[gene_column].nunique():,}")
print(f"Qualifying Dmis variants: {len(dmis_variants):,}")
print(f"Unique Dmis-carrying genes: {dmis_variants[gene_column].nunique():,}")
print(f"Qualifying LoF+Dmis variants: {len(lof_dmis_variants):,}")
print(
   "Unique LoF+Dmis-carrying genes: "
   f"{lof_dmis_variants[gene_column].nunique():,}"
)




# ==========================================================
# Load gene-level LoF and Dmis mutation rates
# ==========================================================


mutation_rates = pd.read_csv(
   MUTATION_RATE_FILE,
   sep="\t",
   low_memory=False,
)

rate_gene_column = find_exact_column(
   mutation_rates.columns,
   ["Gene_name", "gene", "Gene", "HGNC", "SYMBOL"],
)

lof_rate_column = find_exact_column(
   mutation_rates.columns,
   ["LoF"],
)

# Match the final Dmis definition used in the observed-variant filter:
# MisFit_S >= 0.005.
dmis_rate_column = find_exact_column(
   mutation_rates.columns,
   ["Dmis_MisFit_S_0.005"],
)


mutation_rates[rate_gene_column] = (
   mutation_rates[rate_gene_column]
   .map(normalize_gene)
)

for rate_column in [lof_rate_column, dmis_rate_column]:
   mutation_rates[rate_column] = pd.to_numeric(
       mutation_rates[rate_column],
       errors="coerce",
   ).fillna(0.0)

mutation_rates = mutation_rates.loc[
   mutation_rates[rate_gene_column].ne("")
   & ~mutation_rates[rate_gene_column].isin(blacklisted_genes)
].copy()


gene_to_lof_rate = (
   mutation_rates.groupby(rate_gene_column)[lof_rate_column]
   .sum()
   .to_dict()
)

gene_to_dmis_rate = (
   mutation_rates.groupby(rate_gene_column)[dmis_rate_column]
   .sum()
   .to_dict()
)


print(f"Using mutation-rate gene column: {rate_gene_column}")
print(f"Using LoF mutation-rate column: {lof_rate_column}")
print(f"Using Dmis mutation-rate column: {dmis_rate_column}")




# ==========================================================
# Recalculate LoF, Dmis, and LoF+Dmis enrichment after exclusions
# ==========================================================


result_rows: list[dict[str, object]] = []
overlap_rows: list[dict[str, object]] = []

lof_gene_counts = (
   lof_variants[gene_column]
   .value_counts()
   .to_dict()
)
dmis_gene_counts = (
   dmis_variants[gene_column]
   .value_counts()
   .to_dict()
)
lof_dmis_gene_counts = (
   lof_dmis_variants[gene_column]
   .value_counts()
   .to_dict()
)


analysis_definitions = {
   "LoF": {
       "variants": lof_variants,
       "rate_map": gene_to_lof_rate,
   },
   "Dmis": {
       "variants": dmis_variants,
       "rate_map": gene_to_dmis_rate,
   },
   "LoF_Dmis": {
       "variants": lof_dmis_variants,
       "rate_map": {
           gene: (
               gene_to_lof_rate.get(gene, 0.0)
               + gene_to_dmis_rate.get(gene, 0.0)
           )
           for gene in (
               set(gene_to_lof_rate)
               | set(gene_to_dmis_rate)
           )
       },
   },
}


for pathway in metabolic_terms:
   original_genes = (
       pathway_to_genes[pathway]
       - blacklisted_genes
   )
   excluded_overlap = (
       original_genes
       & genes_to_remove
   )
   remaining_genes = (
       original_genes
       - genes_to_remove
   )


   row: dict[str, object] = {
       "Pathway": pathway,
       "Original_Total_Genes": len(original_genes),
       "Excluded_Overlap_Genes": len(excluded_overlap),
       "Remaining_Genes": len(remaining_genes),
       "MisFit_Threshold": MISFIT_THRESHOLD,
       "N_Complex_Cases": N_COMPLEX_CASES,
   }


   for analysis_name, analysis in analysis_definitions.items():
       analysis_variants = analysis["variants"]
       rate_map = analysis["rate_map"]

       original_observed = int(
           analysis_variants[gene_column]
           .isin(original_genes)
           .sum()
       )
       remaining_observed = int(
           analysis_variants[gene_column]
           .isin(remaining_genes)
           .sum()
       )

       original_observed_genes = int(
           analysis_variants.loc[
               analysis_variants[gene_column].isin(original_genes),
               gene_column,
           ].nunique()
       )
       remaining_observed_genes = int(
           analysis_variants.loc[
               analysis_variants[gene_column].isin(remaining_genes),
               gene_column,
           ].nunique()
       )

       original_expected = (
           N_COMPLEX_CASES
           * sum(
               rate_map.get(gene, 0.0)
               for gene in original_genes
           )
       )
       remaining_expected = (
           N_COMPLEX_CASES
           * sum(
               rate_map.get(gene, 0.0)
               for gene in remaining_genes
           )
       )

       original_enrichment, original_p = poisson_stats(
           original_observed,
           original_expected,
       )
       remaining_enrichment, remaining_p = poisson_stats(
           remaining_observed,
           remaining_expected,
       )

       removed_observed = (
           original_observed
           - remaining_observed
       )

       row.update({
           f"Original_{analysis_name}_Observed_Variants": original_observed,
           f"Original_{analysis_name}_Observed_Genes": original_observed_genes,
           f"Original_{analysis_name}_Expected": original_expected,
           f"Original_{analysis_name}_Enrichment": original_enrichment,
           f"Original_{analysis_name}_Raw_P_Value": original_p,
           f"Removed_{analysis_name}_Observed_Variants": removed_observed,
           f"Remaining_{analysis_name}_Observed_Variants": remaining_observed,
           f"Remaining_{analysis_name}_Observed_Genes": remaining_observed_genes,
           f"Remaining_{analysis_name}_Expected": remaining_expected,
           f"Remaining_{analysis_name}_Enrichment": remaining_enrichment,
           f"Remaining_{analysis_name}_Raw_P_Value": remaining_p,
           f"Remaining_{analysis_name}_Significant_P_LT_0_05": (
               remaining_p < 0.05
           ),
           f"Fraction_Observed_{analysis_name}_Removed": (
               removed_observed / original_observed
               if original_observed > 0
               else np.nan
           ),
       })


   result_rows.append(row)


   for gene in sorted(excluded_overlap):
       memberships = [
           exclusion_pathway
           for exclusion_pathway, gene_set
           in exclusion_gene_sets.items()
           if gene in gene_set
       ]

       overlap_rows.append({
           "Metabolic_Pathway": pathway,
           "Excluded_Gene": gene,
           "Exclusion_Pathway_Membership": ";".join(memberships),
           "LoF_Variant_Count_In_Complex_Cases": int(
               lof_gene_counts.get(gene, 0)
           ),
           "Dmis_Variant_Count_In_Complex_Cases": int(
               dmis_gene_counts.get(gene, 0)
           ),
           "LoF_Dmis_Variant_Count_In_Complex_Cases": int(
               lof_dmis_gene_counts.get(gene, 0)
           ),
           "LoF_Carrying_Gene": (
               lof_gene_counts.get(gene, 0) > 0
           ),
           "Dmis_Carrying_Gene": (
               dmis_gene_counts.get(gene, 0) > 0
           ),
           "LoF_Dmis_Carrying_Gene": (
               lof_dmis_gene_counts.get(gene, 0) > 0
           ),
           "LoF_Mutation_Rate": float(
               gene_to_lof_rate.get(gene, 0.0)
           ),
           "Dmis_Mutation_Rate": float(
               gene_to_dmis_rate.get(gene, 0.0)
           ),
           "LoF_Dmis_Mutation_Rate": float(
               gene_to_lof_rate.get(gene, 0.0)
               + gene_to_dmis_rate.get(gene, 0.0)
           ),
       })


results = pd.DataFrame(result_rows).sort_values(
   ["Remaining_LoF_Dmis_Raw_P_Value", "Pathway"]
)
overlap_details = pd.DataFrame(overlap_rows)


Path(OUTPUT_RESULTS).parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_RESULTS, sep="\t", index=False)
overlap_details.to_csv(OUTPUT_OVERLAP, sep="\t", index=False)


print("\nPost-exclusion results:")
print(results[[
   "Pathway",
   "Remaining_LoF_Observed_Variants",
   "Remaining_LoF_Expected",
   "Remaining_LoF_Enrichment",
   "Remaining_LoF_Raw_P_Value",
   "Remaining_Dmis_Observed_Variants",
   "Remaining_Dmis_Expected",
   "Remaining_Dmis_Enrichment",
   "Remaining_Dmis_Raw_P_Value",
   "Remaining_LoF_Dmis_Observed_Variants",
   "Remaining_LoF_Dmis_Expected",
   "Remaining_LoF_Dmis_Enrichment",
   "Remaining_LoF_Dmis_Raw_P_Value",
]].to_string(index=False))


print(f"\nSaved pathway results: {OUTPUT_RESULTS}")
print(f"Saved excluded-gene details: {OUTPUT_OVERLAP}")