
#!/usr/bin/env python3


"""
Post-analysis GO redundancy reduction after two-test figure selection.


This script is run after the original LoF, Dmis, and LoF+Dmis enrichment
analyses and after permutation-based FWER has already been calculated.


For the pathways in the significant-pathway union, it:


1. Treats the input union table as the already-selected final figure set
  (for example, the 15 pathways that passed
  min(Dmis FWER, LoF+Dmis FWER) < 0.025).
2. Builds an induced GO hierarchy using all pathways in that fixed input set.
4. Traverses that retained hierarchy from the bottom upward.
5. For each parent, removes the already-pruned gene sets of its immediate
  retained GO children.
6. Recalculates descriptive raw Poisson p-values and enrichment values for
  Dmis and LoF+Dmis on the remaining genes.
6. Produces exactly one output row per input pathway.
7. Preserves the original Dmis and LoF+Dmis FWER values unchanged.


The post-analysis raw p-values are descriptive and are not subjected to a new
multiple-testing correction. They do not replace the original FWER results.
"""


import os
import re
from typing import Dict, Iterable, Set, Tuple


import numpy as np
import pandas as pd
from scipy.stats import poisson




# ==========================================================
# Files
# ==========================================================


# One row per unique significant pathway.
# Expected to contain Pathway and, when available:
# LoF_FWER, Dmis_FWER, LoF_Dmis_FWER
UNION_FILE = "results/significant_pathway_union_lof_vs_dmis.tsv"


# Shared pathway universe containing the complete gene set for each pathway.
# Required columns: Pathway, Genes
COMMON_PATHWAY_FILE = "results/common_pathway_universe.tsv"


# Download from Gene Ontology and place here.
GO_OBO_FILE = "data/go-basic.obo"


CASE_FILE = "results/EA_TEF_cases_annotated.tsv"
MUTATION_RATE_FILE = "data/gene_mutation_rate.new.txt"
BLACKLIST_FILE = "data/09GENCODEV19_blacklist.txt"


# Set either path to None if the corresponding original analysis did not use
# a synonymous-derived scale factor.
LOF_SCALE_FILE = "results/synonymous_scale_factor_lof.tsv"
DMIS_SCALE_FILE = "results/synonymous_scale_factor_misfit.tsv"


OUTPUT_FILE = "results/post_analysis_go_redundancy_results.tsv"




# ==========================================================
# Parameters
# ==========================================================


N_CASES = 161


# Include GO part_of relations in addition to is_a relations.
INCLUDE_PART_OF = True




# ==========================================================
# Column names and variant definitions
# ==========================================================


CASE_GENE_COLUMN = "HGNC"
CASE_PHENOTYPE_COLUMN = "Syndormic or not"
COMPLEX_LABEL = "complex"
CASE_VARIANT_TYPE_COLUMN = "Var_type"
CASE_MISFIT_COLUMN = "MisFit_S"


RATE_GENE_COLUMN = "Gene_name"
LOF_RATE_COLUMN = "LoF"
DMIS_RATE_COLUMN = "Dmis_MisFit_S_0.005"


MISFIT_THRESHOLD = 0.005


LOF_TYPES = {
   "frameshift",
   "frameshift_variant",
   "stop_gained",
   "stop_gained_variant",
   "splice_acceptor",
   "splice_acceptor_variant",
   "splice_donor",
   "splice_donor_variant",
}


MISSENSE_TYPES = {
   "missense",
   "missense_variant",
}




# ==========================================================
# Helper functions
# ==========================================================


def normalize_gene(value: object) -> str:
   return str(value).strip().upper()




def parse_genes(value: object) -> Set[str]:
   if pd.isna(value):
       return set()


   return {
       normalize_gene(gene)
       for gene in re.split(r"[;,]", str(value))
       if str(gene).strip()
   }




def load_blacklisted_genes(path: str | None) -> Set[str]:
   if path is None:
       return set()


   if not os.path.exists(path):
       raise FileNotFoundError(f"Blacklist file not found: {path}")


   genes: Set[str] = set()


   with open(path, "r", encoding="utf-8", errors="replace") as handle:
       for raw_line in handle:
           line = raw_line.strip()


           if not line or line.startswith("#"):
               continue


           first_field = re.split(r"[\t, ]+", line, maxsplit=1)[0].strip()


           if first_field:
               genes.add(first_field.upper())


   if not genes:
       raise ValueError(f"No genes were loaded from blacklist file: {path}")


   return genes




def load_scale_factor(path: str | None) -> float:
   if path is None:
       return 1.0


   if not os.path.exists(path):
       raise FileNotFoundError(
           f"Scale-factor file not found: {path}"
       )


   table = pd.read_csv(
       path,
       sep="\t",
       low_memory=False,
   )


   # This is the exact column name written by the original
   # synonymous-calibrated enrichment scripts. Do not fall back
   # to the first numeric value, because that could incorrectly
   # load ObservedSynonymous, ExpectedSynonymousUnscaled, or N_CASES.
   required_column = "ScaleFactor"


   if required_column not in table.columns:
       raise ValueError(
           f"{path} does not contain the required "
           f"'{required_column}' column. "
           f"Available columns: {list(table.columns)}"
       )


   values = pd.to_numeric(
       table[required_column],
       errors="coerce",
   ).dropna()


   if values.empty:
       raise ValueError(
           f"The {required_column} column in {path} "
           "contains no numeric value."
       )


   scale_factor = float(values.iloc[0])


   if not np.isfinite(scale_factor) or scale_factor <= 0:
       raise ValueError(
           f"Invalid scale factor in {path}: {scale_factor}"
       )


   return scale_factor



def parse_obo(path: str) -> Tuple[Dict[str, dict], Dict[str, str]]:
   if not os.path.exists(path):
       raise FileNotFoundError(f"GO OBO file not found: {path}")


   terms: Dict[str, dict] = {}
   alternate_ids: Dict[str, str] = {}
   current = None


   def finish(term: dict | None) -> None:
       if not term or "id" not in term:
           return


       if term.get("is_obsolete", "false") == "true":
           return


       term.setdefault("name", "")
       term.setdefault("namespace", "")
       term.setdefault("parents", set())


       terms[term["id"]] = term


       for alternate_id in term.get("alt_ids", []):
           alternate_ids[alternate_id] = term["id"]


   with open(path, "r", encoding="utf-8", errors="replace") as handle:
       for raw_line in handle:
           line = raw_line.rstrip("\n")


           if line == "[Term]":
               finish(current)
               current = {
                   "parents": set(),
                   "alt_ids": [],
               }
               continue


           if line.startswith("[") and line.endswith("]"):
               finish(current)
               current = None
               continue


           if current is None:
               continue


           if line.startswith("id: "):
               current["id"] = line[4:].strip()


           elif line.startswith("name: "):
               current["name"] = line[6:].strip()


           elif line.startswith("namespace: "):
               current["namespace"] = line[11:].strip()


           elif line.startswith("alt_id: "):
               current["alt_ids"].append(line[8:].strip())


           elif line.startswith("is_a: "):
               parent = line[6:].split(" ! ", 1)[0].strip()
               current["parents"].add(parent)


           elif INCLUDE_PART_OF and line.startswith("relationship: part_of "):
               parent = (
                   line[len("relationship: part_of "):]
                   .split(" ! ", 1)[0]
                   .strip()
               )
               current["parents"].add(parent)


           elif line.startswith("is_obsolete: "):
               current["is_obsolete"] = line[13:].strip()


   finish(current)


   return terms, alternate_ids




def infer_go_id(pathway: str, terms: Dict[str, dict]) -> str | None:
   label = str(pathway).strip()


   for prefix in ("GOBP_", "GOCC_", "GOMF_"):
       if label.startswith(prefix):
           label = label[len(prefix):]
           break


   normalized_name = label.replace("_", " ").strip().lower()


   matches = [
       go_id
       for go_id, term in terms.items()
       if str(term["name"]).strip().lower() == normalized_name
   ]


   if len(matches) == 1:
       return matches[0]


   return None




def build_ancestor_map(
   selected_ids: Iterable[str],
   terms: Dict[str, dict],
) -> Dict[str, Set[str]]:
   selected_set = set(selected_ids)
   output: Dict[str, Set[str]] = {}


   for child in selected_set:
       selected_ancestors: Set[str] = set()
       visited: Set[str] = set()
       stack = list(terms[child]["parents"])


       while stack:
           node = stack.pop()


           if node in visited:
               continue


           visited.add(node)


           if node in selected_set:
               selected_ancestors.add(node)


           if node in terms:
               stack.extend(terms[node]["parents"])


       output[child] = selected_ancestors


   return output






def build_immediate_selected_parent_map(
   selected_ids: Iterable[str],
   selected_ancestors: Dict[str, Set[str]],
) -> Dict[str, Set[str]]:
   """
   Build the induced hierarchy among selected GO terms.


   A selected ancestor is an immediate selected parent of a child when there
   is no other selected term between that child and ancestor in the GO DAG.
   This allows retained terms separated by unselected GO nodes to remain
   connected in the retained-term hierarchy.
   """
   selected_set = set(selected_ids)
   immediate_parents: Dict[str, Set[str]] = {
       go_id: set()
       for go_id in selected_set
   }


   for child_id in selected_set:
       candidate_parents = set(selected_ancestors[child_id])


       for parent_id in candidate_parents:
           has_selected_term_between = any(
               intermediate_id != parent_id
               and parent_id in selected_ancestors[intermediate_id]
               for intermediate_id in candidate_parents
           )


           if not has_selected_term_between:
               immediate_parents[child_id].add(parent_id)


   return immediate_parents




def build_depth_map(
   selected_ids: Iterable[str],
   terms: Dict[str, dict],
) -> Dict[str, int]:
   memo: Dict[str, int] = {}


   def depth(go_id: str, visiting: Set[str] | None = None) -> int:
       if go_id in memo:
           return memo[go_id]


       if visiting is None:
           visiting = set()


       if go_id in visiting:
           raise ValueError(f"Cycle detected in GO hierarchy at {go_id}")


       next_visiting = set(visiting)
       next_visiting.add(go_id)


       parents = [
           parent
           for parent in terms[go_id]["parents"]
           if parent in terms
       ]


       if not parents:
           value = 0
       else:
           value = 1 + max(
               depth(parent, next_visiting)
               for parent in parents
           )


       memo[go_id] = value
       return value


   return {
       go_id: depth(go_id)
       for go_id in selected_ids
   }






def calculate_statistics(
   genes: Set[str],
   observed_cases: pd.DataFrame,
   expected_column: str,
   rate_table: pd.DataFrame,
) -> dict:
   genes_with_rates = sorted(
       set(genes) & set(rate_table.index)
   )


   if genes_with_rates:
       expected = float(
           rate_table.loc[genes_with_rates, expected_column].sum()
       )


       observed = int(
           observed_cases[CASE_GENE_COLUMN]
           .isin(genes_with_rates)
           .sum()
       )
   else:
       expected = 0.0
       observed = 0


   if expected > 0:
       enrichment = observed / expected
       raw_p_value = float(
           poisson.sf(observed - 1, expected)
       )
   else:
       enrichment = np.nan
       raw_p_value = np.nan


   return {
       "Genes_With_Rates": len(genes_with_rates),
       "Observed": observed,
       "Expected": expected,
       "Enrichment": enrichment,
       "Raw_P_Value": raw_p_value,
   }




# ==========================================================
# Load union and pathway gene sets
# ==========================================================


print("Loading significant pathway union...")


union = pd.read_csv(
   UNION_FILE,
   sep="\t",
   low_memory=False,
)


if "Pathway" not in union.columns:
   raise ValueError("Union file must contain a Pathway column.")


union["Pathway"] = (
   union["Pathway"]
   .astype(str)
   .str.strip()
)


input_union_count = len(union)


print(
   "Using the union table as a fixed post-analysis pathway set. "
   f"Input pathways: {input_union_count}"
)


if union.empty:
   raise ValueError("The input union table contains no pathways.")


if union["Pathway"].duplicated().any():
   duplicates = (
       union.loc[
           union["Pathway"].duplicated(keep=False),
           "Pathway",
       ]
       .astype(str)
       .tolist()
   )


   raise ValueError(
       "The union file must have exactly one row per pathway. "
       f"Duplicate pathways found: {duplicates}"
   )


common = pd.read_csv(
   COMMON_PATHWAY_FILE,
   sep="\t",
   low_memory=False,
)


required_common = {"Pathway", "Genes"}
missing_common = required_common - set(common.columns)


if missing_common:
   raise ValueError(
       "Common-pathway file is missing columns: "
       f"{sorted(missing_common)}"
   )


common["Pathway"] = (
   common["Pathway"]
   .astype(str)
   .str.strip()
)


if common["Pathway"].duplicated().any():
   raise ValueError(
       "Common-pathway file contains duplicate Pathway values."
   )


union = union.merge(
   common[["Pathway", "Genes"]],
   on="Pathway",
   how="left",
   validate="one_to_one",
)


if union["Genes"].isna().any():
   missing_pathways = (
       union.loc[union["Genes"].isna(), "Pathway"]
       .astype(str)
       .tolist()
   )


   raise ValueError(
       "Some union pathways are absent from the common pathway universe: "
       f"{missing_pathways}"
   )


union["Original_Gene_Set"] = union["Genes"].apply(parse_genes)




# ==========================================================
# Load GO hierarchy and map terms
# ==========================================================


print("Loading GO hierarchy...")


terms, alternate_ids = parse_obo(GO_OBO_FILE)


mapped_go_ids = []


for row in union.itertuples(index=False):
   candidate = getattr(row, "GO_ID", None)
   go_id = None


   if candidate is not None and not pd.isna(candidate):
       candidate = str(candidate).strip()


       if candidate in terms:
           go_id = candidate
       elif candidate in alternate_ids:
           go_id = alternate_ids[candidate]


   if go_id is None:
       go_id = infer_go_id(row.Pathway, terms)


   mapped_go_ids.append(go_id)


union["GO_ID"] = mapped_go_ids


if union["GO_ID"].isna().any():
   unmapped = (
       union.loc[union["GO_ID"].isna(), "Pathway"]
       .astype(str)
       .tolist()
   )


   raise ValueError(
       "Could not map these pathways to unique GO IDs: "
       f"{unmapped}"
   )


if union["GO_ID"].duplicated().any():
   duplicate_rows = union.loc[
       union["GO_ID"].duplicated(keep=False),
       ["Pathway", "GO_ID"],
   ].sort_values("GO_ID")


   raise ValueError(
       "Multiple union pathways mapped to the same GO ID:\n"
       + duplicate_rows.to_string(index=False)
   )




# ==========================================================
# Apply blacklist
# ==========================================================


print("Loading blacklist...")


blacklisted_genes = load_blacklisted_genes(BLACKLIST_FILE)


union["Original_Gene_Set"] = union["Original_Gene_Set"].apply(
   lambda genes: set(genes) - blacklisted_genes
)




# ==========================================================
# Build the retained-term GO hierarchy and prune bottom-up
# ==========================================================


selected_ids = set(union["GO_ID"])


selected_ancestors = build_ancestor_map(
   selected_ids,
   terms,
)


immediate_selected_parents = build_immediate_selected_parent_map(
   selected_ids,
   selected_ancestors,
)


immediate_selected_children: Dict[str, Set[str]] = {
   go_id: set()
   for go_id in selected_ids
}


for child_id, parent_ids in immediate_selected_parents.items():
   for parent_id in parent_ids:
       immediate_selected_children[parent_id].add(child_id)


depths = build_depth_map(
   selected_ids,
   terms,
)


original_gene_sets = dict(
   zip(
       union["GO_ID"],
       union["Original_Gene_Set"],
   )
)


pathway_name_by_id = dict(
   zip(
       union["GO_ID"],
       union["Pathway"],
   )
)


remaining_gene_sets = {
   go_id: set(genes)
   for go_id, genes in original_gene_sets.items()
}


removed_by_children: Dict[str, Set[str]] = {
   go_id: set()
   for go_id in selected_ids
}


children_contributing_removal: Dict[str, Set[str]] = {
   go_id: set()
   for go_id in selected_ids
}


# Deepest retained terms are processed first. Therefore, when a retained
# child's genes are removed from its immediate retained parent, the child's
# gene set has already had its own retained children removed.
ordered_ids = sorted(
   selected_ids,
   key=lambda go_id: (
       depths[go_id],
       go_id,
   ),
   reverse=True,
)


for parent_id in ordered_ids:
   child_ids = immediate_selected_children[parent_id]


   for child_id in sorted(child_ids):
       child_pruned_genes = remaining_gene_sets[child_id]
       overlap = remaining_gene_sets[parent_id] & child_pruned_genes


       if not overlap:
           continue


       remaining_gene_sets[parent_id] -= overlap
       removed_by_children[parent_id].update(overlap)
       children_contributing_removal[parent_id].add(child_id)




print("Retained-term induced GO relationships:")


for parent_id in sorted(
   selected_ids,
   key=lambda go_id: (depths[go_id], pathway_name_by_id.get(go_id, go_id)),
):
   child_ids = immediate_selected_children[parent_id]


   if child_ids:
       child_names = [
           pathway_name_by_id[child_id]
           for child_id in sorted(
               child_ids,
               key=lambda go_id: pathway_name_by_id[go_id],
           )
       ]


       print(
           f"  {pathway_name_by_id[parent_id]} <- "
           + "; ".join(child_names)
       )




# ==========================================================
# Load observed variants
# ==========================================================


print("Loading observed variants...")


cases = pd.read_csv(
   CASE_FILE,
   sep="\t",
   low_memory=False,
   encoding="latin1",
)


required_case_columns = {
   CASE_GENE_COLUMN,
   CASE_PHENOTYPE_COLUMN,
   CASE_VARIANT_TYPE_COLUMN,
   CASE_MISFIT_COLUMN,
}


missing_case_columns = required_case_columns - set(cases.columns)


if missing_case_columns:
   raise ValueError(
       "Case file is missing columns: "
       f"{sorted(missing_case_columns)}"
   )


cases[CASE_GENE_COLUMN] = (
   cases[CASE_GENE_COLUMN]
   .astype(str)
   .str.strip()
   .str.upper()
)


cases[CASE_PHENOTYPE_COLUMN] = (
   cases[CASE_PHENOTYPE_COLUMN]
   .astype("string")
   .str.strip()
   .str.lower()
)


cases[CASE_VARIANT_TYPE_COLUMN] = (
   cases[CASE_VARIANT_TYPE_COLUMN]
   .astype(str)
   .str.strip()
   .str.lower()
)


cases[CASE_MISFIT_COLUMN] = pd.to_numeric(
   cases[CASE_MISFIT_COLUMN],
   errors="coerce",
)


cases.loc[
   cases[CASE_MISFIT_COLUMN].eq(-1),
   CASE_MISFIT_COLUMN,
] = np.nan


cases = cases[
   cases[CASE_PHENOTYPE_COLUMN].eq(COMPLEX_LABEL)
].copy()


cases = cases[
   cases[CASE_GENE_COLUMN].ne("")
   & ~cases[CASE_GENE_COLUMN].isin(blacklisted_genes)
].copy()


lof_cases = cases[
   cases[CASE_VARIANT_TYPE_COLUMN].isin(LOF_TYPES)
].copy()


dmis_cases = cases[
   cases[CASE_VARIANT_TYPE_COLUMN].isin(MISSENSE_TYPES)
   & cases[CASE_MISFIT_COLUMN].notna()
   & cases[CASE_MISFIT_COLUMN].ge(MISFIT_THRESHOLD)
].copy()


combined_cases = pd.concat(
   [lof_cases, dmis_cases],
   ignore_index=True,
)


print(f"Observed LoF variants: {len(lof_cases)}")
print(f"Observed Dmis variants: {len(dmis_cases)}")
print(f"Observed LoF+Dmis variants: {len(combined_cases)}")




# ==========================================================
# Load expected mutation rates
# ==========================================================


print("Loading background mutation rates...")


rates = pd.read_csv(
   MUTATION_RATE_FILE,
   sep="\t",
   low_memory=False,
)


required_rate_columns = {
   RATE_GENE_COLUMN,
   LOF_RATE_COLUMN,
   DMIS_RATE_COLUMN,
}


missing_rate_columns = required_rate_columns - set(rates.columns)


if missing_rate_columns:
   raise ValueError(
       "Mutation-rate file is missing columns: "
       f"{sorted(missing_rate_columns)}"
   )


rates[RATE_GENE_COLUMN] = (
   rates[RATE_GENE_COLUMN]
   .astype(str)
   .str.strip()
   .str.upper()
)


for column in (LOF_RATE_COLUMN, DMIS_RATE_COLUMN):
   rates[column] = pd.to_numeric(
       rates[column],
       errors="coerce",
   ).fillna(0)


rates = rates[
   rates[RATE_GENE_COLUMN].ne("")
   & ~rates[RATE_GENE_COLUMN].isin(blacklisted_genes)
].copy()


rates = (
   rates.groupby(RATE_GENE_COLUMN, as_index=False)[
       [LOF_RATE_COLUMN, DMIS_RATE_COLUMN]
   ]
   .sum()
)


lof_scale_factor = load_scale_factor(LOF_SCALE_FILE)
dmis_scale_factor = load_scale_factor(DMIS_SCALE_FILE)


# Keep this formula identical to the expected-count formula used in the
# original enrichment analysis. Change the factor of 2 only if your original
# scripts did not use diploid scaling.
rates["LoF_Expected"] = (
   2
   * N_CASES
   * lof_scale_factor
   * rates[LOF_RATE_COLUMN]
)


rates["Dmis_Expected"] = (
   2
   * N_CASES
   * dmis_scale_factor
   * rates[DMIS_RATE_COLUMN]
)


rates["LoF_Dmis_Expected"] = (
   rates["LoF_Expected"]
   + rates["Dmis_Expected"]
)


rate_table = rates.set_index(RATE_GENE_COLUMN)


print(f"LoF scale factor: {lof_scale_factor:.8g}")
print(f"Dmis scale factor: {dmis_scale_factor:.8g}")




# ==========================================================
# Calculate Dmis and LoF+Dmis analyses as columns
# ==========================================================


calculated_rows = []


for go_id in ordered_ids:
   original_genes = original_gene_sets[go_id]
   remaining_genes = remaining_gene_sets[go_id]


   child_names = sorted(
       pathway_name_by_id[child_id]
       for child_id in immediate_selected_children[go_id]
   )


   children_with_overlap_names = sorted(
       pathway_name_by_id[child_id]
       for child_id in children_contributing_removal[go_id]
   )




   dmis_original = calculate_statistics(
       original_genes,
       dmis_cases,
       "Dmis_Expected",
       rate_table,
   )


   dmis_post = calculate_statistics(
       remaining_genes,
       dmis_cases,
       "Dmis_Expected",
       rate_table,
   )


   combined_original = calculate_statistics(
       original_genes,
       combined_cases,
       "LoF_Dmis_Expected",
       rate_table,
   )


   combined_post = calculate_statistics(
       remaining_genes,
       combined_cases,
       "LoF_Dmis_Expected",
       rate_table,
   )


   calculated_rows.append(
       {
           "GO_ID": go_id,
           "GO_Term_Name": terms[go_id]["name"],
           "GO_Namespace": terms[go_id]["namespace"],
           "GO_Depth": depths[go_id],


           "Original_Total_Genes": len(original_genes),
           "Post_Analysis_Total_Genes": len(remaining_genes),
           "Genes_Removed": len(removed_by_children[go_id]),


           "Immediate_Retained_Child_Pathways": ";".join(child_names),
           "Children_Contributing_Gene_Removal": ";".join(
               children_with_overlap_names
           ),
           "Removed_Gene_Symbols": ";".join(
               sorted(removed_by_children[go_id])
           ),
           "Remaining_Gene_Symbols": ";".join(
               sorted(remaining_genes)
           ),




           "Dmis_Original_Genes_With_Rates": dmis_original["Genes_With_Rates"],
           "Dmis_Original_Observed": dmis_original["Observed"],
           "Dmis_Original_Expected": dmis_original["Expected"],
           "Dmis_Original_Enrichment": dmis_original["Enrichment"],
           "Dmis_Original_Raw_P_Value_Recalculated": dmis_original["Raw_P_Value"],


           "Dmis_Post_Analysis_Genes_With_Rates": dmis_post["Genes_With_Rates"],
           "Dmis_Post_Analysis_Observed": dmis_post["Observed"],
           "Dmis_Post_Analysis_Expected": dmis_post["Expected"],
           "Dmis_Post_Analysis_Enrichment": dmis_post["Enrichment"],
           "Dmis_Post_Analysis_Raw_P_Value": dmis_post["Raw_P_Value"],


           "LoF_Dmis_Original_Genes_With_Rates": combined_original["Genes_With_Rates"],
           "LoF_Dmis_Original_Observed": combined_original["Observed"],
           "LoF_Dmis_Original_Expected": combined_original["Expected"],
           "LoF_Dmis_Original_Enrichment": combined_original["Enrichment"],
           "LoF_Dmis_Original_Raw_P_Value_Recalculated": combined_original["Raw_P_Value"],


           "LoF_Dmis_Post_Analysis_Genes_With_Rates": combined_post["Genes_With_Rates"],
           "LoF_Dmis_Post_Analysis_Observed": combined_post["Observed"],
           "LoF_Dmis_Post_Analysis_Expected": combined_post["Expected"],
           "LoF_Dmis_Post_Analysis_Enrichment": combined_post["Enrichment"],
           "LoF_Dmis_Post_Analysis_Raw_P_Value": combined_post["Raw_P_Value"],
       }
   )


calculated = pd.DataFrame(calculated_rows)
calculated["Pathway"] = calculated["GO_ID"].map(pathway_name_by_id)




# ==========================================================
# Merge onto fixed input set: exactly one row per input pathway
# ==========================================================


# Remove helper-only columns before final output.
union_output = union.drop(
   columns=[
       "Genes",
       "Original_Gene_Set",
       "GO_ID",
   ],
   errors="ignore",
)


results = union_output.merge(
   calculated,
   on="Pathway",
   how="left",
   validate="one_to_one",
)


if len(results) != len(union):
   raise RuntimeError(
       "Output row count does not match union row count: "
       f"{len(results)} output rows versus {len(union)} union rows."
   )


if results["Pathway"].duplicated().any():
   raise RuntimeError("Output unexpectedly contains duplicate pathways.")


results = results.sort_values(
   [
       "GO_Depth",
       "Pathway",
   ],
   ascending=[
       False,
       True,
   ],
).reset_index(drop=True)




# ==========================================================
# Save output
# ==========================================================


output_directory = os.path.dirname(OUTPUT_FILE)


if output_directory:
   os.makedirs(output_directory, exist_ok=True)


results.to_csv(
   OUTPUT_FILE,
   sep="\t",
   index=False,
)


print()
print(
   results[
       [
           "Pathway",
           "GO_Depth",
           "Genes_Removed",
           "Dmis_Post_Analysis_Enrichment",
           "Dmis_Post_Analysis_Raw_P_Value",
           "LoF_Dmis_Post_Analysis_Enrichment",
           "LoF_Dmis_Post_Analysis_Raw_P_Value",
       ]
   ].to_string(index=False)
)
print()


print(f"Input pathways: {input_union_count}")
print(f"Output pathways: {len(results)}")
print(f"Saved output to: {OUTPUT_FILE}")
print(
   "The input table was treated as the already-selected final pathway set; "
   "no additional FWER filtering was performed."
)
print("Original Dmis and LoF+Dmis FWER values were preserved unchanged.")
print(
   "GO pruning was performed bottom-up using the already-pruned gene sets "
   "of immediate selected children in the induced pathway hierarchy."
)
print("Post-analysis raw p-values were not used to recalculate FWER.")