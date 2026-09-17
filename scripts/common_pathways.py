
#!/usr/bin/env python3


import os


import re


import gseapy as gp
import pandas as pd


# ==========================================================
# Files
# ==========================================================


MUTATION_RATE_FILE = "data/gene_mutation_rate.new.txt"
CASE_FILE = "results/EA_TEF_cases_annotated.tsv"


GMT_FILE = (
  "/share/vault/Users/xl3126/EATEF/data/"
  "c5.all.v2024.1.Hs.symbols.gmt"
)




OUTPUT_FILE = "results/common_pathway_universe.tsv"


# One gene symbol per line, or a delimited file whose first column contains genes.
BLACKLIST_FILE = "data/09GENCODEV19_blacklist.txt"


# ==========================================================
# Parameters
# ==========================================================


N_CASES = 161


GENE_COLUMN = "Gene_name"
LOF_COLUMN = "LoF"
DMIS_COLUMN = "Dmis_MisFit_S_0.005"


CASE_GENE_COLUMN = "HGNC"
CASE_COHORT_COLUMN = "Syndormic or not"
COMPLEX_COHORT_LABEL = "Complex"


CASE_VARIANT_TYPE_COLUMN = "Var_type"
CASE_CONSEQUENCE_COLUMN = "GeneEff"
CASE_MISFIT_COLUMN = "MisFit_S"


DMIS_THRESHOLD = 0.005


MAX_PATHWAY_GENES = 2000


# Apply one shared minimum expected-count threshold using the
# combined LoF + Dmis expected burden.
MIN_COMBINED_EXPECTED = 0.5


# ==========================================================
# Helper functions
# ==========================================================


def load_blacklisted_genes(path):
   """
   Load blacklisted gene symbols from a text file.


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
               blacklisted_genes.add(
                   gene.upper()
               )


   if not blacklisted_genes:
       raise ValueError(
           f"No genes were loaded from {path}"
       )


   return blacklisted_genes


def normalize_text(series):
  """Convert a text column to stripped lowercase strings."""


  return (
      series
      .fillna("")
      .astype(str)
      .str.strip()
      .str.lower()
  )


def identify_lof_variants(cases):
  """
  Identify observed LoF variants.


  If a binary LoF column exists in the case file, use it.
  Otherwise classify variants using Var_type and GeneEff.
  """
  if "LoF" in cases.columns:
      lof_numeric = pd.to_numeric(
          cases["LoF"],
          errors="coerce",
      )


      if lof_numeric.notna().any():
          return lof_numeric.fillna(0).gt(0)


  mask = pd.Series(
      False,
      index=cases.index,
  )


  if CASE_VARIANT_TYPE_COLUMN in cases.columns:
      variant_type = normalize_text(
          cases[CASE_VARIANT_TYPE_COLUMN]
      )


      mask = mask | variant_type.str.contains(
          "lof|frameshift|stop_gained|splice_acceptor|"
          "splice_donor|start_lost",
          regex=True,
          na=False,
      )


  if CASE_CONSEQUENCE_COLUMN in cases.columns:
      consequence = normalize_text(
          cases[CASE_CONSEQUENCE_COLUMN]
      )


      mask = mask | consequence.str.contains(
          "frameshift|stop_gained|splice_acceptor|"
          "splice_donor|start_lost",
          regex=True,
          na=False,
      )


  return mask


def identify_dmis_variants(cases):
  """Identify observed deleterious missense variants."""


  if CASE_MISFIT_COLUMN not in cases.columns:
      raise ValueError(
          f"Missing case column: {CASE_MISFIT_COLUMN}"
      )


  misfit = pd.to_numeric(
      cases[CASE_MISFIT_COLUMN],
      errors="coerce",
  )


  mask = misfit.ge(
      DMIS_THRESHOLD
  )


  if CASE_VARIANT_TYPE_COLUMN in cases.columns:
      variant_type = normalize_text(
          cases[CASE_VARIANT_TYPE_COLUMN]
      )


      mask = mask & variant_type.str.contains(
          "missense",
          regex=False,
          na=False,
      )


  elif CASE_CONSEQUENCE_COLUMN in cases.columns:
      consequence = normalize_text(
          cases[CASE_CONSEQUENCE_COLUMN]
      )


      mask = mask & consequence.str.contains(
          "missense",
          regex=False,
          na=False,
      )


  return mask


# ==========================================================
# Load and clean mutation rates
# ==========================================================


print("Loading blacklist...")


blacklisted_genes = load_blacklisted_genes(
  BLACKLIST_FILE
)


print(
  f"Blacklisted genes loaded: {len(blacklisted_genes)}"
)




print("Loading mutation rates...")


mutation_rate = pd.read_csv(
  MUTATION_RATE_FILE,
  sep="\t",
  low_memory=False,
)


required_columns = {
  GENE_COLUMN,
  LOF_COLUMN,
  DMIS_COLUMN,
}


missing = required_columns - set(
  mutation_rate.columns
)


if missing:
  raise ValueError(
      "Missing required mutation-rate columns: "
      f"{sorted(missing)}"
  )


for column in [
  LOF_COLUMN,
  DMIS_COLUMN,
]:
  mutation_rate[column] = pd.to_numeric(
      mutation_rate[column],
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




mutation_rate_before_blacklist = len(mutation_rate)


mutation_rate = mutation_rate[
  ~mutation_rate[GENE_COLUMN].isin(
      blacklisted_genes
  )
].copy()


print(
  "Mutation-rate rows removed by blacklist: "
  f"{mutation_rate_before_blacklist - len(mutation_rate)}"
)


mutation_rate = (
  mutation_rate
  .groupby(
      GENE_COLUMN,
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
  GENE_COLUMN
)[
  [
      LOF_COLUMN,
      DMIS_COLUMN,
  ]
]


rate_genes = set(
  rate_lookup.index
)


# ==========================================================
# Load observed case variants
# ==========================================================


print("Loading observed case variants...")


cases = pd.read_csv(
  CASE_FILE,
  sep="\t",
  low_memory=False,
)


if CASE_GENE_COLUMN not in cases.columns:
  raise ValueError(
      f"Missing case gene column: {CASE_GENE_COLUMN}"
  )


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
  "Observed variant rows removed by blacklist: "
  f"{cases_before_blacklist - len(cases)}"
)


if CASE_COHORT_COLUMN in cases.columns:
  cohort = normalize_text(
      cases[CASE_COHORT_COLUMN]
  )


  cases = cases[
      cohort == COMPLEX_COHORT_LABEL.lower()
  ].copy()


  print(
      f"Restricted observed variants to "
      f"{COMPLEX_COHORT_LABEL} cases."
  )
else:
  print(
      f"Warning: '{CASE_COHORT_COLUMN}' was not found; "
      "all observed variants will be used."
  )


lof_mask = identify_lof_variants(
  cases
)


dmis_mask = identify_dmis_variants(
  cases
)


lof_observed_genes = set(
  cases.loc[
      lof_mask,
      CASE_GENE_COLUMN,
  ]
)


dmis_observed_genes = set(
  cases.loc[
      dmis_mask,
      CASE_GENE_COLUMN,
  ]
)


print(
  "Unique genes carrying observed LoF variants: "
  f"{len(lof_observed_genes)}"
)


print(
  "Unique genes carrying observed Dmis variants: "
  f"{len(dmis_observed_genes)}"
)



print(
  "Example observed LoF genes: "
  f"{sorted(lof_observed_genes)[:10]}"
)


print(
  "Example observed Dmis genes: "
  f"{sorted(dmis_observed_genes)[:10]}"
)


# ==========================================================
# Build one common pathway universe
# ==========================================================


print("Loading GO pathways...")


gmt = gp.parser.read_gmt(
  GMT_FILE
)


rows = []


excluded_non_go = 0
excluded_too_large = 0
excluded_no_rate_genes = 0
excluded_nonpositive_combined_expected = 0
excluded_low_combined_expected = 0


for pathway_name, pathway_genes in gmt.items():


  if not pathway_name.startswith("GO"):
      excluded_non_go += 1
      continue


  gene_set = {
      str(gene).strip().upper()
      for gene in pathway_genes
      if str(gene).strip()
  }


  if len(gene_set) > MAX_PATHWAY_GENES:
      excluded_too_large += 1
      continue


  genes_with_rates = sorted(
      gene_set & rate_genes
  )


  if not genes_with_rates:
      excluded_no_rate_genes += 1
      continue


  lof_rate_sum = float(
      rate_lookup.loc[
          genes_with_rates,
          LOF_COLUMN,
      ].sum()
  )


  dmis_rate_sum = float(
      rate_lookup.loc[
          genes_with_rates,
          DMIS_COLUMN,
      ].sum()
  )


  lof_expected = (
      2
      * N_CASES
      * lof_rate_sum
  )


  dmis_expected = (
      2
      * N_CASES
      * dmis_rate_sum
  )


  combined_expected = (
      lof_expected
      + dmis_expected
  )


  if combined_expected <= 0:
      excluded_nonpositive_combined_expected += 1
      continue


  if combined_expected < MIN_COMBINED_EXPECTED:
      excluded_low_combined_expected += 1
      continue


  pathway_lof_genes = sorted(
      gene_set
      & lof_observed_genes
  )


  pathway_dmis_genes = sorted(
      gene_set
      & dmis_observed_genes
  )


  rows.append(
      {
          "Pathway": pathway_name,
          "GenesInPathway": len(
              gene_set
          ),
          "GenesWithMutationRate": len(
              genes_with_rates
          ),
          "LoF_Rate_Sum": lof_rate_sum,
          "Dmis_Rate_Sum": dmis_rate_sum,
          "LoF_Expected": lof_expected,
          "Dmis_Expected": dmis_expected,
          "LoF_Dmis_Expected": combined_expected,
          "LoF_Genes": ";".join(
              pathway_lof_genes
          ),
          "Dmis_Genes": ";".join(
              pathway_dmis_genes
          ),
          "Genes": ";".join(
              genes_with_rates
          ),
      }
  )


# ==========================================================
# Validate and save
# ==========================================================


common_pathways = pd.DataFrame(
  rows
)


if common_pathways.empty:
  raise ValueError(
      "No pathways passed the shared pathway-universe "
      "criteria. Check MIN_COMBINED_EXPECTED and the "
      "mutation-rate columns."
  )


if common_pathways[
  "Pathway"
].duplicated().any():


  duplicate_pathways = (
      common_pathways.loc[
          common_pathways[
              "Pathway"
          ].duplicated(
              keep=False
          ),
          "Pathway",
      ]
      .astype(str)
      .tolist()
  )


  raise ValueError(
      "Duplicate pathways were found: "
      f"{duplicate_pathways[:10]}"
  )


common_pathways = (
  common_pathways
  .sort_values(
      "Pathway"
  )
  .reset_index(
      drop=True
  )
)


output_directory = os.path.dirname(
  OUTPUT_FILE
)


if output_directory:
  os.makedirs(
      output_directory,
      exist_ok=True,
  )


nonempty_lof_pathways = (
  common_pathways["LoF_Genes"]
  .fillna("")
  .astype(str)
  .str.strip()
  .ne("")
  .sum()
)


nonempty_dmis_pathways = (
  common_pathways["Dmis_Genes"]
  .fillna("")
  .astype(str)
  .str.strip()
  .ne("")
  .sum()
)


print(
  "Pathways containing at least one observed LoF gene: "
  f"{nonempty_lof_pathways}"
)


print(
  "Pathways containing at least one observed Dmis gene: "
  f"{nonempty_dmis_pathways}"
)


if lof_observed_genes and nonempty_lof_pathways == 0:
  raise ValueError(
      "Observed LoF genes were found, but none matched any GMT pathway. "
      "Check gene-symbol normalization and GMT contents."
  )


if dmis_observed_genes and nonempty_dmis_pathways == 0:
  raise ValueError(
      "Observed Dmis genes were found, but none matched any GMT pathway. "
      "Check gene-symbol normalization and GMT contents."
  )


common_pathways.to_csv(
  OUTPUT_FILE,
  sep="\t",
  index=False,
)


print()
print(
  "Shared filter used: "
  f"LoF+Dmis expected >= "
  f"{MIN_COMBINED_EXPECTED}"
)


print(
  "No separate minimum LoF or Dmis expected value was required."
)


print(
  f"Maximum pathway size: "
  f"{MAX_PATHWAY_GENES}"
)


print(
  f"Common pathways retained: "
  f"{len(common_pathways)}"
)


print(
  f"Excluded non-GO pathways: "
  f"{excluded_non_go}"
)


print(
  f"Excluded pathways larger than "
  f"{MAX_PATHWAY_GENES}: "
  f"{excluded_too_large}"
)


print(
  "Excluded pathways with no mutation-rate genes: "
  f"{excluded_no_rate_genes}"
)


print(
  "Excluded pathways with nonpositive combined expected: "
  f"{excluded_nonpositive_combined_expected}"
)


print(
  "Excluded pathways with combined expected below "
  f"{MIN_COMBINED_EXPECTED}: "
  f"{excluded_low_combined_expected}"
)


print()


print(
  common_pathways[
      [
          "Pathway",
          "GenesInPathway",
          "GenesWithMutationRate",
          "LoF_Expected",
          "Dmis_Expected",
          "LoF_Dmis_Expected",
          "LoF_Genes",
          "Dmis_Genes",
      ]
  ]
  .head(20)
  .to_string(
      index=False
  )
)


print()


print(
  f"Saved {len(common_pathways)} common pathways "
  f"to {OUTPUT_FILE}"
)