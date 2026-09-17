
#!/usr/bin/env python3


import os
import numpy as np
import pandas as pd
from scipy.stats import poisson


# ==========================================================
# Files
# ==========================================================


CASE_FILE = "results/variants_with_esm.tsv"
EXPECTED_FILE = "results/gene_mutation_rate_with_esm.tsv"


OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# Cohort size: Complex cases only
# ==========================================================


N_CASES = 161


# ==========================================================
# Thresholds
# ==========================================================


thresholds = {


   "MisFit_S": [0.002, 0.005, 0.01, 0.02],


   "AlphaMissense": [0.7, 0.9],


   "REVEL": [0.5, 0.6, 0.7],


   "CADD": [20, 25],


}


mutation_rate_columns = {


   "MisFit_S": {
       0.002: "Dmis_MisFit_S_0.002",
       0.005: "Dmis_MisFit_S_0.005",
       0.01: "Dmis_MisFit_S_0.01",
       0.02: "Dmis_MisFit_S_0.02",
   },


   "AlphaMissense": {
       0.7: "Dmis_AlphaMissense_0.7",
       0.9: "Dmis_AlphaMissense_0.9",
   },


   "REVEL": {
       0.5: "Dmis_REVEL_0.5",
       0.6: "Dmis_REVEL_0.6",
       0.7: "Dmis_REVEL_0.7",
   },


   "CADD": {
       20: "Dmis_CADD_20",
       25: "Dmis_CADD_25",
   },


}


# ==========================================================
# Load files
# ==========================================================


print("Loading EA/TEF variants...")


cases = pd.read_csv(
   CASE_FILE,
   sep="\t",
   low_memory=False,
)

# ==========================================================
# Keep Complex cases only
# ==========================================================

PHENOTYPE_COLUMN = "Syndormic or not"

if PHENOTYPE_COLUMN not in cases.columns:
   raise ValueError(
       f"Missing required phenotype column: {PHENOTYPE_COLUMN}"
   )

cases[PHENOTYPE_COLUMN] = (
   cases[PHENOTYPE_COLUMN]
   .astype("string")
   .str.strip()
   .str.lower()
)

cases = cases[
   cases[PHENOTYPE_COLUMN].eq("complex")
].copy()

print("Complex-only variants:", len(cases))
print(
   "Complex-only unique samples:",
   cases["Sample_id"].nunique(),
)


print(os.path.abspath(CASE_FILE))


print(
   cases[
       ["AlphaMissense", "REVEL", "MisFit_S"]
   ].notna().sum()
)


mutation_rate = pd.read_csv(
   EXPECTED_FILE,
   sep="\t",
)


# ==========================================================
# Numeric conversion
# ==========================================================


score_columns = [
   "MisFit_S",
   "AlphaMissense",
   "REVEL",
   "CADD",
]


for col in score_columns:


   if col not in cases.columns:
       continue


   print(f"\n===== {col} BEFORE =====")
   print(cases[col].head(10))
   print(cases[col].dtype)


   converted = pd.to_numeric(
       cases[col],
       errors="coerce",
   )


   print(
       f"After conversion: {converted.notna().sum()} non-missing"
   )


   bad = cases.loc[
       converted.isna(),
       col,
   ].drop_duplicates()


   print("Unique values converted to NaN:")
   print(bad.tolist()[:30])


   cases[col] = converted
  
cases["MisFit_S"] = cases["MisFit_S"].replace(-1, np.nan)


# ==========================================================
# Keep missense variants only
# ==========================================================


cases = cases[
   cases["Var_type"]
   .astype(str)
   .str.lower()
   .isin(["mis", "missense"])
].copy()


print("\nAfter missense filter")
print(
   cases[
       ["AlphaMissense", "REVEL", "MisFit_S"]
   ].notna().sum()
)


print("\nSample missing rows")
print(
   cases.loc[
       cases["AlphaMissense"].isna(),
       ["HGNC", "AlphaMissense", "REVEL", "MisFit_S"],
   ].head(20)
)


print(f"Missense variants: {len(cases)}")


# ==========================================================
# Burden analysis
# ==========================================================


results = []


for method, method_thresholds in thresholds.items():


   if method not in cases.columns:
       print(f"Skipping {method}")
       continue


   valid = cases.dropna(subset=[method])


   for threshold in method_thresholds:


       observed = (
           valid[method] >= threshold
       ).sum()


       rate_column = mutation_rate_columns[method][threshold]


       expected = (
           2
           * N_CASES
           * mutation_rate[rate_column].sum()
       )


       enrichment = (
           observed / expected
           if expected > 0
           else np.nan
       )


       # One-sided Poisson exact test
       p_value = poisson.sf(
           observed - 1,
           expected,
       )


       results.append({


           "Method": method,


           "Threshold": threshold,


           "Observed": observed,


           "ObservedRate": observed / len(valid),


           "Expected": expected,


           "EstimatedRiskVariants": observed - expected,


           "Enrichment": enrichment,


           "P-Value": p_value,


           "ScoredVariants": len(valid),


       })


# ==========================================================
# Save results
# ==========================================================


results = (
   pd.DataFrame(results)
   .sort_values(
       ["Method", "Threshold"]
   )
   .reset_index(drop=True)
)


print("\nResults:")
print(results)


results.to_csv(
   os.path.join(
       OUTPUT_DIR,
       "eatef_burden_results_complex.tsv",
   ),
   sep="\t",
   index=False,
)


print("\nFinished.")