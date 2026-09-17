#!/usr/bin/env python3

import os
import re
import pandas as pd

# ==========================================================
# Files
# ==========================================================

INPUT_FILE = "results/EA_TEF_cases_annotated.tsv"

BIOMART_FILE = "data/ensp_to_uniprot.tsv"

ESM_FOLDER = "data/content/ALL_hum_isoforms_ESM1b_LLR"

OUTPUT_FILE = "results/variants_with_esm.tsv"

# ==========================================================
# Load variants
# ==========================================================

variants = pd.read_csv(
    INPUT_FILE,
    sep="\t",
    low_memory=False,
)

variants["ENSP"] = (
    variants["HGVSp"]
    .astype(str)
    .str.extract(r"(ENSP\d+)")
)

print("Rows:", len(variants))
print(variants["Var_type"].value_counts())
print("Unique samples:", variants["Sample_id"].nunique())

# ==========================================================
# Load BioMart mapping
# ==========================================================

mapping = pd.read_csv(
    BIOMART_FILE,
    sep="\t",
)

mapping = (
    mapping[
        [
            "Protein stable ID",
            "UniProtKB/Swiss-Prot ID",
        ]
    ]
    .rename(
        columns={
            "Protein stable ID": "ENSP",
            "UniProtKB/Swiss-Prot ID": "UniProt",
        }
    )
    .dropna(subset=["ENSP", "UniProt"])
    .drop_duplicates(subset=["ENSP"])
)
# ==========================================================
# Merge ENSP → UniProt
# ==========================================================

variants = variants.merge(
    mapping,
    on="ENSP",
    how="left",
)

print("Variants with UniProt:",
      variants["UniProt"].notna().sum())

# ==========================================================
# Amino-acid conversion
# ==========================================================

aa = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
}

pattern = re.compile(
    r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})"
)

# ==========================================================
# Annotate ESM
# ==========================================================

esm_cache = {}
esm_scores = []

for row in variants.itertuples(index=False):

    if str(row.Var_type).lower() != "missense":
        esm_scores.append(None)
        continue

    if pd.isna(row.HGVSp):
        esm_scores.append(None)
        continue

    m = pattern.search(str(row.HGVSp))

    if m is None:
        esm_scores.append(None)
        continue

    ref3 = m.group(1)
    pos = int(m.group(2))
    alt3 = m.group(3)

    if ref3 not in aa or alt3 not in aa:
        esm_scores.append(None)
        continue

    ref = aa[ref3]
    alt = aa[alt3]

    if pd.isna(row.UniProt):
        esm_scores.append(None)
        continue

    if row.UniProt not in esm_cache:

        esm_file = os.path.join(
            ESM_FOLDER,
            f"{row.UniProt}_LLR.csv",
        )

        if os.path.exists(esm_file):
            esm_cache[row.UniProt] = pd.read_csv(
                esm_file,
                index_col=0,
            )
        else:
            esm_cache[row.UniProt] = None

    esm = esm_cache[row.UniProt]

    if esm is None:
        esm_scores.append(None)
        continue

    column = f"{ref} {pos}"

    if column not in esm.columns:
        esm_scores.append(None)
        continue

    if alt not in esm.index:
        esm_scores.append(None)
        continue

    esm_scores.append(
        esm.at[alt, column]
    )

# ==========================================================
# Save
# ==========================================================

variants["ESM"] = esm_scores

variants.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False,
)

print()
print("Finished!")
print("Variants with UniProt:",
      variants["UniProt"].notna().sum())
print("ESM annotated:",
      variants["ESM"].notna().sum())
print("Rows:",
      len(variants))
print("Saved to:",
      OUTPUT_FILE)