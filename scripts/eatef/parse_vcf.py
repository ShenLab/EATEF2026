#!/usr/bin/env python3

import gzip
import os
import pandas as pd


# ==========================================================
# Files
# ==========================================================

VCF_FILE = "data/EATEF_SPARK-unaff-sibs.ann.vcf.gz"

TABLE_S2_FILE = (
    "data/TableS2.Denovo_Variants_allCases_burden.txt"
)

EATEF_OUTPUT_FILE = (
    "results/EA_TEF_cases_annotated.tsv"
)

SPARK_OUTPUT_FILE = (
    "results/SPARK_controls_annotated.tsv"
)


# ==========================================================
# Parse INFO field
# ==========================================================

def parse_info(info_string):
    info = {}

    for item in info_string.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            info[key] = value

    return info


# ==========================================================
# Parse VEP CSQ field
# ==========================================================

def parse_csq(info):
    if "CSQ" not in info:
        return {}

    transcripts = info["CSQ"].split(",")

    selected = None

    for transcript in transcripts:
        fields = transcript.split("|")

        # PICK is field 22 in the CSQ definition below,
        # so its zero-based index is 21.
        if len(fields) > 21 and fields[21] == "1":
            selected = transcript
            break

    if selected is None:
        selected = transcripts[0]

    cols = selected.split("|")

    names = [
        "Allele",
        "Consequence",
        "IMPACT",
        "SYMBOL",
        "Gene",
        "Feature_type",
        "Feature",
        "BIOTYPE",
        "EXON",
        "INTRON",
        "HGVSc",
        "HGVSp",
        "cDNA_position",
        "CDS_position",
        "Protein_position",
        "Amino_acids",
        "Codons",
        "Existing_variation",
        "DISTANCE",
        "STRAND",
        "FLAGS",
        "PICK",
        "SYMBOL_SOURCE",
        "HGNC_ID",
        "CANONICAL",
        "MANE",
        "MANE_SELECT",
        "MANE_PLUS_CLINICAL",
        "APPRIS",
        "HGVS_OFFSET",
    ]

    annotation = {}

    for i, name in enumerate(names):
        if i < len(cols):
            annotation[name] = cols[i]

    return annotation


# ==========================================================
# Read Table S2 with encoding fallback
# ==========================================================

def read_table_s2(filename):
    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin-1",
    ]

    for encoding in encodings:
        try:
            table = pd.read_csv(
                filename,
                sep="\t",
                encoding=encoding,
                low_memory=False,
            )

            print(
                "Successfully read Table S2 using "
                f"encoding: {encoding}"
            )

            return table

        except UnicodeDecodeError:
            continue

    raise RuntimeError(
        "Could not decode Table S2 using UTF-8, "
        "UTF-8-SIG, CP1252, or Latin-1."
    )


# ==========================================================
# Parse annotated VCF
# ==========================================================

rows = []

print("Parsing annotated VCF...")

with gzip.open(VCF_FILE, "rt") as f:
    for line in f:
        if line.startswith("#"):
            continue

        fields = line.rstrip().split("\t")

        if len(fields) < 8:
            continue

        (
            chrom,
            pos,
            vid,
            ref,
            alt,
            qual,
            filt,
            info_string,
        ) = fields[:8]

        info = parse_info(info_string)
        csq = parse_csq(info)

        rows.append(
            {
                "Sample_id": info.get("sample"),
                "HGNC": info.get("gene"),
                "Var_type": info.get("effect"),
                "Cohort": info.get("cohort"),
                "Chr": chrom,
                "Position": int(pos),
                "Ref": ref,
                "Alt": alt,
                "GeneEff": csq.get("Consequence"),
                "HGVSc": csq.get("HGVSc"),
                "HGVSp": csq.get("HGVSp"),
                "AlphaMissense": info.get(
                    "am_pathogenicity"
                ),
                "REVEL": info.get("REVEL"),
                "MisFit_S": info.get("MisFit1.5_S"),
                "MisFit_D": info.get("MisFit1.5_D"),
                "ESM": info.get("ESM1b_LLR"),
                "CADD": info.get("CADD_phred"),
                "s_het": info.get("Shet"),
                "gnomAD4_AF": info.get(
                    "gnomad4j_AF"
                ),
            }
        )


df = pd.DataFrame(rows)


# ==========================================================
# Convert numeric columns
# ==========================================================

numeric_columns = [
    "AlphaMissense",
    "REVEL",
    "MisFit_S",
    "MisFit_D",
    "ESM",
    "CADD",
    "s_het",
    "gnomAD4_AF",
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )


# ==========================================================
# Normalize Sample_id
# ==========================================================

df["Sample_id"] = (
    df["Sample_id"]
    .astype("string")
    .str.strip()
)


# ==========================================================
# Split cohorts
# ==========================================================

eatef = df[
    df["Cohort"] == "EATEF"
].copy()

spark = df[
    df["Cohort"] == "SPARK"
].copy()


# ==========================================================
# Load syndromic classification from old Table S2
# ==========================================================

print()
print("Loading syndromic classification from:")
print(f"  {TABLE_S2_FILE}")

table_s2 = read_table_s2(
    TABLE_S2_FILE
)


# Remove leading or trailing whitespace from headers
table_s2.columns = (
    table_s2.columns
    .astype(str)
    .str.strip()
)

print()
print("Table S2 columns:")
print(table_s2.columns.tolist())


# ==========================================================
# Find the syndromic column
# ==========================================================

possible_syndromic_columns = [
    "Syndormic or not",
    "Syndromic or not",
    "Syndormic",
    "Syndromic",
]

syndromic_source_column = None

for column in possible_syndromic_columns:
    if column in table_s2.columns:
        syndromic_source_column = column
        break

if syndromic_source_column is None:
    raise ValueError(
        "Could not find a syndromic classification "
        "column in Table S2. Tried: "
        + ", ".join(possible_syndromic_columns)
    )

if "Sample_id" not in table_s2.columns:
    raise ValueError(
        "Table S2 does not contain a Sample_id column."
    )


print()
print(
    "Using Table S2 column:",
    syndromic_source_column,
)


# ==========================================================
# Build one classification per Sample_id
# ==========================================================

syndromic_lookup = table_s2[
    [
        "Sample_id",
        syndromic_source_column,
    ]
].copy()


syndromic_lookup["Sample_id"] = (
    syndromic_lookup["Sample_id"]
    .astype("string")
    .str.strip()
)

syndromic_lookup[
    syndromic_source_column
] = (
    syndromic_lookup[
        syndromic_source_column
    ]
    .astype("string")
    .str.strip()
)


# Convert empty strings and text representations
# of missing values to proper missing values.
syndromic_lookup[
    syndromic_source_column
] = (
    syndromic_lookup[
        syndromic_source_column
    ]
    .replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "NA": pd.NA,
            "None": pd.NA,
            "<NA>": pd.NA,
        }
    )
)


# Remove rows without a usable sample ID
syndromic_lookup = (
    syndromic_lookup
    .dropna(subset=["Sample_id"])
)

syndromic_lookup = syndromic_lookup[
    syndromic_lookup["Sample_id"] != ""
].copy()


# ==========================================================
# Check for conflicting classifications
# ==========================================================

classification_counts = (
    syndromic_lookup
    .dropna(
        subset=[
            syndromic_source_column
        ]
    )
    .groupby("Sample_id")[
        syndromic_source_column
    ]
    .nunique()
)

conflicting_samples = (
    classification_counts[
        classification_counts > 1
    ]
)

if not conflicting_samples.empty:
    print()
    print(
        "WARNING: Some Sample_id values have "
        "multiple syndromic classifications:"
    )

    print(
        conflicting_samples.to_string()
    )

    print()
    print(
        "The first non-missing classification "
        "will be used for those samples."
    )


# ==========================================================
# Collapse Table S2 to one row per sample
# ==========================================================

def first_non_missing(series):
    non_missing = series.dropna()

    if len(non_missing) == 0:
        return pd.NA

    return non_missing.iloc[0]


syndromic_lookup = (
    syndromic_lookup
    .groupby(
        "Sample_id",
        as_index=False,
    )[
        syndromic_source_column
    ]
    .agg(first_non_missing)
)


# Rename the output column to the exact requested name
syndromic_lookup = (
    syndromic_lookup.rename(
        columns={
            syndromic_source_column:
                "Syndormic or not"
        }
    )
)


# ==========================================================
# Match EA/TEF variants by Sample_id
# ==========================================================

original_eatef_columns = (
    eatef.columns.tolist()
)

original_eatef_row_count = len(eatef)

eatef = eatef.merge(
    syndromic_lookup,
    on="Sample_id",
    how="left",
    validate="many_to_one",
)


if len(eatef) != original_eatef_row_count:
    raise RuntimeError(
        "The Sample_id merge changed the number "
        "of EA/TEF rows."
    )


# Keep all original columns in their original order
# and place the new column at the end.
eatef = eatef[
    original_eatef_columns
    + ["Syndormic or not"]
]


# SPARK controls do not receive an EA/TEF
# syndromic classification.
spark["Syndormic or not"] = pd.NA

original_spark_columns = [
    column
    for column in original_eatef_columns
    if column in spark.columns
]

spark = spark[
    original_spark_columns
    + ["Syndormic or not"]
]


# ==========================================================
# Save outputs
# ==========================================================

os.makedirs(
    "results",
    exist_ok=True,
)

eatef.to_csv(
    EATEF_OUTPUT_FILE,
    sep="\t",
    index=False,
)

spark.to_csv(
    SPARK_OUTPUT_FILE,
    sep="\t",
    index=False,
)


# ==========================================================
# Print summary
# ==========================================================

matched_rows = (
    eatef["Syndormic or not"]
    .notna()
    .sum()
)

unmatched_rows = (
    eatef["Syndormic or not"]
    .isna()
    .sum()
)

matched_samples = (
    eatef.loc[
        eatef[
            "Syndormic or not"
        ].notna(),
        "Sample_id",
    ]
    .nunique()
)

unmatched_sample_ids = (
    eatef.loc[
        eatef[
            "Syndormic or not"
        ].isna(),
        "Sample_id",
    ]
    .dropna()
    .drop_duplicates()
    .sort_values()
)


print()
print("EA/TEF variants:", len(eatef))
print(
    "EA/TEF samples:",
    eatef["Sample_id"].nunique(),
)

print()
print(
    "EA/TEF variants with syndromic "
    "classification:",
    matched_rows,
)

print(
    "EA/TEF variants without syndromic "
    "classification:",
    unmatched_rows,
)

print(
    "EA/TEF samples with syndromic "
    "classification:",
    matched_samples,
)

print()
print(
    "Syndromic classification counts "
    "by variant:"
)

print(
    eatef["Syndormic or not"]
    .fillna("Missing")
    .value_counts(dropna=False)
    .to_string()
)


if len(unmatched_sample_ids) > 0:
    print()
    print(
        "EA/TEF Sample_id values not matched "
        "to Table S2:"
    )

    print(
        unmatched_sample_ids
        .head(50)
        .to_string(index=False)
    )

    if len(unmatched_sample_ids) > 50:
        print(
            f"... and "
            f"{len(unmatched_sample_ids) - 50} "
            "more unmatched samples."
        )


print()
print("SPARK variants:", len(spark))
print(
    "SPARK samples:",
    spark["Sample_id"].nunique(),
)

print()
print("Saved:")
print(f"  {EATEF_OUTPUT_FILE}")
print(f"  {SPARK_OUTPUT_FILE}")

