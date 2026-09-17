import os
import re
from collections import defaultdict

import pandas as pd

MUTRATE_FILE = "data/hg38_GeneCodeV41_VariantRate_anno.tsv.gz"
UNIPROT_FILE = "data/content/ALL_hum_isoforms_ESM1b_LLR/000_uniprot_df.csv"
ESM_FOLDER = "data/content/ALL_hum_isoforms_ESM1b_LLR"

OUTPUT_FILE = "results/esm_mutation_rate.tsv"

THRESHOLD = -7.5
CHUNK_SIZE = 2_000_000

AA_REGEX = re.compile(r"(\d+)([A-Z])>(\d+)([A-Z*])")

print("Loading UniProt mapping...")

uniprot = (
    pd.read_csv(UNIPROT_FILE)
    .rename(columns={"gene": "SYMBOL", "id": "UniProt"})
)

uniprot_dict = dict(zip(uniprot["SYMBOL"], uniprot["UniProt"]))

gene_totals = defaultdict(float)
esm_cache = {}

reader = pd.read_csv(
    MUTRATE_FILE,
    sep="\t",
    compression="gzip",
    chunksize=CHUNK_SIZE,
    low_memory=False,
    usecols=[
        "SYMBOL",
        "VariantFunc",
        "MutationRate",
        "AminoAcidChange",
    ],
)

print("Calculating ESM mutation rates...")

for chunk_number, chunk in enumerate(reader, start=1):

    print(f"Chunk {chunk_number}")

    chunk = chunk.loc[chunk["VariantFunc"] == "missense"].copy()

    if chunk.empty:
        continue

    chunk["MutationRate"] = pd.to_numeric(
        chunk["MutationRate"],
        errors="coerce",
    )

    chunk["UniProt"] = chunk["SYMBOL"].map(uniprot_dict)

    chunk = chunk.dropna(
        subset=[
            "MutationRate",
            "UniProt",
            "AminoAcidChange",
        ]
    )

    chunk = chunk.sort_values("UniProt")

    for row in chunk.itertuples(index=False):

        match = AA_REGEX.match(str(row.AminoAcidChange))

        if match is None:
            continue

        position = int(match.group(1))
        ref = match.group(2)
        alt = match.group(4)

        if alt == "*":
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
            continue

        column = f"{ref} {position}"

        if column not in esm.columns:
            continue

        if alt not in esm.index:
            continue

        score = esm.at[alt, column]

        if pd.isna(score):
            continue

        if score <= THRESHOLD:
            gene_totals[row.SYMBOL] += row.MutationRate

esm_rates = (
    pd.DataFrame(
        gene_totals.items(),
        columns=[
            "HGNC",
            "Dmis_ESM_7.5",
        ],
    )
    .sort_values("HGNC")
)

os.makedirs("results", exist_ok=True)

esm_rates.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False,
)

print()
print("Finished")
print(f"Genes: {len(esm_rates)}")
print(f"Saved to {OUTPUT_FILE}")