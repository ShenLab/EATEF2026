import pandas as pd

# ----------------------------
# Load original mutation-rate table
# ----------------------------
mutation_rate = pd.read_csv(
    "data/gene_mutation_rate.new.txt",
    sep="\t"
)

# ----------------------------
# Load ESM mutation rates
# ----------------------------
esm = pd.read_csv(
    "results/esm_mutation_rate.tsv",
    sep="\t"
)

# ----------------------------
# Make column names consistent
# ----------------------------
esm = esm.rename(columns={"HGNC": "Gene_name"})

# ----------------------------
# Merge
# ----------------------------
merged = mutation_rate.merge(
    esm,
    on="Gene_name",
    how="left"
)

# ----------------------------
# Missing genes get mutation rate 0
# ----------------------------
merged["Dmis_ESM_7.5"] = merged["Dmis_ESM_7.5"].fillna(0)

# ----------------------------
# Save
# ----------------------------
merged.to_csv(
    "results/gene_mutation_rate_with_esm.tsv",
    sep="\t",
    index=False,
)

print("Finished!")
print(f"Genes: {len(merged)}")
print("Saved to results/gene_mutation_rate_with_esm.tsv")