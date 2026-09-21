# Rare variant burden and pathway enrichment analysis for EA/TEF

Scripts for case-control burden testing of ultra-rare variants to identify candidate risk genes, and pathway enrichment analysis of de novo variants. Two independent analyses with separate data requirements and workflows.

Run every command from the repository root. Paths and analysis parameters are hardcoded near the top of each script; edit those values before running.

## Environment

Python >= 3.9

packages: `pandas`, `numpy`, `scipy`, `matplotlib`, `adjustText`, `gseapy`, `networkx`, `openpyxl`, `statsmodels`

The GO subtree figure also needs Graphviz (`dot` on `PATH`):

```bash
conda install -c conda-forge graphviz
```

## Data Files

**Included in data folder:**
- `TS2_Denovo_Variants_allCases_anno.tsv` — De novo variants for pathway enrichment analysis (DNV only)
- `gene_mutation_rate.new.txt` — Per-gene, per-consequence mutation rates for pathway enrichment
- `c5.all.v2024.1.Hs.symbols.gmt` — MSigDB Gene Ontology terms (symbols)
- `09GENCODEV19_blacklist.txt` — Gene blacklist for pathway analysis

**Additional required files** (prepare and place in `data/`):
- `go-basic.obo` — Gene Ontology structure (download from [Gene Ontology](http://geneontology.org/))

Default settings: 161 complex cases (pathway), 401 unrelated cases + 47,419 unaffected parental controls from SPARK (burden), MisFit-S threshold 0.005 for damaging missense, 1,000,000 permutations.

---

## Rare variant case-control burden test (RISK GENES)

**Purpose**: Identify genes with significant enrichment of LoF and/or damaging missense ultra-rare variants in 401 unrelated EA/TEF cases vs 47,419 unaffected unrelated parental controls from SPARK cohort.

**Input**: Per-gene ultra-rare variant (allele frequency < 1e-5) counts:
- Count table (TSV) with columns:
  - Gene name
  - CaseLofN, CtrlLofN — LoF variant counts
  - case_MisFit_D_03N, control_MisFit_D_03N — Damaging missense at MisFit_D ≥ 0.3
  - case_MisFit_D_04N, control_MisFit_D_04N — Damaging missense at MisFit_D ≥ 0.4

**Workflow**:

Prepare per-gene ultra-rare variant counts from case and control cohorts. Remove genes with variants carried by single samples (violating independence assumption) before this step.

```bash
python risk_gene_analysis.py \
    --counts <your_counts_table.tsv> \
    --n-case <N_cases> --n-ctrl <N_controls> \
    --n-genes <total_protein_coding_genes> \
    --out risk_gene_burden_results.tsv
```

**Statistical methods**:
- Per-gene: mid-p binomial test (LoF and damaging missense at each threshold)
- Combines thresholds: Cauchy combination test
- Integrates LoF + missense signals: Fisher's method + Cauchy combination
- Multiple testing correction: Benjamini-Hochberg FDR across 18,225 protein-coding genes

**Pre-filtering**: Remove genes with variants carried by single samples (violating independence assumption) before preparing the count table. This ensures each variant is treated as independent observation.

**Output**: `risk_gene_burden_results.tsv`
- Genes ranked by FDR-adjusted q-value
- Columns: Gene, case/control counts (LoF, MisFit_D thresholds), relative risk per variant class, p-values (LoF, each MisFit_D threshold, combined missense, integrated gene-level), FDR
- Candidate risk genes identified at FDR < threshold

---

## Pathway enrichment of de novo variants (DNV only)

**Purpose**: Identify biological pathways enriched for LoF and/or damaging missense de novo variants in 161 complex EA/TEF cases.

**Input**: De novo variants from cases only (no controls in this analysis):
- `data/TS2_Denovo_Variants_allCases_anno.tsv` — Case DNVs with gene name (HGNC), variant type (Var_type), gene consequence (GeneEff), damaging missense score (MisFit_S)
- `data/gene_mutation_rate.new.txt` — Background mutation rates per gene per consequence type
- `data/c5.all.v2024.1.Hs.symbols.gmt` — Gene Ontology pathway definitions
- `data/09GENCODEV19_blacklist.txt` — Genes to exclude from analysis

**Workflow**:

```bash
python scripts/common_pathways.py
python scripts/lof_dmis/pathway_enrichment_lof_dmis.py
python scripts/lof_dmis/permutation_lof_dmis.py
python scripts/lof_dmis/adjust_pvalues_lof_dmis.py
```

**Statistical methods**:
- Per-pathway: observed vs expected LoF and damaging missense counts
- Expected counts derived from background mutation rates scaled by case number and gene length
- Poisson test for descriptive p-values
- Permutation test (1,000,000 iterations) for empirical null distribution
- FWER-adjusted p-values across all pathways

---

## Post-analysis: Figures and visualization

After both analyses are complete, generate figures and post-hoc results:

```bash
python scripts/pathway_similarity_network.py
python scripts/compare_significant_pathways_lof_dmis.py
python scripts/post_analysis_go_redundancy.py
python scripts/extract_significant_pathway_subtree.py
python scripts/build_figure1c_pathway_table.py
python scripts/metabolic_exclude_transport_autophagy.py
```

Requires `results/significant_pathway_union_lof_vs_dmis.tsv`. Run `scripts/pathway_similarity_network.py` before `scripts/compare_significant_pathways_lof_dmis.py`.

---