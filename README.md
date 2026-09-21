# EA/TEF DNV pathway analysis

Scripts for annotating esophageal atresia / tracheoesophageal fistula (EA/TEF) de novo variants, testing gene-set enrichment, and generating pathway figures. Run every command from the repository root. Paths and analysis parameters are hardcoded near the top of each script; edit those values before running.

## Environment

Python >= 3.9

packages: `pandas`, `numpy`, `scipy`, `matplotlib`, `adjustText`, `gseapy`, `networkx`, `openpyxl`

The GO subtree figure also needs Graphviz (`dot` on `PATH`):

```bash
conda install -c conda-forge graphviz
```

Place input files under `data/`. The MSigDB GO GMT currently points to `/share/vault/Users/xl3126/EATEF/data/c5.all.v2024.1.Hs.symbols.gmt`; change `GMT_FILE` in `scripts/common_pathways.py` and `scripts/common_pathway_universe_spark.py` (or copy the file to `data/c5.all.v2024.1.Hs.symbols.gmt`). Download `data/go-basic.obo` from [Gene Ontology](http://geneontology.org/).

Expected data files:

- `data/EATEF_SPARK-unaff-sibs.ann.vcf.gz`
- `data/TableS2.Denovo_Variants_allCases_burden.txt`
- `data/ensp_to_uniprot.tsv`
- `data/content/ALL_hum_isoforms_ESM1b_LLR/`
- `data/hg38_GeneCodeV41_VariantRate_anno.tsv.gz`
- `data/gene_mutation_rate.new.txt`
- `data/09GENCODEV19_blacklist.txt`
- `data/c5.all.v2024.1.Hs.symbols.gmt`
- `data/go-basic.obo`
- `data/5_Supplementary_Datasets_MisFit1.5.1.xlsx`

Default analysis settings: 161 complex cases, MisFit-S damaging-missense threshold `0.005`, 1,000,000 permutations (`RANDOM_SEED = 42`).

## Usage

### 1. Annotate variants and build mutation rates

Modify input paths in `scripts/eatef/parse_vcf.py` if your VCF or Table S2 files differ.

Then run those scripts in command line step by step:

```bash
python scripts/eatef/parse_vcf.py
python scripts/eatef/annotate_missense.py
python scripts/eatef/calculate_esm_mutation_rate.py
python scripts/eatef/merge_mutation_rates.py
python scripts/eatef/eatef_burden_thresholds.py
python scripts/eatef/plot_eatef_thresholds.py
```

### 2. Model-based LoF+Dmis pathway enrichment

Modify the DNV table, mutation-rate table, GMT, and blacklist in `scripts/common_pathways.py`. The same `results/common_pathway_universe.tsv` must be used by enrichment, permutation, and FWER adjustment.

```bash
python scripts/common_pathways.py
python scripts/lof_dmis/pathway_enrichment_lof_dmis.py
python scripts/lof_dmis/permutation_lof_dmis.py
python scripts/lof_dmis/adjust_pvalues_lof_dmis.py
```

`permutation_lof_dmis.py` runs 1,000,000 permutations and is slow. Change `N_PERMUTATIONS` in the script if you need a shorter test run.

### 3. SPARK-control background sensitivity analysis

This parallel analysis replaces model-based gene mutation rates with empirical rates from SPARK unaffected siblings (Dataset S3; default `N_Controls = 9789`).

```bash
python scripts/build_spark_control_gene_rates.py
python scripts/common_pathway_universe_spark.py
python scripts/lof_spark/pathway_enrichment_lof_spark.py
python scripts/lof_spark/pathway_permutation_lof_spark.py
python scripts/lof_spark/pathway_adjust_pvalues_lof_spark.py
python scripts/lof_dmis_spark/pathway_enrichment_lof_dmis_spark.py
python scripts/lof_dmis_spark/pathway_permutation_lof_dmis_spark.py
python scripts/lof_dmis_spark/pathway_adjust_pvalues_lof_dmis_spark.py
python scripts/compare_og_vs_spark_pathways.py
```

`compare_og_vs_spark_pathways.py` also looks for LoF-only and Dmis-only adjusted tables (`results/pathway_enrichment_adjusted_lof.tsv`, `results/pathway_enrichment_adjusted_misfit.tsv`, and the SPARK Dmis file). Those companion analyses are not in this repository; comment those entries out of `FILES` if you only ran LoF+Dmis and SPARK LoF / LoF+Dmis.

### 4. Figures and post-analysis

These scripts expect `results/significant_pathway_union_lof_vs_dmis.tsv` (one row per selected pathway, with LoF / Dmis / LoF+Dmis FWER columns). Run `scripts/pathway_similarity_network.py` before `scripts/compare_significant_pathways_lof_dmis.py`.

```bash
python scripts/pathway_similarity_network.py
python scripts/compare_significant_pathways_lof_dmis.py
python scripts/post_analysis_go_redundancy.py
python scripts/extract_significant_pathway_subtree.py
python scripts/build_figure1c_pathway_table.py
python scripts/metabolic_exclude_transport_autophagy.py
```

`post_analysis_go_redundancy.py` and `metabolic_exclude_transport_autophagy.py` recalculate descriptive Poisson p-values. They do not recompute empirical FWER.

## Output

### Variant annotation and burden

- `results/EA_TEF_cases_annotated.tsv` and `results/SPARK_controls_annotated.tsv`
- `results/variants_with_esm.tsv`
- `results/esm_mutation_rate.tsv`
- `results/gene_mutation_rate_with_esm.tsv`
- `results/eatef_burden_results_complex.tsv`
- `results/eatef_threshold_optimization.png` and `.pdf`

### Model-based LoF+Dmis enrichment

- `results/common_pathway_universe.tsv`
- `results/pathway_enrichment_lof_dmis.tsv`
- `results/synonymous_scale_factor_lof_dmis.tsv`
- `results/permutation_best_pvalues_lof_dmis.tsv`
- `results/pathway_enrichment_adjusted_lof_dmis.tsv`

### SPARK-control background

- `results/control_background/SPARK_gene_DNV_rates.tsv` and `SPARK_control_variant_qc.tsv`
- `results/control_background/common_pathway_universe_spark.tsv`
- `results/control_background/pathway_enrichment_*_spark.tsv`
- `results/control_background/permutation_best_pvalues_*_spark.tsv`
- `results/control_background/pathway_enrichment_adjusted_*_spark.tsv`
- `results/control_background/pathway_model_vs_spark_comparison.tsv`
- `results/control_background/pathway_model_vs_spark_significance_summary.tsv`

### Figures and post-analysis

- `results/pathway_network_nodes.tsv` and `results/pathway_network_edges.tsv` can be used as input to Cytoscape for visualization and clustering
- `results/pathway_network_clusters.tsv`, `.pdf`, and `.png`
- `results/significant_pathway_union_lof_vs_dmis.tsv`, `.pdf`, and `.png`
- `results/post_analysis_go_redundancy_results.tsv`
- `results/significant_pathway_go_subtree.pdf`, `.png`, and `.svg`
- `results/Figure1C_pathway_statistics_and_DNV_genes.tsv`
- `results/metabolic_enrichment_excluding_transport_autophagy.tsv`
- `results/metabolic_overlap_excluded_genes.tsv`
