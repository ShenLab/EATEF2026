#!/usr/bin/env python3
"""
Gene-level case-control burden test for EA/TEF risk genes.

Workflow (per gene)
-------------------
LoF burden
    one-sided mid-p binomial test (cases vs controls)          -> P_LoF
D-Mis burden
    one-sided mid-p binomial test at MisFit_D >= 0.3            -> P_0.3
    one-sided mid-p binomial test at MisFit_D >= 0.4            -> P_0.4
    Cauchy combination test (P_0.3, P_0.4)                      -> P_DMis
Fisher's method (P_LoF, P_DMis)                                 -> P_Fisher
Cauchy combination test (P_Fisher, P_DMis)                      -> P_gene
Benjamini-Hochberg FDR over all protein-coding genes (M)        -> FDR

Gene exclusions
---------------
The binomial test treats every counted variant as an independent
observation.  A gene whose case count comes from a single variant shared
by several related individuals of one family violates that assumption and
would receive a spurious significance.  Such genes are removed with
--exclude-genes (in the EA/TEF analysis: KCTD3, variants carried by
members of a single family).

Under the null the probability that a variant observed in the pooled sample
comes from a case is p = N_case / (N_case + N_ctrl); the mid-p binomial test
asks whether the number of case variants exceeds that expectation.

Input
-----
Per-gene count table (TSV) with at least these columns:
    Gene, CaseLofN, CtrlLofN,
    case_MisFit_D_03N, control_MisFit_D_03N,
    case_MisFit_D_04N, control_MisFit_D_04N

Usage
-----
python risk_gene_burden_test.py \
    --counts worst_case_control_counts_by_gene.tsv \
    --n-case 401 --n-ctrl 47419 --n-genes 18225 \
    --out risk_gene_burden_results.tsv
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import binom, combine_pvalues
from statsmodels.stats.multitest import multipletests

DMIS_THRESHOLDS = ["03", "04"]  # MisFit_D >= 0.3, >= 0.4


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def mid_p_binom(case_n: int, ctrl_n: int, p_case: float) -> float:
    """One-sided mid-p binomial test: P(X > case_n) + 0.5 * P(X = case_n),
    X ~ Binomial(case_n + ctrl_n, p_case)."""
    n = case_n + ctrl_n
    if n == 0:
        return 1.0
    return 1.0 - binom.cdf(case_n - 1, n, p_case) - 0.5 * binom.pmf(case_n, n, p_case)


def cauchy_combination(pvals) -> float:
    """Cauchy combination test (Liu & Xie 2020), equal weights."""
    p = np.clip(np.asarray(pvals, dtype=float), 1e-16, 1 - 1e-16)
    stat = np.mean(np.tan(np.pi * (0.5 - p)))
    return 0.5 - np.arctan(stat) / np.pi


def fisher_combination(pvals) -> float:
    return combine_pvalues(np.asarray(pvals, dtype=float), method="fisher")[1]


def relative_risk(case_n, ctrl_n, n_case, n_ctrl) -> float:
    if ctrl_n == 0:
        return np.inf if case_n > 0 else np.nan
    return (case_n / n_case) / (ctrl_n / n_ctrl)


# ----------------------------------------------------------------------------
# per-gene test
# ----------------------------------------------------------------------------
def test_gene(row: pd.Series, p_case: float, n_case: int, n_ctrl: int) -> pd.Series:
    out = {}

    # LoF
    lof_case, lof_ctrl = int(row["CaseLofN"]), int(row["CtrlLofN"])
    out["LoF_RR"] = relative_risk(lof_case, lof_ctrl, n_case, n_ctrl)
    out["P_LoF"] = mid_p_binom(lof_case, lof_ctrl, p_case)

    # D-Mis at each MisFit_D threshold
    p_thr = []
    for t in DMIS_THRESHOLDS:
        c = int(row[f"case_MisFit_D_{t}N"])
        k = int(row[f"control_MisFit_D_{t}N"])
        out[f"DMis_RR_{t}"] = relative_risk(c, k, n_case, n_ctrl)
        out[f"P_{t}"] = mid_p_binom(c, k, p_case)
        p_thr.append(out[f"P_{t}"])

    out["P_DMis"] = cauchy_combination(p_thr)
    out["P_Fisher"] = fisher_combination([out["P_LoF"], out["P_DMis"]])
    # NOTE: per the study design P_DMis enters twice (inside P_Fisher and again
    # here); this up-weights the missense signal relative to LoF.
    out["P_gene"] = cauchy_combination([out["P_Fisher"], out["P_DMis"]])
    return pd.Series(out)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", required=True, help="per-gene case/control count table (TSV)")
    ap.add_argument("--out", required=True, help="output TSV")
    ap.add_argument("--n-case", type=int, required=True, help="number of cases")
    ap.add_argument("--n-ctrl", type=int, required=True, help="number of controls")
    ap.add_argument("--n-genes", type=int, default=18225,
                    help="number of protein-coding genes for BH correction; genes absent "
                         "from the table are padded with p=1 (default 18225)")
    ap.add_argument("--exclude-genes", nargs="*", default=[],
                    help="genes to drop before testing because their case variants are not "
                         "independent observations (single variant shared within one family); "
                         "EA/TEF analysis: KCTD3")
    args = ap.parse_args()

    p_case = args.n_case / (args.n_case + args.n_ctrl)

    genes = pd.read_csv(args.counts, sep="\t")
    genes = genes.drop_duplicates("Gene")
    if args.exclude_genes:
        dropped = genes.loc[genes["Gene"].isin(args.exclude_genes), "Gene"].tolist()
        genes = genes[~genes["Gene"].isin(args.exclude_genes)]
        print(f"excluded (related carriers within one family): {', '.join(dropped) or 'none found'}")

    res = genes.apply(test_gene, axis=1, p_case=p_case, n_case=args.n_case, n_ctrl=args.n_ctrl)
    genes = pd.concat([genes.reset_index(drop=True), res.reset_index(drop=True)], axis=1)

    # BH-FDR over the full gene universe: untested genes contribute p = 1
    n_pad = max(args.n_genes - len(genes), 0)
    p_full = np.concatenate([genes["P_gene"].values, np.ones(n_pad)])
    genes["FDR"] = multipletests(p_full, method="fdr_bh")[1][: len(genes)]

    genes = genes.sort_values(["FDR", "P_gene"]).reset_index(drop=True)
    genes.to_csv(args.out, sep="\t", index=False)

    print(f"cases={args.n_case} controls={args.n_ctrl} p_case={p_case:.5f}")
    print(f"genes tested={len(genes)} padded to M={args.n_genes}")
    print(f"FDR<0.05: {(genes['FDR'] < 0.05).sum()}   FDR<0.1: {(genes['FDR'] < 0.1).sum()}")
    print(genes[["Gene", "CaseLofN", "CtrlLofN", "P_LoF", "P_DMis", "P_Fisher", "P_gene", "FDR"]]
          .head(15).to_string(index=False))


if __name__ == "__main__":
    main()