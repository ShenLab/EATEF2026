
#!/usr/bin/env python3
"""
Build empirical gene-level de novo variant rates from Dataset S3
(SPARK unaffected siblings) in 5_Supplementary_Datasets_MisFit1.5.1.xlsx.


The output can replace model-based gene mutation rates in a parallel
SPARK-control sensitivity analysis.


Dataset S3 structure used by this script
----------------------------------------
Row 1: dataset title
Rows 2-3: two-level column header
Data begin on row 4


Relevant columns:
- Sample ID
- Gene name
- DNV effect
- MisFit-S / v1.5.1
- LOFTEE / Confidence
- In reliable / region


Outputs
-------
results/control_background/SPARK_gene_DNV_rates.tsv
results/control_background/SPARK_control_variant_qc.tsv


The gene-rate output contains:
Gene
Control_LoF_Count
Control_Dmis_Count
Control_LoF_Dmis_Count
Control_LoF_Rate
Control_Dmis_Rate
Control_LoF_Dmis_Rate
N_Controls
MisFit_Threshold


Definitions
-----------
Dmis:
   DNV effect == "Missense"
   MisFit-S v1.5.1 >= 0.005
   variant is in a reliable region, when that column is available


LoF:
   DNV effect == "LoF"
   variant is in a reliable region, when that column is available


By default, LoF is not restricted to LOFTEE HC because the primary EA/TEF
pathway analysis should determine the matching LoF definition. Set
--require-loftee-hc if the case analysis uses only LOFTEE high-confidence LoF.
"""


from __future__ import annotations


import argparse
import re
from pathlib import Path


import numpy as np
import pandas as pd




DEFAULT_INPUT = "data/5_Supplementary_Datasets_MisFit1.5.1.xlsx"
DEFAULT_SHEET = "Dataset S3"
DEFAULT_OUTPUT = "results/control_background/SPARK_gene_DNV_rates.tsv"
DEFAULT_QC_OUTPUT = "results/control_background/SPARK_control_variant_qc.tsv"


DEFAULT_N_CONTROLS = 9789
DEFAULT_MISFIT_THRESHOLD = 0.005


BLACKLIST_CANDIDATES = [
   "data/09GENCODEV19_blacklist.txt",
   "data/blacklisted_genes.txt",
   "results/blacklisted_genes.txt",
   "scripts/data/blacklisted_genes.txt",
]




def normalize_text(value: object) -> str:
   """Normalize a value for case-insensitive matching."""
   if pd.isna(value):
       return ""
   return re.sub(r"\s+", " ", str(value).strip())




def normalize_gene(value: object) -> str:
   """Normalize HGNC gene symbols."""
   return normalize_text(value).upper()




def flatten_column(column: object) -> str:
   """
   Flatten the two-row Excel header.


   Examples:
     ("Gene name", "Unnamed: ...") -> "Gene name"
     ("MisFit-S", "v1.5.1")        -> "MisFit-S v1.5.1"
     ("In reliable", "region")      -> "In reliable region"
   """
   if not isinstance(column, tuple):
       return normalize_text(column)


   pieces: list[str] = []
   for piece in column:
       text = normalize_text(piece)
       if not text:
           continue
       if text.lower().startswith("unnamed:"):
           continue
       if text not in pieces:
           pieces.append(text)


   return " ".join(pieces)




def find_column(
   columns: list[str],
   candidates: list[str],
   *,
   required: bool = True,
) -> str | None:
   """Find a column using exact normalized matches, then substring matches."""
   normalized = {
       re.sub(r"[^a-z0-9]+", "", column.lower()): column
       for column in columns
   }


   for candidate in candidates:
       key = re.sub(r"[^a-z0-9]+", "", candidate.lower())
       if key in normalized:
           return normalized[key]


   for candidate in candidates:
       key = re.sub(r"[^a-z0-9]+", "", candidate.lower())
       matches = [
           original
           for normalized_name, original in normalized.items()
           if key and key in normalized_name
       ]
       if len(matches) == 1:
           return matches[0]


   if required:
       raise ValueError(
           "Could not identify a required column from "
           f"{candidates}. Available columns: {columns}"
       )
   return None




def parse_boolean(series: pd.Series) -> pd.Series:
   """Interpret Excel booleans, 0/1 values, and common text labels."""
   numeric = pd.to_numeric(series, errors="coerce")
   result = numeric.eq(1)


   text = series.astype(str).str.strip().str.lower()
   result = result | text.isin(
       {"true", "t", "yes", "y", "reliable", "1.0"}
   )
   return result.fillna(False)




def load_blacklist(explicit_path: str | None) -> tuple[set[str], str | None]:
   """Load a one-gene-per-line blacklist, if available."""
   candidates = [explicit_path] if explicit_path else BLACKLIST_CANDIDATES


   for candidate in candidates:
       if not candidate:
           continue
       path = Path(candidate)
       if not path.exists():
           continue


       genes: set[str] = set()
       with path.open("r", encoding="utf-8", errors="replace") as handle:
           for line in handle:
               stripped = line.strip()
               if not stripped or stripped.startswith("#"):
                   continue
               first_field = re.split(r"[\t,\s]+", stripped)[0]
               gene = normalize_gene(first_field)
               if gene:
                   genes.add(gene)


       return genes, str(path)


   return set(), None




def load_dataset_s3(input_path: Path, sheet_name: str) -> pd.DataFrame:
   """
   Read Dataset S3 using its two-row header.


   header=[1, 2] means Excel rows 2 and 3 are treated as the column header.
   """
   workbook = pd.ExcelFile(input_path)
   if sheet_name not in workbook.sheet_names:
       raise ValueError(
           f'Sheet "{sheet_name}" was not found. '
           f"Available sheets: {workbook.sheet_names}"
       )


   df = pd.read_excel(
       input_path,
       sheet_name=sheet_name,
       header=[1, 2],
       engine="openpyxl",
   )
   df.columns = [flatten_column(column) for column in df.columns]


   # Remove fully empty rows and columns.
   df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
   return df




def main() -> None:
   parser = argparse.ArgumentParser(
       description=(
           "Build empirical gene-level LoF and Dmis rates from "
           "SPARK unaffected-sibling de novo variants."
       )
   )
   parser.add_argument(
       "--input",
       default=DEFAULT_INPUT,
       help=f"Input Excel workbook (default: {DEFAULT_INPUT})",
   )
   parser.add_argument(
       "--sheet",
       default=DEFAULT_SHEET,
       help=f"Control DNV sheet (default: {DEFAULT_SHEET})",
   )
   parser.add_argument(
       "--output",
       default=DEFAULT_OUTPUT,
       help=f"Gene-rate output TSV (default: {DEFAULT_OUTPUT})",
   )
   parser.add_argument(
       "--qc-output",
       default=DEFAULT_QC_OUTPUT,
       help=f"Variant-level QC output TSV (default: {DEFAULT_QC_OUTPUT})",
   )
   parser.add_argument(
       "--n-controls",
       type=int,
       default=DEFAULT_N_CONTROLS,
       help=(
           "Number of SPARK unaffected siblings "
           f"(default: {DEFAULT_N_CONTROLS})"
       ),
   )
   parser.add_argument(
       "--misfit-threshold",
       type=float,
       default=DEFAULT_MISFIT_THRESHOLD,
       help=(
           "MisFit-S v1.5.1 threshold for Dmis "
           f"(default: {DEFAULT_MISFIT_THRESHOLD})"
       ),
   )
   parser.add_argument(
       "--blacklist",
       default=None,
       help="Optional explicit blacklist path.",
   )
   parser.add_argument(
       "--require-loftee-hc",
       action="store_true",
       help=(
           "Restrict LoF controls to LOFTEE Confidence == HC. "
           "Use only if the primary case LoF definition also requires HC."
       ),
   )
   parser.add_argument(
       "--ignore-reliable-region",
       action="store_true",
       help=(
           "Do not require the Dataset S3 'In reliable region' flag. "
           "The default is to require it when the column exists."
       ),
   )
   args = parser.parse_args()


   if args.n_controls <= 0:
       raise ValueError("--n-controls must be greater than zero.")


   input_path = Path(args.input)
   if not input_path.exists():
       raise FileNotFoundError(f"Input workbook not found: {input_path}")


   df = load_dataset_s3(input_path, args.sheet)
   columns = list(df.columns)


   sample_column = find_column(
       columns,
       ["Sample ID", "Sample_ID", "Sample"],
   )
   gene_column = find_column(
       columns,
       ["Gene name", "Gene_name", "Gene", "HGNC", "SYMBOL"],
   )
   effect_column = find_column(
       columns,
       ["DNV effect", "DNV_effect", "Effect"],
   )
   consequence_column = find_column(
       columns,
       ["Canon Consequence", "Canonical Consequence", "Consequence"],
       required=False,
   )
   misfit_column = find_column(
       columns,
       [
           "MisFit-S v1.5.1",
           "MisFit S v1.5.1",
           "MisFit-S",
           "MisFit_S",
           "MisFit1.5_S",
       ],
   )
   reliable_column = find_column(
       columns,
       ["In reliable region", "Reliable region"],
       required=False,
   )
   loftee_column = find_column(
       columns,
       ["LOFTEE Confidence", "LOFTEE"],
       required=False,
   )


   print("Resolved Dataset S3 columns:")
   print(f"  Sample: {sample_column}")
   print(f"  Gene: {gene_column}")
   print(f"  DNV effect: {effect_column}")
   print(f"  Canonical consequence: {consequence_column}")
   print(f"  MisFit-S: {misfit_column}")
   print(f"  Reliable region: {reliable_column}")
   print(f"  LOFTEE: {loftee_column}")


   df[gene_column] = df[gene_column].map(normalize_gene)
   df[misfit_column] = pd.to_numeric(df[misfit_column], errors="coerce")


   blacklist, blacklist_path = load_blacklist(args.blacklist)
   if blacklist_path:
       print(
           f"Using blacklist: {blacklist_path} "
           f"({len(blacklist):,} genes)"
       )
   else:
       print("Warning: no blacklist file found; using zero blacklisted genes.")


   valid_gene = df[gene_column].ne("") & ~df[gene_column].isin(blacklist)


   if reliable_column and not args.ignore_reliable_region:
       in_reliable_region = parse_boolean(df[reliable_column])
   else:
       in_reliable_region = pd.Series(True, index=df.index)


   effect = (
       df[effect_column]
       .astype(str)
       .str.strip()
       .str.lower()
   )


   # Dataset S3 explicitly labels these classes as Missense and LoF.
   is_missense = effect.eq("missense")
   is_lof = effect.eq("lof")


   # Fall back to canonical consequence labels when necessary.
   if consequence_column:
       consequence = (
           df[consequence_column]
           .astype(str)
           .str.strip()
           .str.lower()
       )
       is_missense = is_missense | consequence.str.contains(
           "missense",
           regex=False,
           na=False,
       )
       is_lof = is_lof | consequence.str.contains(
           (
               r"frameshift|stop_gained|stop_lost|start_lost|"
               r"splice_acceptor|splice_donor"
           ),
           regex=True,
           na=False,
       )


   is_dmis = (
       is_missense
       & df[misfit_column].ge(args.misfit_threshold)
       & df[misfit_column].ne(-1)
   )


   if args.require_loftee_hc:
       if not loftee_column:
           raise ValueError(
               "--require-loftee-hc was requested, but no LOFTEE "
               "confidence column was found."
           )
       loftee_hc = (
           df[loftee_column]
           .astype(str)
           .str.strip()
           .str.upper()
           .eq("HC")
       )
       is_lof = is_lof & loftee_hc


   qualifying_lof = valid_gene & in_reliable_region & is_lof
   qualifying_dmis = valid_gene & in_reliable_region & is_dmis


   # Count variants, not merely unique genes. This matches pathway observed
   # counts when multiple qualifying DNVs occur in one gene.
   lof_counts = (
       df.loc[qualifying_lof, gene_column]
       .value_counts()
       .astype(int)
   )
   dmis_counts = (
       df.loc[qualifying_dmis, gene_column]
       .value_counts()
       .astype(int)
   )


   all_observed_genes = sorted(
       set(lof_counts.index) | set(dmis_counts.index)
   )
   rates = pd.DataFrame({"Gene": all_observed_genes})


   rates["Control_LoF_Count"] = (
       rates["Gene"].map(lof_counts).fillna(0).astype(int)
   )
   rates["Control_Dmis_Count"] = (
       rates["Gene"].map(dmis_counts).fillna(0).astype(int)
   )
   rates["Control_LoF_Dmis_Count"] = (
       rates["Control_LoF_Count"]
       + rates["Control_Dmis_Count"]
   )


   rates["Control_LoF_Rate"] = (
       rates["Control_LoF_Count"] / args.n_controls
   )
   rates["Control_Dmis_Rate"] = (
       rates["Control_Dmis_Count"] / args.n_controls
   )
   rates["Control_LoF_Dmis_Rate"] = (
       rates["Control_LoF_Dmis_Count"] / args.n_controls
   )


   rates["N_Controls"] = args.n_controls
   rates["MisFit_Threshold"] = args.misfit_threshold
   rates["Reliable_Region_Required"] = not args.ignore_reliable_region
   rates["LOFTEE_HC_Required"] = args.require_loftee_hc


   rates = rates.sort_values(
       [
           "Control_LoF_Dmis_Count",
           "Control_Dmis_Count",
           "Control_LoF_Count",
           "Gene",
       ],
       ascending=[False, False, False, True],
   )


   output_path = Path(args.output)
   output_path.parent.mkdir(parents=True, exist_ok=True)
   rates.to_csv(output_path, sep="\t", index=False)


   # Save a variant-level audit file so every included/excluded call can be
   # checked without reopening the workbook.
   qc_columns = [
       sample_column,
       gene_column,
       effect_column,
       misfit_column,
   ]
   for optional_column in [
       consequence_column,
       reliable_column,
       loftee_column,
   ]:
       if optional_column and optional_column not in qc_columns:
           qc_columns.append(optional_column)


   qc = df[qc_columns].copy()
   qc["Valid_Gene"] = valid_gene
   qc["In_Reliable_Region"] = in_reliable_region
   qc["Qualifying_LoF"] = qualifying_lof
   qc["Qualifying_Dmis"] = qualifying_dmis


   qc_path = Path(args.qc_output)
   qc_path.parent.mkdir(parents=True, exist_ok=True)
   qc.to_csv(qc_path, sep="\t", index=False)


   unique_samples = df[sample_column].dropna().astype(str).nunique()


   print("\nDataset S3 summary")
   print("------------------")
   print(f"Rows read: {len(df):,}")
   print(f"Unique sample IDs represented: {unique_samples:,}")
   print(f"Configured control cohort size: {args.n_controls:,}")
   print(f"Qualifying LoF variants: {int(qualifying_lof.sum()):,}")
   print(
       "Unique LoF-carrying genes: "
       f"{df.loc[qualifying_lof, gene_column].nunique():,}"
   )
   print(f"Qualifying Dmis variants: {int(qualifying_dmis.sum()):,}")
   print(
       "Unique Dmis-carrying genes: "
       f"{df.loc[qualifying_dmis, gene_column].nunique():,}"
   )
   print(
       "Qualifying LoF+Dmis variants: "
       f"{int(qualifying_lof.sum() + qualifying_dmis.sum()):,}"
   )
   print(f"Gene-rate rows written: {len(rates):,}")
   print(f"Saved gene rates: {output_path}")
   print(f"Saved QC table: {qc_path}")




if __name__ == "__main__":
   main()