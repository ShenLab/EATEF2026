#!/usr/bin/env python3

"""
Build a pathway similarity network whose EDGES represent overlap among
genes carrying qualifying DNVs, while NODE COLORS / CLUSTERS are derived
from the Gene Ontology DAG.

Inputs
------
results/significant_pathway_union_lof_vs_dmis.tsv
results/common_pathway_universe.tsv
data/go-basic.obo

Outputs
-------
results/pathway_network_nodes.tsv
results/pathway_network_edges.tsv
results/pathway_network_clusters.tsv
results/pathway_network_clusters.pdf
results/pathway_network_clusters.png
"""

import os
import re
import textwrap
from collections import defaultdict
from functools import lru_cache

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

# ==========================================================
# Files
# ==========================================================

SELECTED_PATHWAY_FILE = (
 "results/significant_pathway_union_lof_vs_dmis.tsv"
)
COMMON_PATHWAY_FILE = "results/common_pathway_universe.tsv"
POST_ANALYSIS_FILE = (
 "results/post_analysis_go_redundancy_results.tsv"
)
CASE_FILE = "results/EA_TEF_cases_annotated.tsv"
BLACKLIST_FILE = "data/09GENCODEV19_blacklist.txt"
GO_OBO_FILE = "data/go-basic.obo"

OUTPUT_NODE_FILE = "results/pathway_network_nodes.tsv"
OUTPUT_EDGE_FILE = "results/pathway_network_edges.tsv"
OUTPUT_CLUSTER_FILE = "results/pathway_network_clusters.tsv"
OUTPUT_PDF = "results/pathway_network_clusters.pdf"
OUTPUT_PNG = "results/pathway_network_clusters.png"

# ==========================================================
# Parameters
# ==========================================================


MIN_JACCARD = 0.05
ONTOLOGY_MAX_CLUSTERS = 5
INCLUDE_PART_OF_RELATIONSHIPS = True

# Qualifying DNV definition used in the pathway analyses.
CASE_GENE_COLUMN = "HGNC"
PHENOTYPE_COLUMN = "Syndormic or not"
SCORE_COLUMN = "MisFit_S"
MISFIT_THRESHOLD = 0.005

LOF_VARIANT_TYPES = {
 "lof",
 "frameshift",
 "stop_gained",
 "splice_acceptor",
 "splice_donor",
 "start_lost",
}

MISSENSE_VARIANT_TYPES = {
 "mis",
 "missense",
}

# A pathway is retained in the main network only when residual enrichment
# remains after genes assigned to retained child GO terms are removed.
POST_ANALYSIS_P_THRESHOLD = 0.05

POST_ANALYSIS_P_COLUMNS = [
 "Dmis_Post_Analysis_Raw_P_Value",
 "LoF_Dmis_Post_Analysis_Raw_P_Value",
]

RANDOM_SEED = 42
PNG_DPI = 450
PLOT_ISOLATED_NODES = True

FIGURE_WIDTH = 14.0
FIGURE_HEIGHT = 10.5
NODE_SIZE_BASE = 720
NODE_SIZE_SCALE = 210
LABEL_WRAP_WIDTH = 23

NETWORK_LABEL_FONT_SIZE = 11.5
NETWORK_TITLE_FONT_SIZE = 19
NETWORK_KEY_TITLE_FONT_SIZE = 13.5
NETWORK_KEY_HEADER_FONT_SIZE = 11.5
NETWORK_KEY_BODY_FONT_SIZE = 9.8
NETWORK_NOTE_FONT_SIZE = 9.5
NETWORK_KEY_BOX_ALPHA = 0.055

HIGH_CONTRAST_CLUSTER_COLORS = [
 "#0047AB",
 "#FF3B00",
 "#00A651",
 "#9C27B0",
 "#FFC000",
 "#00B8D9",
 "#E91E63",
 "#222222",
]

# ==========================================================
# General helpers
# ==========================================================

def normalize_go_name(value):
 """Normalize an MSigDB/GO label for matching to an OBO term name."""
 value = str(value).strip()

 for prefix in ("GOBP_", "GOCC_", "GOMF_"):
     if value.startswith(prefix):
         value = value[len(prefix):]
         break

 value = value.replace("_", " ").lower()
 value = re.sub(r"[^a-z0-9 ]+", " ", value)
 value = re.sub(r"\s+", " ", value).strip()
 return value

def expected_namespace(pathway):
 if pathway.startswith("GOBP_"):
     return "biological_process"
 if pathway.startswith("GOCC_"):
     return "cellular_component"
 if pathway.startswith("GOMF_"):
     return "molecular_function"
 return None

def readable_pathway_name(pathway):
 label = str(pathway).strip()

 for prefix in ("GOBP_", "GOCC_", "GOMF_"):
     if label.startswith(prefix):
         label = label[len(prefix):]
         break

 return label.replace("_", " ").lower()

def display_title(value):
 """Return journal-style capitalization for cluster and pathway labels."""
 text = str(value).strip().replace("_", " ")
 small_words = {"and", "or", "of", "the", "in", "on", "to", "for", "with"}

 words = text.lower().split()
 output = []

 for index, word in enumerate(words):
     if index > 0 and word in small_words:
         output.append(word)
     else:
         output.append(word.capitalize())

 return " ".join(output)

def short_label(pathway):
 return "\n".join(
     textwrap.wrap(
         readable_pathway_name(pathway),
         width=LABEL_WRAP_WIDTH,
         break_long_words=False,
         break_on_hyphens=False,
     )
 )

def parse_genes(value):
 if pd.isna(value):
     return set()

 return {
     gene.strip().upper()
     for gene in str(value).split(";")
     if gene.strip()
 }

def load_blacklisted_genes(path):
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
             blacklisted_genes.add(gene.upper())

 if not blacklisted_genes:
     raise ValueError(
         f"No genes were loaded from {path}"
     )

 return blacklisted_genes

def jaccard(set_a, set_b):
 union = set_a | set_b
 return len(set_a & set_b) / len(union) if union else 0.0

def cluster_color(cluster):
 """Return the exact same cluster color used by the scatter script."""
 return HIGH_CONTRAST_CLUSTER_COLORS[
     (int(cluster) - 1) % len(HIGH_CONTRAST_CLUSTER_COLORS)
 ]

# ==========================================================
# Minimal OBO parser
# ==========================================================

def parse_go_obo(path):
 if not os.path.exists(path):
     raise FileNotFoundError(
         f"GO ontology file not found: {path}\n"
         "Download go-basic.obo and save it at this path."
     )

 terms = {}
 current = None

 def finish_term(term):
     if not term or "id" not in term:
         return
     if term.get("is_obsolete", "false").lower() == "true":
         return

     term.setdefault("name", "")
     term.setdefault("namespace", "")
     term.setdefault("parents", set())
     term.setdefault("alt_id", [])
     terms[term["id"]] = term

 with open(path, "r", encoding="utf-8", errors="replace") as handle:
     for raw_line in handle:
         line = raw_line.rstrip("\n")

         if line == "[Term]":
             finish_term(current)
             current = {
                 "parents": set(),
                 "alt_id": [],
             }
             continue

         if line.startswith("[") and line.endswith("]"):
             finish_term(current)
             current = None
             continue

         if current is None or not line or line.startswith("!"):
             continue

         if line.startswith("id: "):
             current["id"] = line[4:].strip()

         elif line.startswith("name: "):
             current["name"] = line[6:].strip()

         elif line.startswith("namespace: "):
             current["namespace"] = line[11:].strip()

         elif line.startswith("alt_id: "):
             current["alt_id"].append(line[8:].strip())

         elif line.startswith("is_a: "):
             parent = line[6:].split(" ! ", 1)[0].strip()
             current["parents"].add(parent)

         elif (
             INCLUDE_PART_OF_RELATIONSHIPS
             and line.startswith("relationship: part_of ")
         ):
             parent = (
                 line[len("relationship: part_of "):]
                 .split(" ! ", 1)[0]
                 .strip()
             )
             current["parents"].add(parent)

         elif line.startswith("is_obsolete: "):
             current["is_obsolete"] = line[13:].strip()

 finish_term(current)

 alt_lookup = {}
 for go_id, term in terms.items():
     for alt_id in term["alt_id"]:
         alt_lookup[alt_id] = go_id

 return terms, alt_lookup

def build_name_lookup(terms):
 lookup = defaultdict(list)

 for go_id, term in terms.items():
     key = (
         term["namespace"],
         normalize_go_name(term["name"]),
     )
     lookup[key].append(go_id)

 return lookup

# ==========================================================
# Ontology graph calculations
# ==========================================================

def ontology_helpers(terms):
 roots = {
     "biological_process": "GO:0008150",
     "cellular_component": "GO:0005575",
     "molecular_function": "GO:0003674",
 }

 @lru_cache(maxsize=None)
 def ancestors_including_self(go_id):
     result = {go_id}

     for parent in terms.get(go_id, {}).get("parents", set()):
         if parent in terms:
             result |= ancestors_including_self(parent)

     return frozenset(result)

 @lru_cache(maxsize=None)
 def depth(go_id):
     parents = [
         parent
         for parent in terms.get(go_id, {}).get("parents", set())
         if parent in terms
     ]

     if not parents:
         return 0


     return 1 + max(depth(parent) for parent in parents)


 def deepest_common_ancestor(go_ids):
     go_ids = list(go_ids)

     if not go_ids:
         return None

     shared = set(ancestors_including_self(go_ids[0]))

     for go_id in go_ids[1:]:
         shared &= set(ancestors_including_self(go_id))

     if not shared:
         return None

     namespace = terms[go_ids[0]]["namespace"]
     namespace_root = roots.get(namespace)

     informative = [
         ancestor
         for ancestor in shared
         if ancestor != namespace_root
     ]








     candidates = informative or list(shared)








     return max(
         candidates,
         key=lambda ancestor: (
             depth(ancestor),
             terms[ancestor]["name"],
         ),
     )








 return ancestors_including_self, depth, deepest_common_ancestor
















def map_pathways_to_go_ids(pathways, terms, name_lookup):
 mapping = {}
 failures = []








 for pathway in pathways:
     namespace = expected_namespace(pathway)
     normalized_name = normalize_go_name(pathway)
     candidates = name_lookup.get(
         (namespace, normalized_name),
         [],
     )








     if len(candidates) == 1:
         mapping[pathway] = candidates[0]
         continue








     if len(candidates) > 1:
         candidates = sorted(
             candidates,
             key=lambda go_id: go_id,
         )
         mapping[pathway] = candidates[0]
         print(
             f"Warning: multiple GO IDs matched {pathway}; "
             f"using {candidates[0]}."
         )
         continue








     failures.append(pathway)








 if failures:
     raise ValueError(
         "Could not map these pathway names to GO terms in "
         f"{GO_OBO_FILE}: {failures[:20]}\n"
         "Make sure the OBO version is compatible with the GMT file."
     )








 return mapping
















def ontology_distance(
 go_a,
 go_b,
 terms,
 ancestors_including_self,
 depth,
 deepest_common_ancestor,
):
 namespace_a = terms[go_a]["namespace"]
 namespace_b = terms[go_b]["namespace"]








 if namespace_a != namespace_b:
     return 1000.0








 common = deepest_common_ancestor([go_a, go_b])








 if common is None:
     return 1000.0








 return float(
     depth(go_a)
     + depth(go_b)
     - 2 * depth(common)
 )
















def assign_ontology_clusters(
 pathway_to_go,
 terms,
 ancestors_including_self,
 depth,
 deepest_common_ancestor,
):
 pathways = sorted(pathway_to_go)
 n_pathways = len(pathways)








 if n_pathways == 1:
     labels = np.array([1], dtype=int)








 else:
     distance_matrix = np.zeros(
         (n_pathways, n_pathways),
         dtype=float,
     )








     for i, pathway_a in enumerate(pathways):
         for j in range(i + 1, n_pathways):
             pathway_b = pathways[j]








             distance = ontology_distance(
                 pathway_to_go[pathway_a],
                 pathway_to_go[pathway_b],
                 terms,
                 ancestors_including_self,
                 depth,
                 deepest_common_ancestor,
             )








             distance_matrix[i, j] = distance
             distance_matrix[j, i] = distance








     condensed = squareform(
         distance_matrix,
         checks=True,
     )








     hierarchy = linkage(
         condensed,
         method="average",
     )








     requested_clusters = min(
         ONTOLOGY_MAX_CLUSTERS,
         n_pathways,
     )








     labels = fcluster(
         hierarchy,
         t=requested_clusters,
         criterion="maxclust",
     )








 raw_members = defaultdict(list)








 for pathway, raw_label in zip(pathways, labels):
     raw_members[int(raw_label)].append(pathway)








 # Stable numbering: largest cluster first, then alphabetically.
 ordered_groups = sorted(
     raw_members.values(),
     key=lambda members: (
         -len(members),
         sorted(members)[0],
     ),
 )








 cluster_lookup = {}
 cluster_metadata = {}








 for cluster_number, members in enumerate(ordered_groups, start=1):
     go_ids = [
         pathway_to_go[pathway]
         for pathway in members
     ]








     common_ancestor = deepest_common_ancestor(go_ids)








     if common_ancestor is None:
         namespaces = sorted(
             {
                 terms[go_id]["namespace"]
                 for go_id in go_ids
             }
         )
         cluster_name = " / ".join(namespaces)
         common_ancestor = ""
     else:
         cluster_name = terms[common_ancestor]["name"]








     for pathway in members:
         cluster_lookup[pathway] = cluster_number








     cluster_metadata[cluster_number] = {
         "Ontology_Cluster_Name": cluster_name,
         "Ontology_Ancestor_GO_ID": common_ancestor,
         "Pathway_Count": len(members),
     }








 return cluster_lookup, cluster_metadata
















# ==========================================================
# Load data
# ==========================================================








print("Loading qualifying DNV genes...")

blacklisted_genes = load_blacklisted_genes(
 BLACKLIST_FILE
)

cases = pd.read_csv(
 CASE_FILE,
 sep="\t",
 low_memory=False,
 encoding="latin1",
)

required_case_columns = {
 "Var_type",
 CASE_GENE_COLUMN,
 PHENOTYPE_COLUMN,
 SCORE_COLUMN,
}

missing_case_columns = (
 required_case_columns
 - set(cases.columns)
)

if missing_case_columns:
 raise ValueError(
     "Case table is missing required columns: "
     f"{sorted(missing_case_columns)}"
 )

cases[PHENOTYPE_COLUMN] = (
 cases[PHENOTYPE_COLUMN]
 .astype("string")
 .str.strip()
 .str.lower()
)

cases["Var_type"] = (
 cases["Var_type"]
 .astype(str)
 .str.strip()
 .str.lower()
)

cases[SCORE_COLUMN] = pd.to_numeric(
 cases[SCORE_COLUMN],
 errors="coerce",
)

cases.loc[
 cases[SCORE_COLUMN] == -1,
 SCORE_COLUMN,
] = np.nan

lof_mask = cases["Var_type"].isin(
 LOF_VARIANT_TYPES
)

dmis_mask = (
 cases["Var_type"].isin(
     MISSENSE_VARIANT_TYPES
 )
 & cases[SCORE_COLUMN].notna()
 & cases[SCORE_COLUMN].ge(
     MISFIT_THRESHOLD
 )
)

cases = cases[
 cases[PHENOTYPE_COLUMN].eq("complex")
 & (lof_mask | dmis_mask)
].copy()

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
 cases[CASE_GENE_COLUMN].ne("")
 & ~cases[CASE_GENE_COLUMN].isin(
     blacklisted_genes
 )
].copy()

dnv_gene_set = set(
 cases[CASE_GENE_COLUMN]
)

print(
 f"Qualifying DNV-carrying genes loaded: "
 f"{len(dnv_gene_set)}"
)


print("Loading selected pathways...")








selected = pd.read_csv(
 SELECTED_PATHWAY_FILE,
 sep="\t",
 low_memory=False,
)








required_selected = {
 "Pathway",
 "Minimum_FWER",
 "LoF_Dmis_Observed",
}








missing = required_selected - set(selected.columns)








if missing:
 raise ValueError(
     "Selected pathway table is missing columns: "
     f"{sorted(missing)}"
 )








selected["Pathway"] = (
 selected["Pathway"]
 .astype(str)
 .str.strip()
)








selected["Minimum_FWER"] = pd.to_numeric(
 selected["Minimum_FWER"],
 errors="coerce",
)








selected["LoF_Dmis_Observed"] = pd.to_numeric(
 selected["LoF_Dmis_Observed"],
 errors="coerce",
)








selected = selected[
 selected["Pathway"].ne("")
].drop_duplicates("Pathway").copy()








# ==========================================================
# Remove pathways without residual post-pruning significance
# ==========================================================








if not os.path.exists(POST_ANALYSIS_FILE):
 raise FileNotFoundError(
     "Post-analysis GO redundancy results were not found: "
     f"{POST_ANALYSIS_FILE}\n"
     "Run the GO redundancy post-analysis before building "
     "the main pathway network."
 )








post_analysis = pd.read_csv(
 POST_ANALYSIS_FILE,
 sep="\t",
 low_memory=False,
)








required_post_columns = {
 "Pathway",
 *POST_ANALYSIS_P_COLUMNS,
}








missing_post_columns = (
 required_post_columns
 - set(post_analysis.columns)
)








if missing_post_columns:
 raise ValueError(
     "Post-analysis table is missing required columns: "
     f"{sorted(missing_post_columns)}"
 )








post_analysis["Pathway"] = (
 post_analysis["Pathway"]
 .astype(str)
 .str.strip()
)








if post_analysis["Pathway"].duplicated().any():
 duplicated = (
     post_analysis.loc[
         post_analysis["Pathway"].duplicated(
             keep=False
         ),
         "Pathway",
     ]
     .drop_duplicates()
     .tolist()
 )
 raise ValueError(
     "Post-analysis table contains duplicate pathways: "
     f"{duplicated[:20]}"
 )








for column in POST_ANALYSIS_P_COLUMNS:
 post_analysis[column] = pd.to_numeric(
     post_analysis[column],
     errors="coerce",
 )








post_analysis["Minimum_Post_Analysis_Raw_P_Value"] = (
 post_analysis[
     POST_ANALYSIS_P_COLUMNS
 ].min(
     axis=1,
     skipna=True,
 )
)








post_analysis[
 "Retained_After_Child_Gene_Removal"
] = (
 post_analysis[
     "Minimum_Post_Analysis_Raw_P_Value"
 ]
 < POST_ANALYSIS_P_THRESHOLD
)








selected_before_post_filter = len(selected)








selected = selected.merge(
 post_analysis[
     [
         "Pathway",
         *POST_ANALYSIS_P_COLUMNS,
         "Minimum_Post_Analysis_Raw_P_Value",
         "Retained_After_Child_Gene_Removal",
     ]
 ],
 on="Pathway",
 how="left",
 validate="one_to_one",
)








missing_post_results = selected.loc[
 selected[
     "Minimum_Post_Analysis_Raw_P_Value"
 ].isna(),
 "Pathway",
].tolist()








if missing_post_results:
 raise ValueError(
     "Selected pathways are missing usable post-analysis "
     "raw p-values: "
     f"{missing_post_results[:20]}"
 )








removed_pathways = (
 selected.loc[
     ~selected[
         "Retained_After_Child_Gene_Removal"
     ],
     [
         "Pathway",
         *POST_ANALYSIS_P_COLUMNS,
         "Minimum_Post_Analysis_Raw_P_Value",
     ],
 ]
 .sort_values(
     [
         "Minimum_Post_Analysis_Raw_P_Value",
         "Pathway",
     ],
     ascending=[
         True,
         True,
     ],
 )
)








selected = selected.loc[
 selected[
     "Retained_After_Child_Gene_Removal"
 ]
].copy()








if selected.empty:
 raise ValueError(
     "No pathways remain after requiring residual "
     "post-analysis raw p-value < "
     f"{POST_ANALYSIS_P_THRESHOLD}."
 )








print(
 "Post-pruning figure criterion: "
 "minimum of Dmis and LoF+Dmis post-analysis "
 f"raw p-values < {POST_ANALYSIS_P_THRESHOLD}"
)








print(
 "Pathways before post-pruning filter: "
 f"{selected_before_post_filter}"
)








print(
 "Pathways retained after child-gene removal: "
 f"{len(selected)}"
)








print(
 "Pathways removed after child-gene removal: "
 f"{len(removed_pathways)}"
)








if not removed_pathways.empty:
 print("Removed pathways:")
 for _, row in removed_pathways.iterrows():
     print(
         f"  - {row['Pathway']}: "
         f"minimum post-analysis raw p = "
         f"{row['Minimum_Post_Analysis_Raw_P_Value']:.6g}"
     )








selected_pathways = set(selected["Pathway"])








print(f"Selected pathways plotted: {len(selected_pathways)}")








common = pd.read_csv(
 COMMON_PATHWAY_FILE,
 sep="\t",
 low_memory=False,
)








required_common = {"Pathway", "Genes"}
missing = required_common - set(common.columns)








if missing:
 raise ValueError(
     "Common pathway table is missing columns: "
     f"{sorted(missing)}"
 )








common["Pathway"] = (
 common["Pathway"]
 .astype(str)
 .str.strip()
)








common = common[
 common["Pathway"].isin(selected_pathways)
].copy()








# For Figure 1C, similarity is based only on qualifying DNV-carrying
# genes in each pathway, rather than on the complete GO gene sets.
common["GeneSet"] = common["Genes"].apply(
 lambda value: parse_genes(value) & dnv_gene_set
)








missing_pathways = selected_pathways - set(common["Pathway"])








if missing_pathways:
 raise ValueError(
     "Selected pathways missing from common pathway universe: "
     f"{sorted(missing_pathways)[:20]}"
 )








nodes = (
 selected[
     [
         "Pathway",
         "Minimum_FWER",
         "LoF_Dmis_Observed",
         "Dmis_Post_Analysis_Raw_P_Value",
         "LoF_Dmis_Post_Analysis_Raw_P_Value",
         "Minimum_Post_Analysis_Raw_P_Value",
         "Retained_After_Child_Gene_Removal",
     ]
 ]
 .merge(
     common[["Pathway", "GeneSet"]],
     on="Pathway",
     how="left",
     validate="one_to_one",
 )
)








nodes = nodes.rename(
 columns={
     "Minimum_FWER": "BestFWER",
     "LoF_Dmis_Observed": "ObservedForSize",
 }
)








nodes["DNVGeneCount"] = nodes["GeneSet"].map(len)
















# ==========================================================
# Ontology clustering
# ==========================================================








print("Loading Gene Ontology DAG...")








terms, alt_lookup = parse_go_obo(GO_OBO_FILE)
name_lookup = build_name_lookup(terms)








pathway_to_go = map_pathways_to_go_ids(
 nodes["Pathway"].tolist(),
 terms,
 name_lookup,
)








(
 ancestors_including_self,
 depth,
 deepest_common_ancestor,
) = ontology_helpers(terms)








cluster_lookup, cluster_metadata = assign_ontology_clusters(
 pathway_to_go,
 terms,
 ancestors_including_self,
 depth,
 deepest_common_ancestor,
)








nodes["GO_ID"] = nodes["Pathway"].map(pathway_to_go)
nodes["GO_Term_Name"] = nodes["GO_ID"].map(
 lambda go_id: terms[go_id]["name"]
)
nodes["GO_Namespace"] = nodes["GO_ID"].map(
 lambda go_id: terms[go_id]["namespace"]
)
nodes["GO_Depth"] = nodes["GO_ID"].map(depth)
nodes["Cluster"] = nodes["Pathway"].map(cluster_lookup).astype(int)
nodes["Ontology_Cluster_Name"] = nodes["Cluster"].map(
 lambda cluster: cluster_metadata[cluster][
     "Ontology_Cluster_Name"
 ]
)
nodes["Ontology_Ancestor_GO_ID"] = nodes["Cluster"].map(
 lambda cluster: cluster_metadata[cluster][
     "Ontology_Ancestor_GO_ID"
 ]
)








print(
 f"Ontology-derived clusters: "
 f"{nodes['Cluster'].nunique()}"
)








for cluster in sorted(cluster_metadata):
 metadata = cluster_metadata[cluster]
 print(
     f"  Cluster {cluster}: "
     f"{metadata['Ontology_Cluster_Name']} "
     f"({metadata['Pathway_Count']} pathways)"
 )
















# ==========================================================
# DNV-gene-overlap network
# ==========================================================








graph = nx.Graph()








for _, row in nodes.iterrows():
 graph.add_node(
     row["Pathway"],
     observed=float(row["ObservedForSize"]),
     cluster=int(row["Cluster"]),
 )








edge_rows = []
records = list(
 nodes[
     ["Pathway", "GeneSet"]
 ].itertuples(index=False, name=None)
)








for i, (pathway_a, genes_a) in enumerate(records):
 for pathway_b, genes_b in records[i + 1:]:
     shared = genes_a & genes_b
     score = jaccard(genes_a, genes_b)








     if score < MIN_JACCARD:
         continue








     graph.add_edge(
         pathway_a,
         pathway_b,
         weight=score,
         jaccard=score,
         shared_gene_count=len(shared),
     )








     edge_rows.append(
         {
             "Pathway1": pathway_a,
             "Pathway2": pathway_b,
             "SharedDNVGeneCount": len(shared),
             "SharedDNVGenes": ";".join(sorted(shared)),
             "Jaccard": score,
         }
     )








print(f"Network nodes: {graph.number_of_nodes()}")
print(f"Network edges: {graph.number_of_edges()}")
















# ==========================================================
# Save tables
# ==========================================================








os.makedirs("results", exist_ok=True)








node_output = (
 nodes.drop(columns=["GeneSet"])
 .sort_values(
     ["Cluster", "BestFWER", "Pathway"],
     ascending=[True, True, True],
 )
)








node_output.to_csv(
 OUTPUT_NODE_FILE,
 sep="\t",
 index=False,
)








edge_output = pd.DataFrame(
 edge_rows,
 columns=[
     "Pathway1",
     "Pathway2",
     "SharedDNVGeneCount",
     "SharedDNVGenes",
     "Jaccard",
 ],
)








if not edge_output.empty:
 edge_output = edge_output.sort_values(
     ["Jaccard", "SharedDNVGeneCount"],
     ascending=[False, False],
 )








edge_output.to_csv(
 OUTPUT_EDGE_FILE,
 sep="\t",
 index=False,
)








node_output[
 [
     "Cluster",
     "Ontology_Cluster_Name",
     "Ontology_Ancestor_GO_ID",
     "Pathway",
     "GO_ID",
     "GO_Term_Name",
     "GO_Namespace",
     "GO_Depth",
     "BestFWER",
     "ObservedForSize",
     "DNVGeneCount",
     "Dmis_Post_Analysis_Raw_P_Value",
     "LoF_Dmis_Post_Analysis_Raw_P_Value",
     "Minimum_Post_Analysis_Raw_P_Value",
     "Retained_After_Child_Gene_Removal",
 ]
].to_csv(
 OUTPUT_CLUSTER_FILE,
 sep="\t",
 index=False,
)
















# ==========================================================
# Plot
# ==========================================================




if PLOT_ISOLATED_NODES:
  plot_graph = graph.copy()
else:
  connected_nodes = [
      node
      for node, degree_value in graph.degree()
      if degree_value > 0
  ]
  plot_graph = graph.subgraph(connected_nodes).copy()




if plot_graph.number_of_nodes() == 0:
  raise ValueError("No pathways are available to plot.")




positions = nx.spring_layout(
  plot_graph,
  weight="weight",
  seed=RANDOM_SEED,
  iterations=5000,
  k=2.10 / np.sqrt(plot_graph.number_of_nodes()),
)




# Recenter and normalize the network to fill the panel evenly.
position_array = np.array(list(positions.values()))


if len(position_array) > 0:
  position_array = position_array - position_array.mean(axis=0)
  max_extent = np.abs(position_array).max()


  if max_extent > 0:
      position_array = position_array / max_extent


  positions = {
      node: position_array[index]
      for index, node in enumerate(positions)
  }




# ==========================================================
# Publication layout
# ==========================================================
#
# Top: pathway network
# Bottom: two clean rows
#   Row 1: ontology-cluster legend
#   Row 2: node-size and edge-width encodings
# ==========================================================




figure = plt.figure(
  figsize=(14.0, 10.2),
  constrained_layout=False,
)




outer_grid = figure.add_gridspec(
  3,
  1,
  height_ratios=[8.0, 1.25, 0.95],
  hspace=0.10,
)




axis = figure.add_subplot(outer_grid[0, 0])
cluster_legend_axis = figure.add_subplot(outer_grid[1, 0])
metric_legend_axis = figure.add_subplot(outer_grid[2, 0])




for legend_axis in (
  cluster_legend_axis,
  metric_legend_axis,
):
  legend_axis.set_xlim(0, 1)
  legend_axis.set_ylim(0, 1)
  legend_axis.axis("off")




# ----------------------------
# Draw DNV-gene-overlap edges
# ----------------------------
for pathway_a, pathway_b, edge_data in plot_graph.edges(data=True):
  score = edge_data["jaccard"]


  nx.draw_networkx_edges(
      plot_graph,
      positions,
      edgelist=[(pathway_a, pathway_b)],
      width=0.85 + 8.8 * score,
      alpha=min(0.20 + 1.45 * score, 0.78),
      edge_color="#5F9FD3",
      ax=axis,
  )




node_order = list(plot_graph.nodes())




node_colors = [
  cluster_color(cluster_lookup[pathway])
  for pathway in node_order
]




node_sizes = [
  NODE_SIZE_BASE
  + NODE_SIZE_SCALE
  * np.sqrt(
      max(
          float(plot_graph.nodes[pathway]["observed"]),
          0.0,
      )
  )
  for pathway in node_order
]




nx.draw_networkx_nodes(
  plot_graph,
  positions,
  nodelist=node_order,
  node_size=node_sizes,
  node_color=node_colors,
  edgecolors="#202020",
  linewidths=1.25,
  ax=axis,
)




# Labels are offset from nodes and use a thin white outline instead of
# opaque text boxes, so network edges remain visible.
label_positions = {}


for pathway in node_order:
  x_value, y_value = positions[pathway]


  # Push labels away from the network center.
  direction = np.array([x_value, y_value], dtype=float)
  norm = np.linalg.norm(direction)


  if norm == 0:
      direction = np.array([0.0, 1.0])
  else:
      direction = direction / norm


  label_positions[pathway] = (
      positions[pathway]
      + 0.115 * direction
      + np.array([0.0, 0.025])
  )




label_artists = nx.draw_networkx_labels(
  plot_graph,
  label_positions,
  labels={
      pathway: short_label(pathway)
      for pathway in node_order
  },
  font_size=11.6,
  font_weight="semibold",
  font_color="#161616",
  ax=axis,
)




for artist in label_artists.values():
  artist.set_path_effects(
      [
          path_effects.Stroke(
              linewidth=3.2,
              foreground="white",
          ),
          path_effects.Normal(),
      ]
  )




axis.set_title(
  "Pathways Retaining Enrichment After GO Redundancy Pruning",
  fontsize=20,
  fontweight="bold",
  pad=16,
)




axis.margins(x=0.18, y=0.19)
axis.axis("off")




# ----------------------------
# Ontology-cluster legend
# ----------------------------
cluster_ids = sorted(cluster_metadata)




cluster_legend_axis.text(
  0.5,
  0.92,
  "Ontology Clusters",
  ha="center",
  va="top",
  fontsize=14.5,
  fontweight="bold",
  color="#171717",
)




# Use two rows to prevent long cluster names from colliding.
cluster_slots = {
  1: (0.16, 0.54),
  2: (0.50, 0.54),
  3: (0.84, 0.54),
  4: (0.32, 0.14),
  5: (0.68, 0.14),
}




for cluster in cluster_ids:
  if cluster not in cluster_slots:
      continue


  x_position, y_position = cluster_slots[cluster]
  color = cluster_color(cluster)
  cluster_name = display_title(
      cluster_metadata[cluster]["Ontology_Cluster_Name"]
  )


  wrapped_name = "\n".join(
      textwrap.wrap(
          cluster_name,
          width=27,
          break_long_words=False,
          break_on_hyphens=False,
      )
  )


  cluster_legend_axis.scatter(
      [x_position - 0.085],
      [y_position],
      s=135,
      facecolor=color,
      edgecolor="#202020",
      linewidth=0.85,
      transform=cluster_legend_axis.transAxes,
      clip_on=False,
  )


  cluster_legend_axis.text(
      x_position - 0.060,
      y_position,
      f"{cluster}. {wrapped_name}",
      transform=cluster_legend_axis.transAxes,
      ha="left",
      va="center",
      fontsize=10.8,
      fontweight="bold",
      color=color,
      linespacing=1.02,
  )




# ----------------------------
# Encoding legend
# ----------------------------
metric_legend_axis.axhline(
  0.93,
  xmin=0.04,
  xmax=0.96,
  color="#B8B8B8",
  linewidth=0.8,
)




metric_legend_axis.text(
  0.16,
  0.72,
  "Node Size = Observed Variants",
  ha="center",
  va="center",
  fontsize=11.2,
  fontweight="bold",
)




observed_values = sorted(
  {
      int(round(value))
      for value in nodes["ObservedForSize"].dropna()
      if value >= 0
  }
)




if len(observed_values) > 3:
  indices = np.linspace(
      0,
      len(observed_values) - 1,
      3,
      dtype=int,
  )
  observed_values = [
      observed_values[index]
      for index in indices
  ]




if not observed_values:
  observed_values = [0]




node_legend_x = np.linspace(
  0.32,
  0.48,
  len(observed_values),
)




for x_position, observed_value in zip(
  node_legend_x,
  observed_values,
):
  legend_size = (
      NODE_SIZE_BASE
      + NODE_SIZE_SCALE * np.sqrt(max(observed_value, 0))
  ) * 0.17


  metric_legend_axis.scatter(
      [x_position],
      [0.70],
      s=legend_size,
      facecolor="#D2D2D2",
      edgecolor="#555555",
      linewidth=0.8,
      transform=metric_legend_axis.transAxes,
      clip_on=False,
  )


  metric_legend_axis.text(
      x_position,
      0.28,
      str(observed_value),
      transform=metric_legend_axis.transAxes,
      ha="center",
      va="center",
      fontsize=9.8,
      color="#303030",
  )




metric_legend_axis.text(
  0.66,
  0.72,
  "Edge Width = Jaccard Overlap of DNV Genes",
  ha="center",
  va="center",
  fontsize=11.2,
  fontweight="bold",
)




jaccard_examples = [
  MIN_JACCARD,
  0.20,
  0.35,
]




edge_legend_x = [
  0.80,
  0.87,
  0.94,
]




for x_position, value in zip(
  edge_legend_x,
  jaccard_examples,
):
  metric_legend_axis.plot(
      [x_position - 0.025, x_position + 0.025],
      [0.70, 0.70],
      color="#5F9FD3",
      linewidth=0.85 + 8.8 * value,
      alpha=min(0.20 + 1.45 * value, 0.78),
      solid_capstyle="butt",
      transform=metric_legend_axis.transAxes,
      clip_on=False,
  )


  metric_legend_axis.text(
      x_position,
      0.28,
      f"{value:.2f}",
      transform=metric_legend_axis.transAxes,
      ha="center",
      va="center",
      fontsize=9.8,
      color="#303030",
  )




figure.text(
  0.5,
  0.015,
  (
      f"Edges shown for DNV-gene Jaccard \u2265 {MIN_JACCARD:.2f}. "
      f"Only pathways with post-pruning raw P < "
      f"{POST_ANALYSIS_P_THRESHOLD:.2f} are displayed."
  ),
  ha="center",
  va="bottom",
  fontsize=9.5,
  color="#3A3A3A",
)




figure.subplots_adjust(
  left=0.045,
  right=0.975,
  top=0.955,
  bottom=0.070,
)

figure.savefig(
  OUTPUT_PDF,
  bbox_inches="tight",
)

figure.savefig(
  OUTPUT_PNG,
  dpi=PNG_DPI,
  bbox_inches="tight",
)

plt.close(figure)

print(f"Saved node table: {OUTPUT_NODE_FILE}")
print(f"Saved edge table: {OUTPUT_EDGE_FILE}")
print(f"Saved ontology cluster table: {OUTPUT_CLUSTER_FILE}")
print(f"Saved network PDF: {OUTPUT_PDF}")
print(f"Saved network PNG: {OUTPUT_PNG}")