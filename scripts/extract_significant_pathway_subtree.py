
#!/usr/bin/env python3




"""
Create a publication-ready Gene Ontology subtree figure for significant pathways.


Design goals
------------
- Legible when the complete figure is viewed at approximately US Letter paper width.
- Vector PDF output for publication.
- Large, readable selected-term labels.
- Restrained, professional visual hierarchy.
- Minimal ontology structure: selected terms, namespace roots, and informative
branch/merge points.
- Separate namespace panels when needed, preventing one dense branch from
compressing the entire figure.




Inputs
------
data/go-basic.obo
results/pathway_network_clusters.tsv




Required cluster-table columns
------------------------------
Pathway
GO_ID
Cluster
Ontology_Cluster_Name




Outputs
-------
results/significant_pathway_go_subtree.pdf
results/significant_pathway_go_subtree.png
results/significant_pathway_go_subtree.svg
"""


from __future__ import annotations


import os
import shutil
import subprocess
import tempfile
import textwrap
from collections import defaultdict


import pandas as pd


# ==========================================================
# Files
# ==========================================================


GO_OBO_FILE = "data/go-basic.obo"
CLUSTER_FILE = "results/pathway_network_clusters.tsv"


OUTPUT_PDF = "results/significant_pathway_go_subtree.pdf"
OUTPUT_PNG = "results/significant_pathway_go_subtree.png"
OUTPUT_SVG = "results/significant_pathway_go_subtree.svg"


# ==========================================================
# Figure and ontology parameters
# ==========================================================


INCLUDE_PART_OF = True


# Keep only selected terms, roots, branch points, and merge points.
KEEP_ALL_ANCESTORS = False


DROP_UNUSED_NAMESPACE_ROOTS = True


TARGET_NAMESPACES = {
 "biological_process",
 "cellular_component",
 "molecular_function",
}


SHOW_GO_IDS = False

# Force specific GO terms to use selected-term styling even if they are not
# present in the current cluster table.
FORCE_SELECTED_TERMS = {
    "regulation of cellular catabolic process": 3,
}

TITLE_FONT_SIZE = 44


# Label wrapping is intentionally narrow so terms remain readable at page width.
SELECTED_LABEL_WRAP_WIDTH = 28
ANCESTOR_LABEL_WRAP_WIDTH = 30


PNG_DPI = 600


# Graphviz layout controls. LR generally produces larger, more readable labels
# than a tall top-to-bottom tree when viewed at page width.
RANK_DIRECTION = "LR"
RANK_SEPARATION = 0.70
NODE_SEPARATION = 0.26


# These values are in inches in Graphviz.
SELECTED_NODE_WIDTH = 3.80
SELECTED_NODE_HEIGHT = 1.18
ANCESTOR_NODE_WIDTH = 2.75
ANCESTOR_NODE_HEIGHT = 0.76


SELECTED_FONT_SIZE = 32
ROOT_FONT_SIZE = 42
ANCESTOR_FONT_SIZE = 24
LEGEND_FONT_SIZE = 14


# Approximate US Letter landscape page dimensions.
PAGE_WIDTH_IN = 12.6
PAGE_HEIGHT_IN = 8.4


# Publication palette. Cluster colors match the main figures.
CLUSTER_COLORS = [
 "#0047AB",
 "#D94F04",
 "#168B45",
 "#7A3DB8",
 "#C28A00",
 "#008EA8",
 "#C2185B",
 "#333333",
]


ANCESTOR_FILL = "#F2F2F2"
ROOT_FILL = "#D7D7D7"
EDGE_COLOR = "#999999"
TEXT_COLOR = "#171717"
NODE_BORDER = "#3F3F3F"


# ==========================================================
# Text helpers
# ==========================================================


def cluster_color(cluster: int) -> str:
 return CLUSTER_COLORS[
     (int(cluster) - 1) % len(CLUSTER_COLORS)
 ]


def display_title(value: str) -> str:
 """Convert pathway names to readable title capitalization."""
 text = str(value).replace("_", " ").strip()


 small_words = {
     "and", "or", "of", "the", "in", "on", "to",
     "for", "with", "by", "from", "via",
 }


 words = text.lower().split()
 output = []


 for index, word in enumerate(words):
     if index > 0 and word in small_words:
         output.append(word)
     else:
         output.append(word.capitalize())


 return " ".join(output)


def wrap_label(value: str, width: int) -> str:
 return "\n".join(
     textwrap.wrap(
         display_title(value),
         width=width,
         break_long_words=False,
         break_on_hyphens=False,
     )
 )


def dot_escape(value: str) -> str:
 return (
     str(value)
     .replace("\\", "\\\\")
     .replace('"', '\\"')
     .replace("\n", "\\n")
 )


def namespace_display_name(namespace: str) -> str:
 mapping = {
     "biological_process": "Biological Process",
     "cellular_component": "Cellular Component",
     "molecular_function": "Molecular Function",
 }
 return mapping.get(namespace, display_title(namespace))


# ==========================================================
# OBO parser
# ==========================================================


def parse_go_obo(path: str):
 if not os.path.exists(path):
     raise FileNotFoundError(
         f"GO ontology file not found: {path}"
     )


 terms = {}
 alt_to_primary = {}
 current = None


 def finish_term(term):
     if not term or "id" not in term:
         return


     if term.get("is_obsolete", "false").lower() == "true":
         return


     term.setdefault("name", "")
     term.setdefault("namespace", "")
     term.setdefault("parents", {})
     term.setdefault("alt_ids", [])


     terms[term["id"]] = term


     for alt_id in term["alt_ids"]:
         alt_to_primary[alt_id] = term["id"]


 with open(
     path,
     "r",
     encoding="utf-8",
     errors="replace",
 ) as handle:
     for raw_line in handle:
         line = raw_line.rstrip("\n")


         if line == "[Term]":
             finish_term(current)
             current = {
                 "parents": {},
                 "alt_ids": [],
             }
             continue


         if line.startswith("[") and line.endswith("]"):
             finish_term(current)
             current = None
             continue


         if current is None or not line:
             continue


         if line.startswith("id: "):
             current["id"] = line[4:].strip()


         elif line.startswith("name: "):
             current["name"] = line[6:].strip()


         elif line.startswith("namespace: "):
             current["namespace"] = line[11:].strip()


         elif line.startswith("alt_id: "):
             current["alt_ids"].append(
                 line[8:].strip()
             )


         elif line.startswith("is_a: "):
             parent = (
                 line[6:]
                 .split(" ! ", 1)[0]
                 .strip()
             )
             current["parents"][parent] = "is_a"


         elif (
             INCLUDE_PART_OF
             and line.startswith("relationship: part_of ")
         ):
             parent = (
                 line[len("relationship: part_of "):]
                 .split(" ! ", 1)[0]
                 .strip()
             )
             current["parents"][parent] = "part_of"


         elif line.startswith("is_obsolete: "):
             current["is_obsolete"] = line[13:].strip()


 finish_term(current)
 return terms, alt_to_primary


# ==========================================================
# Ontology extraction
# ==========================================================


def canonical_go_id(go_id, terms, alt_to_primary):
 go_id = str(go_id).strip()


 if go_id in terms:
     return go_id


 if go_id in alt_to_primary:
     return alt_to_primary[go_id]


 raise ValueError(
     f"GO ID {go_id} was not found in {GO_OBO_FILE}."
 )


def collect_ancestor_closure(selected_go_ids, terms):
 closure = set(selected_go_ids)
 stack = list(selected_go_ids)


 while stack:
     child = stack.pop()


     for parent in terms[child]["parents"]:
         if parent not in terms:
             continue


         if parent not in closure:
             closure.add(parent)
             stack.append(parent)


 return closure


def build_graph(node_ids, terms):
 """
 Return:
   children[parent] -> set(children)
   parents[child] -> set(parents)
   edge_type[(parent, child)] -> relationship
 """
 children = defaultdict(set)
 parents = defaultdict(set)
 edge_type = {}


 for node in node_ids:
     children[node]
     parents[node]


 for child in node_ids:
     for parent, relationship in terms[child]["parents"].items():
         if parent not in node_ids:
             continue


         children[parent].add(child)
         parents[child].add(parent)
         edge_type[(parent, child)] = relationship


 return children, parents, edge_type


def remove_node(node, children, parents, edge_type):
 node_parents = list(parents[node])
 node_children = list(children[node])


 for parent in node_parents:
     children[parent].discard(node)
     edge_type.pop((parent, node), None)


 for child in node_children:
     parents[child].discard(node)
     edge_type.pop((node, child), None)


 children.pop(node, None)
 parents.pop(node, None)


def reduce_connectors(
 node_ids,
 children,
 parents,
 edge_type,
 selected_go_ids,
 namespace_roots,
):
 """
 Remove nonselected one-parent/one-child connector terms.
 Branch points, merge points, roots, and selected terms are retained.
 """
 changed = True


 while changed:
     changed = False


     for node in list(node_ids):
         if node in selected_go_ids:
             continue


         if node in namespace_roots:
             continue


         if (
             len(parents[node]) == 1
             and len(children[node]) == 1
         ):
             parent = next(iter(parents[node]))
             child = next(iter(children[node]))


             remove_node(
                 node,
                 children,
                 parents,
                 edge_type,
             )
             node_ids.remove(node)


             children[parent].add(child)
             parents[child].add(parent)
             edge_type[(parent, child)] = "compressed_path"


             changed = True
             break


 return node_ids, children, parents, edge_type


def drop_unused_roots(
 node_ids,
 children,
 parents,
 edge_type,
 selected_go_ids,
):
 """
 Remove roots and upstream branches that do not lead to selected terms.
 """
 useful = set(selected_go_ids)
 stack = list(selected_go_ids)


 while stack:
     node = stack.pop()


     for parent in parents[node]:
         if parent not in useful:
             useful.add(parent)
             stack.append(parent)


 for node in list(node_ids):
     if node not in useful:
         remove_node(
             node,
             children,
             parents,
             edge_type,
         )
         node_ids.remove(node)


 return node_ids, children, parents, edge_type


# ==========================================================
# Load selected pathways
# ==========================================================


if not os.path.exists(CLUSTER_FILE):
 raise FileNotFoundError(
     f"Cluster table not found: {CLUSTER_FILE}\n"
     "Run the ontology-based pathway clustering script first."
 )


selected = pd.read_csv(
 CLUSTER_FILE,
 sep="\t",
 low_memory=False,
)


required_columns = {
 "Pathway",
 "GO_ID",
 "Cluster",
 "Ontology_Cluster_Name",
}


missing = required_columns - set(selected.columns)


if missing:
 raise ValueError(
     "Cluster table is missing required columns: "
     f"{sorted(missing)}"
 )


selected = selected.dropna(
 subset=["Pathway", "GO_ID", "Cluster"]
).copy()


selected["Pathway"] = (
 selected["Pathway"]
 .astype(str)
 .str.strip()
)


selected["GO_ID"] = (
 selected["GO_ID"]
 .astype(str)
 .str.strip()
)


selected["Cluster"] = pd.to_numeric(
 selected["Cluster"],
 errors="raise",
).astype(int)


selected = selected.drop_duplicates(
 subset=["GO_ID"]
).copy()


print(f"Selected GO terms: {len(selected)}")


# ==========================================================
# Extract the minimal GO DAG
# ==========================================================


print("Loading Gene Ontology...")


terms, alt_to_primary = parse_go_obo(
 GO_OBO_FILE
)

# Add forced selected terms that are missing from the cluster table.
# Matching is performed by exact GO term name in go-basic.obo.
existing_go_ids = set(
 selected["GO_ID"].astype(str).str.strip()
)

for forced_name, forced_cluster in FORCE_SELECTED_TERMS.items():
 matching_go_ids = [
     go_id
     for go_id, term in terms.items()
     if term.get("name", "").strip().lower() == forced_name.lower()
 ]

 if not matching_go_ids:
     raise ValueError(
         f'Forced GO term "{forced_name}" was not found in {GO_OBO_FILE}.'
     )

 forced_go_id = matching_go_ids[0]

 if forced_go_id not in existing_go_ids:
     selected = pd.concat(
         [
             selected,
             pd.DataFrame(
                 [
                     {
                         "Pathway": forced_name,
                         "GO_ID": forced_go_id,
                         "Cluster": int(forced_cluster),
                         "Ontology_Cluster_Name": (
                             "Regulation of Catabolic Process"
                         ),
                     }
                 ]
             ),
         ],
         ignore_index=True,
     )
     existing_go_ids.add(forced_go_id)
     print(
         f'Forced selected styling: {forced_name} '
         f'({forced_go_id}), cluster {forced_cluster}'
     )


selected["GO_ID"] = selected["GO_ID"].map(
 lambda go_id: canonical_go_id(
     go_id,
     terms,
     alt_to_primary,
 )
)


selected["GO_Namespace"] = selected["GO_ID"].map(
 lambda go_id: terms[go_id]["namespace"]
)


excluded = selected[
 ~selected["GO_Namespace"].isin(TARGET_NAMESPACES)
].copy()


if not excluded.empty:
 print("Excluding selected terms from unsupported GO namespaces:")
 for _, row in excluded.iterrows():
     print(
         f"  {row['GO_ID']}  "
         f"{terms[row['GO_ID']]['name']}  "
         f"({row['GO_Namespace']})"
     )


selected = selected[
 selected["GO_Namespace"].isin(TARGET_NAMESPACES)
].copy()


if selected.empty:
 raise ValueError(
     "No selected pathways belong to the supported GO namespaces."
 )


print("Selected terms by GO namespace:")
for namespace, count in (
 selected["GO_Namespace"]
 .value_counts()
 .sort_index()
 .items()
):
 print(f"  {namespace}: {count}")


selected_go_ids = set(selected["GO_ID"])


selected_cluster = dict(
 zip(selected["GO_ID"], selected["Cluster"])
)


closure = collect_ancestor_closure(
 selected_go_ids,
 terms,
)


node_ids = set(closure)


children, parents, edge_type = build_graph(
 node_ids,
 terms,
)


namespace_roots = {
 "GO:0008150",
 "GO:0005575",
 "GO:0003674",
}


if DROP_UNUSED_NAMESPACE_ROOTS:
 (
     node_ids,
     children,
     parents,
     edge_type,
 ) = drop_unused_roots(
     node_ids,
     children,
     parents,
     edge_type,
     selected_go_ids,
 )


if not KEEP_ALL_ANCESTORS:
 (
     node_ids,
     children,
     parents,
     edge_type,
 ) = reduce_connectors(
     node_ids,
     children,
     parents,
     edge_type,
     selected_go_ids,
     namespace_roots,
 )


print(f"Subtree nodes: {len(node_ids)}")
print(
 "Subtree edges: "
 f"{sum(len(v) for v in children.values())}"
)


# ==========================================================
# Build Graphviz DOT
# ==========================================================


if shutil.which("dot") is None:
 raise RuntimeError(
     "Graphviz 'dot' was not found.\n"
     "Install it with:\n"
     "  conda install -c conda-forge graphviz\n"
     "or load a Graphviz module on the cluster."
 )


cluster_summary = (
 selected[
     ["Cluster", "Ontology_Cluster_Name"]
 ]
 .drop_duplicates()
 .sort_values("Cluster")
)


dot_lines = [
 "digraph GO_Subtree {",
 f'  rankdir="{RANK_DIRECTION}";',
 '  graph [',
 '    bgcolor="white",',
 '    pad="0.01",',
 '    margin="0.0",',
 f'    ranksep="{RANK_SEPARATION}",',
 f'    nodesep="{NODE_SEPARATION}",',
 '    splines="ortho",',
 '    outputorder="edgesfirst",',
 '    overlap="false",',
 '    pack="true",',
 '    packmode="array_t1",',
 '    ratio="compress",',
 '    concentrate="false",',
 '    newrank="true",',
 '    center="true",',
 '    fontname="Arial",',
 '    labelloc="t",',
 '    labeljust="c",',
 f'    fontsize="{TITLE_FONT_SIZE}",',
 '    label="Gene Ontology Hierarchy of Enriched Pathways"',
 "  ];",
 '  node [',
 '    shape="box",',
 '    style="rounded,filled",',
 '    fontname="Arial",',
 f'    color="{NODE_BORDER}",',
 f'    fontcolor="{TEXT_COLOR}",',
 '    penwidth="1.0",',
 '    margin="0.11,0.07"',
 "  ];",
 '  edge [',
 f'    color="{EDGE_COLOR}",',
 '    penwidth="1.30",',
 '    arrowsize="0.62"',
 "  ];",
]


# Create one subgraph per represented namespace. This improves spacing and
# prevents terms from different ontologies from visually intermixing.
namespace_order = {
 "cellular_component": 0,
 "biological_process": 1,
 "molecular_function": 2,
}


represented_namespaces = sorted(
 {
     terms[go_id]["namespace"]
     for go_id in node_ids
 },
 key=lambda namespace: namespace_order.get(namespace, 99),
)


for namespace_index, namespace in enumerate(represented_namespaces):
 namespace_nodes = sorted(
     go_id
     for go_id in node_ids
     if terms[go_id]["namespace"] == namespace
 )


 dot_lines.extend(
     [
         f'  subgraph "cluster_namespace_{namespace_index}" {{',
         '    label="";',
         f'    sortv="{namespace_index + 1}";',
         '    color="white";',
         '    penwidth="0.0";',
         '    style="solid";',
         '    margin="3";',
         '    labelloc="t";',
         '    labeljust="c";',
         '    fontsize="1";',
         '    fontname="Arial Bold";',
     ]
 )


 for go_id in namespace_nodes:
     dot_lines.append(f'    "{go_id}";')




 dot_lines.append("  }")


# Nodes
for go_id in sorted(node_ids):
 term_name = terms[go_id]["name"]


 if go_id in selected_go_ids:
     label = wrap_label(
         term_name,
         SELECTED_LABEL_WRAP_WIDTH,
     )
     if SHOW_GO_IDS:
         label = f"{label}\\n{go_id}"


     fill = cluster_color(
         selected_cluster[go_id]
     )


     attrs = {
         "label": label,
         "fillcolor": fill,
         "fontcolor": "#FFFFFF",
         "fontsize": str(SELECTED_FONT_SIZE),
         "penwidth": "1.55",
         "width": str(SELECTED_NODE_WIDTH),
         "height": str(SELECTED_NODE_HEIGHT),
         "fontname": "Arial Bold",
     }


 elif len(parents[go_id]) == 0:
     label = wrap_label(
         term_name,
         ANCESTOR_LABEL_WRAP_WIDTH,
     )
     attrs = {
         "label": label,
         "fillcolor": ROOT_FILL,
         "fontcolor": TEXT_COLOR,
         "fontsize": "16",
         "penwidth": "1.45",
         "width": "2.35",
         "height": "0.72",
         "fontname": "Arial Bold",
     }


 else:
     label = wrap_label(
         term_name,
         ANCESTOR_LABEL_WRAP_WIDTH,
     )
     attrs = {
         "label": label,
         "fillcolor": ANCESTOR_FILL,
         "fontcolor": TEXT_COLOR,
         "fontsize": str(ANCESTOR_FONT_SIZE),
         "penwidth": "0.85",
         "width": str(ANCESTOR_NODE_WIDTH),
         "height": str(ANCESTOR_NODE_HEIGHT),
         "fontname": "Arial",
     }


 attr_text = ", ".join(
     f'{key}="{dot_escape(value)}"'
     for key, value in attrs.items()
 )


 dot_lines.append(
     f'  "{go_id}" [{attr_text}];'
 )


# Edges
for parent in sorted(node_ids):
 for child in sorted(children[parent]):
     if child not in node_ids:
         continue


     relationship = edge_type.get(
         (parent, child),
         "is_a",
     )


     if relationship == "part_of":
         attrs = 'style="dashed", penwidth="1.0"'
     elif relationship == "compressed_path":
         attrs = 'style="solid", penwidth="1.05", color="#B8B8B8"'
     else:
         attrs = 'style="solid"'


     dot_lines.append(
         f'  "{parent}" -> "{child}" [{attrs}];'
     )


# Legend
legend_entries = []


for _, row in cluster_summary.iterrows():
 cluster = int(row["Cluster"])
 cluster_name = display_title(
     row["Ontology_Cluster_Name"]
 )
 color = cluster_color(cluster)


 legend_entries.append(
     (
         f'<TD FIXEDSIZE="TRUE" WIDTH="18" HEIGHT="18" '
         f'BGCOLOR="{color}"></TD>'
         f'<TD ALIGN="LEFT"><FONT FACE="Arial" POINT-SIZE="{LEGEND_FONT_SIZE}">'
         f'Cluster {cluster}: {cluster_name}</FONT></TD>'
     )
 )


legend_entries.extend(
 [
     (
         f'<TD FIXEDSIZE="TRUE" WIDTH="18" HEIGHT="18" '
         f'BGCOLOR="{ANCESTOR_FILL}"></TD>'
         f'<TD ALIGN="LEFT"><FONT FACE="Arial" POINT-SIZE="{LEGEND_FONT_SIZE}">'
         f'Connecting GO ancestor</FONT></TD>'
     ),
     (
         f'<TD FIXEDSIZE="TRUE" WIDTH="18" HEIGHT="18" '
         f'BGCOLOR="{ROOT_FILL}"></TD>'
         f'<TD ALIGN="LEFT"><FONT FACE="Arial" POINT-SIZE="{LEGEND_FONT_SIZE}">'
         f'GO namespace root</FONT></TD>'
     ),
 ]
)


# Use at most three entries per row for readable legend text.
legend_rows = []
for start in range(0, len(legend_entries), 3):
 legend_rows.append(
     "<TR>"
     + "".join(legend_entries[start:start + 3])
     + "</TR>"
 )


legend_html = (
 '<<TABLE BORDER="1" COLOR="#666666" BGCOLOR="white" CELLBORDER="0" CELLSPACING="20" '
 'CELLPADDING="8">'
 '<TR><TD COLSPAN="6" ALIGN="CENTER"><FONT FACE="Arial" POINT-SIZE="22"><B>Figure Key</B></FONT></TD></TR>'
 + "".join(legend_rows)
 + '</TABLE>>'
)


# Place the legend beneath the ontology and center it across the figure.
dot_lines.extend(
[
    '  legend [',
    '    margin="0",',
    '    shape="plain",',
    f'    label={legend_html}',
    '  ];',
    '  { rank=sink; legend; }',
    "}",
]
)


dot_text = "\n".join(dot_lines)


# ==========================================================
# Render output
# ==========================================================


os.makedirs("results", exist_ok=True)


with tempfile.TemporaryDirectory() as temp_dir:
 dot_file = os.path.join(
     temp_dir,
     "significant_pathway_go_subtree.dot",
 )


 with open(
     dot_file,
     "w",
     encoding="utf-8",
 ) as handle:
     handle.write(dot_text)


 subprocess.run(
     [
         "dot",
         "-Tpdf",
         dot_file,
         "-o",
         OUTPUT_PDF,
     ],
     check=True,
 )


 subprocess.run(
     [
         "dot",
         "-Tsvg",
         dot_file,
         "-o",
         OUTPUT_SVG,
     ],
     check=True,
 )


 subprocess.run(
     [
         "dot",
         "-Tpng",
         f"-Gdpi={PNG_DPI}",
         dot_file,
         "-o",
         OUTPUT_PNG,
     ],
     check=True,
 )


print(f"Saved subtree PDF: {OUTPUT_PDF}")
print(f"Saved subtree SVG: {OUTPUT_SVG}")
print(f"Saved subtree PNG: {OUTPUT_PNG}")
