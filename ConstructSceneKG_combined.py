import json
import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from difflib import SequenceMatcher

# === Configuration ===
STORY_ID = "1"  # Change this to switch story
BASE_FOLDER = "StoryFiles/"
COMBINED_FILE = f"{BASE_FOLDER}{STORY_ID}_scene_kg_combined.json"
RELATIONS_FILE = f"{BASE_FOLDER}{STORY_ID}_scene_from_relations.json"
OUTPUT_DIR = f"{BASE_FOLDER}output_KG_story_{STORY_ID}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Helpers ===
def normalize(name):
    return name.lower().replace(" ", "_")

def is_similar(a, b, threshold=0.85):
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio() >= threshold

def merge_node(name, existing_nodes):
    for existing in existing_nodes:
        if is_similar(name, existing):
            return existing
    return name

# === Load Data ===
with open(COMBINED_FILE, "r", encoding="utf-8") as f:
    object_data = json.load(f)

with open(RELATIONS_FILE, "r", encoding="utf-8") as f:
    relation_data = json.load(f)

scene_kgs = []
scene_graphs = {}
global_nodes = set()
merged_graph = nx.DiGraph()

# === Build KGs per scene ===
for obj_scene, rel_scene in zip(object_data, relation_data):
    title = obj_scene["scene_title"]
    G = nx.DiGraph()
    nodes = set()

    # Add object triples
    for s, p, o in obj_scene["triples"]:
        source = merge_node(s, nodes)
        target = merge_node(o, nodes)
        G.add_edge(source, target, label=p)
        nodes.update([source, target])

    # Add relation triples
    for triple in rel_scene["triples"]:
        s = merge_node(triple["source"], nodes)
        t = merge_node(triple["target"], nodes)
        r = triple["relation"]
        G.add_edge(s, t, label=r)
        nodes.update([s, t])

    # Save scene-level graph
    scene_graphs[title] = {
        "nodes": list(nodes),
        "edges": [
            {"source": u, "target": v, "relation": d["label"]}
            for u, v, d in G.edges(data=True)
        ]
    }

    # Draw scene KG
    plt.figure(figsize=(10, 6))
    pos = nx.shell_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=1500, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['label'] for u, v, d in G.edges(data=True)}, font_color="red")
    plt.title(f"Scene KG: {title}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{normalize(title)}_kg.png"))
    plt.close()

    # Track for merged KG
    global_nodes.update(nodes)
    merged_graph.add_nodes_from(nodes)
    merged_graph.add_edges_from(G.edges(data=True))
    scene_kgs.append((title, G))

# === Add temporal links ===
scene_titles = [s for s, _ in scene_kgs]
for i in range(len(scene_titles) - 1):
    merged_graph.add_edge(scene_titles[i], scene_titles[i+1], label="precedes")

# === Draw merged KG ===
plt.figure(figsize=(12, 8))
# pos = nx.multipartite_layout(merged_graph, subset_key=lambda n: 0 if n in scene_titles else 1)
# Add subset attributes to distinguish scene vs object nodes
for node in merged_graph.nodes():
    if node in scene_titles:
        merged_graph.nodes[node]["subset"] = 0  # Scene level
    else:
        merged_graph.nodes[node]["subset"] = 1  # Object level

# Now layout using these subsets
pos = nx.multipartite_layout(merged_graph, subset_key="subset")

nx.draw(merged_graph, pos, with_labels=True, node_color="lightgreen", node_size=1800, font_size=9)
nx.draw_networkx_edge_labels(merged_graph, pos, edge_labels={(u, v): d['label'] for u, v, d in merged_graph.edges(data=True)}, font_color="brown")
plt.title("Merged Story-Level KG")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{STORY_ID}_merged_kg.png"))
plt.close()

# === Save structured KG data ===
scene_kg_output = {
    "story_id": STORY_ID,
    "scene_kgs": scene_graphs,
    "merged_kg": {
        "nodes": list(merged_graph.nodes),
        "edges": [
            {"source": u, "target": v, "relation": d["label"]}
            for u, v, d in merged_graph.edges(data=True)
        ]
    }
}
with open(os.path.join(OUTPUT_DIR, f"{STORY_ID}_kg_data.json"), "w", encoding="utf-8") as f:
    json.dump(scene_kg_output, f, indent=2)
