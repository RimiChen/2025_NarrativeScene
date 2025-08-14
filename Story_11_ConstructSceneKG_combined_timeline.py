import json
import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from difflib import SequenceMatcher

# === Configuration ===
STORY_ID = "0"  # Change this to your story ID
BASE_FOLDER = "StoryFiles/"
COMBINED_FILE = f"{BASE_FOLDER}{STORY_ID}_scene_kg_combined.json"
RELATIONS_FILE = f"{BASE_FOLDER}{STORY_ID}_scene_from_relations.json"
TIMELINE_FILE = f"{BASE_FOLDER}{STORY_ID}_adventure_scene_output_FIXED.json"
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

with open(TIMELINE_FILE, "r", encoding="utf-8") as f:
    timeline_data = json.load(f)
scene_order = [s["title"] for s in timeline_data["time_frames"]]

# === Build Scene-Level KGs ===
scene_kgs = []
scene_graphs = {}
global_nodes = set()
merged_graph = nx.DiGraph()

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

    # Draw and save
    plt.figure(figsize=(10, 6))
    pos = nx.shell_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=1500, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['label'] for u, v, d in G.edges(data=True)}, font_color="red")
    plt.title(f"Scene KG: {title}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{normalize(title)}_kg.png"))
    plt.close()

    global_nodes.update(nodes)
    merged_graph.add_nodes_from(nodes)
    merged_graph.add_edges_from(G.edges(data=True))
    scene_kgs.append((title, G))

# === Add Temporal Links ===
for i in range(len(scene_order) - 1):
    merged_graph.add_edge(scene_order[i], scene_order[i + 1], label="precedes")

# === Assign Subset Attributes for Layout ===
for node in merged_graph.nodes():
    merged_graph.nodes[node]["subset"] = 0 if node in scene_order else 1

# === Draw and Save Merged Graph ===
plt.figure(figsize=(12, 8))
pos = nx.multipartite_layout(merged_graph, subset_key="subset")
nx.draw(merged_graph, pos, with_labels=True, node_color="lightgreen", node_size=1800, font_size=9)
nx.draw_networkx_edge_labels(merged_graph, pos, edge_labels={(u, v): d['label'] for u, v, d in merged_graph.edges(data=True)}, font_color="brown")
plt.title("Merged Story-Level KG")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{STORY_ID}_merged_kg.png"))
plt.close()

# === Save Structured KG Data ===
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
