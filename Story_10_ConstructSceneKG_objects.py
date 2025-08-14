import json
import networkx as nx
import matplotlib.pyplot as plt

# === Input Files ===
STORY_ID = "0"
base_path = "StoryFiles/"
scene_file = f"{base_path}{STORY_ID}_scene_generation_decisions.json"
env_file = f"{base_path}{STORY_ID}_scene_object_propagation_CONVERTED.json"
char_file = f"{base_path}{STORY_ID}_character_objects_expanded.json"
interact_file = f"{base_path}{STORY_ID}_interactive_objects_expanded.json"
item_file = f"{base_path}{STORY_ID}_item_collectibles_expanded.json"

# === Load Data ===
def load_json(path): return json.load(open(path, "r", encoding="utf-8"))
scene_data = load_json(scene_file)
env_data = load_json(env_file)
char_data = load_json(char_file)
interact_data = load_json(interact_file)
item_data = load_json(item_file)

# === Helper: convert list of {title, objects} into scene -> object list
# === Replace the function with this ===
def scene_object_dict(data, key_options):
    out = {}
    for d in data:
        title = d.get("title") or d.get("scene_title")
        for key in key_options:
            if key in d:
                out[title] = d[key]
                break
        else:
            out[title] = []
    return out

# === Use these to collect the scene-object mappings ===
env_map = scene_object_dict(env_data, ["propagated_objects"])
char_map = scene_object_dict(char_data, ["characters"])  # ✅ correct key
interact_map = scene_object_dict(interact_data, ["interactive_objects"])  # ✅ correct key
item_map = scene_object_dict(item_data, ["items"])  # ✅ correct key

# === Build KGs ===
scene_kgs = []
for scene in scene_data:
    title = scene["scene_title"]
    triples = []

    triples.append([title, "has_base", scene["chosen_base"]])
    # for patch in scene.get("final_patch_for_base", []):
    #     triples.append([title, "has_patch", patch])
    for patch in scene.get("final_patch_for_base", []):
        if patch != "<no patch>":
            triples.append([title, "has_patch", patch])    
    for o in env_map.get(title, []):
        triples.append([title, "has_environment_object", o])
    for c in char_map.get(title, []):
        triples.append([title, "has_character", c])
    for i in item_map.get(title, []):
        triples.append([title, "has_item", i])
    for i in interact_map.get(title, []):
        triples.append([title, "has_interactive_object", i])

    scene_kgs.append({"scene_title": title, "triples": triples})

# === Save
with open(f"{base_path}{STORY_ID}_scene_kg_combined.json", "w", encoding="utf-8") as f:
    json.dump(scene_kgs, f, indent=2)

# === Visualize
fig, axes = plt.subplots(len(scene_kgs), 1, figsize=(10, 5 * len(scene_kgs)))
if len(scene_kgs) == 1: axes = [axes]
for i, entry in enumerate(scene_kgs):
    G = nx.DiGraph()
    for s, p, o in entry["triples"]:
        G.add_edge(s, o, label=p)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=axes[i], with_labels=True, node_color="lightblue", node_size=2500, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(s, o): p for s, o, p in G.edges(data='label')}, ax=axes[i])
    axes[i].set_title(entry["scene_title"], fontsize=12)

plt.tight_layout()
plt.show()
