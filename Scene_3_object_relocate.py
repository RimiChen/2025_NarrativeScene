import json
import numpy as np
import difflib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- CONFIGURABLE VARIABLES ---
story_id = 0
input_layer_file = f"StoryFiles/{story_id}_scene_object_affordance_layers.json"
input_summary_file = f"StoryFiles/{story_id}_scene_summaries.json"
output_layer_file = f"StoryFiles/{story_id}_scene_object_affordance_layers_RELOCATED.json"
output_img_folder = f"StoryFiles/{story_id}_relocation_visualizations"
os.makedirs(output_img_folder, exist_ok=True)

# --- HELPER FUNCTIONS ---
def check_relation(source_pos, target_pos, relation):
    sx, sy = source_pos
    tx, ty = target_pos
    return (
        (relation == "at the left of" and sx == tx and sy == ty - 3)
        or (relation == "at the right of" and sx == tx and sy == ty + 3)
        or (relation == "above" and sx == tx - 3 and sy == ty)
        or (relation == "below" and sx == tx + 3 and sy == ty)
        or (relation == "on top of" and sx == tx and sy == ty)
    )

def apply_relation(pos, relation):
    x, y = pos
    return {
        "at the left of": (x, y - 1),
        "at the right of": (x, y + 1),
        "above": (x - 1, y),
        "below": (x + 1, y),
        "on top of": (x, y)
    }.get(relation, (x, y))

def normalize_name(name, available, alias_dict, cutoff=0.6):
    if name.lower() in alias_dict:
        return alias_dict[name.lower()]
    matches = difflib.get_close_matches(name.lower().replace(" ", "_"), available, n=1, cutoff=cutoff)
    return matches[0] if matches else None

# --- ALIAS DICTIONARY (EXTENDABLE) ---
alias_dict = {
    "hollow oak": "hollow_oak",
    "ancient map": "ancient_map",
    "guardian dragon": "guardian_dragon",
    "crystal throne": "crystal_throne",
    "crystal cavern entrance": "crystal_cavern_entrance",
    "dense bushes": "dense_bushes",
    "wild creatures": "wild_creatures",
    "crystal cavern": "crystal_cavern",
    "rocky path": "rocky_path",
    "forest canopy": "forest_canopy",
    "shimmering light": "shimmering_light",
    "treacherous paths": "treacherous_paths",
}

# --- LOAD FILES ---
with open(input_layer_file, "r", encoding="utf-8") as f:
    scene_layers = json.load(f)

with open(input_summary_file, "r", encoding="utf-8") as f:
    scene_summaries = json.load(f)

# --- PROCESS EACH SCENE ---
for scene in scene_summaries:
    title = scene["scene_title"]
    relations = scene.get("spatial_relations", [])
    layers = scene_layers[title]

    # Collect all available normalized object names
    all_names = []
    for obj_type in ["characters", "items", "interactive_objects", "environment_objects"]:
        all_names.extend([n.lower().replace(" ", "_") for n in scene.get(obj_type, [])])

    # Build object name to layer/position map
    name_to_layer = {}
    for key, obj_type in {
        "matrix_character": "characters",
        "matrix_item": "items",
        "matrix_interactive": "interactive_objects",
        "matrix_environment": "environment_objects",
    }.items():
        matrix = np.array(layers[key])
        ys, xs = np.where(matrix == 1)
        names = scene.get(obj_type, [])
        for (x, y), name in zip(zip(xs, ys), names):
            norm = name.lower().replace(" ", "_")
            name_to_layer[norm] = (key, (y, x))

    # Apply spatial relocation
    for rel in relations:
        raw_source, raw_target, relation = rel["source"], rel["target"], rel["relation"]
        source = normalize_name(raw_source, all_names, alias_dict)
        target = normalize_name(raw_target, all_names, alias_dict)

        if not source or not target:
            continue

        tgt_layer, tgt_pos = name_to_layer.get(target, (None, None))
        src_layer, src_pos = name_to_layer.get(source, (None, None))

        if tgt_layer and tgt_pos:
            x_new, y_new = apply_relation(tgt_pos, relation)
            h, w = np.array(layers["matrix_base"]).shape
            if 0 <= x_new < h and 0 <= y_new < w:
                if src_layer and src_pos:
                    x_old, y_old = src_pos
                    layers[src_layer][x_old][y_old] = 0
                elif not src_layer:
                    for key_check, names in {
                        "matrix_character": "characters",
                        "matrix_item": "items",
                        "matrix_interactive": "interactive_objects",
                        "matrix_environment": "environment_objects"
                    }.items():
                        if source in [n.lower().replace(" ", "_") for n in scene.get(names, [])]:
                            src_layer = key_check
                            break
                if src_layer:
                    layers[src_layer][x_new][y_new] = 1
                    name_to_layer[source] = (src_layer, (x_new, y_new))

    # --- VISUALIZE ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(title)
    ax.axis("off")
    base = np.array(layers["matrix_base"])
    ax.imshow(base, cmap="Greys", alpha=0.8, zorder=0)

    type_map = {
        "matrix_character": ("red", "characters"),
        "matrix_item": ("blue", "items"),
        "matrix_interactive": ("green", "interactive_objects"),
        "matrix_environment": ("orange", "environment_objects")
    }

    for key, (color, obj_type) in type_map.items():
        matrix = np.array(layers[key])
        ys, xs = np.where(matrix == 1)
        names = scene.get(obj_type, [])
        for (x, y), name in zip(zip(xs, ys), names):
            ax.scatter(x, y, s=80, color=color, edgecolors="black", zorder=2)
            ax.text(
                x + 0.5, y, name, fontsize=9, fontweight='bold',
                color='black', ha='left', va='center', zorder=3,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.8)
            )

    # handles = [mpatches.Patch(color=c, label=l.title()) for _, (c, l) in type_map.items()]
    # ax.legend(handles=handles, loc="upper right")
    # plt.tight_layout()
    handles = [mpatches.Patch(color=c, label=l.title()) for _, (c, l) in type_map.items()]
    # Move legend outside the plot
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room on the right for legend    
    output_path = os.path.join(output_img_folder, f"{title.replace(' ', '_')}_placement.png")
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved visualization: {output_path}")

# --- SAVE OUTPUT LAYERS ---
with open(output_layer_file, "w", encoding="utf-8") as f:
    json.dump(scene_layers, f, indent=2)
print(f"✅ Saved relocated object layers to {output_layer_file}")
