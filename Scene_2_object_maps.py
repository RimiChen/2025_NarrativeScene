import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import os

# === CHANGE STORY ID HERE ===
STORY_ID = 0

# === Load Files ===
tile_file = f"StoryFiles/{STORY_ID}_tile_matrix_with_objects.json"
summary_file = f"StoryFiles/{STORY_ID}_scene_summaries.json"
with open(tile_file, "r", encoding="utf-8") as f:
    raw_matrix_data = json.load(f)
with open(summary_file, "r", encoding="utf-8") as f:
    scene_data = json.load(f)

# === Reformat matrix data by scene title ===
scene_names = list(raw_matrix_data["base_maps"].keys())
matrix_data = {
    name: {
        "matrix_base": raw_matrix_data["base_maps"][name],
        "matrix_patch": raw_matrix_data["patch_maps"].get(name, [[0]])
    }
    for name in scene_names
}

# === Reformat scene summary list to dict ===
scene_data_index = {entry["scene_title"]: entry for entry in scene_data}

# === Affordance Labels & Colors ===
affordance_keys = {
    "Characters": "matrix_character",
    "Items and Collectibles": "matrix_item",
    "Interactive Object": "matrix_interactive",
    "Environmental Object": "matrix_environment"
}
affordance_colors = {
    "Characters": "red",
    "Items and Collectibles": "blue",
    "Interactive Object": "green",
    "Environmental Object": "orange"
}

# === Output Containers ===
output = {}
output_folder = f"StoryFiles/scene_visualizations_{STORY_ID}"
os.makedirs(output_folder, exist_ok=True)

# === Main Placement Loop ===
for scene_title, terrain in matrix_data.items():
    base = np.array(terrain["matrix_base"])
    patch = np.array(terrain["matrix_patch"])
    H, W = base.shape

    layers = {k: np.zeros((H, W), dtype=int) for k in affordance_keys.values()}
    name_positions = {}

    # Get all scene objects with their affordance type
    scene_entry = scene_data_index.get(scene_title, {})
    objects = []
    for category, aff in [
        ("characters", "Characters"),
        ("items", "Items and Collectibles"),
        ("interactive_objects", "Interactive Object"),
        ("environment_objects", "Environmental Object")
    ]:
        for obj in scene_entry.get(category, []):
            objects.append({"name": obj["name"], "affordance": aff})

    # Random Placement per object
    for obj in objects:
        name = obj["name"]
        aff = obj["affordance"]
        layer_key = affordance_keys.get(aff)
        if not layer_key:
            continue

        valid_mask = (base == 1) | (patch == 2)
        possible_positions = list(zip(*np.where(valid_mask & (layers[layer_key] == 0))))
        if not possible_positions:
            continue

        x, y = random.choice(possible_positions)
        layers[layer_key][x, y] = 1
        name_positions[name] = (x, y, aff)

    # === Visualization ===
    fig, ax = plt.subplots(figsize=(8, 6))
    combined = base + patch
    masked = np.ma.masked_where(combined == 0, combined)
    ax.imshow(masked, cmap="viridis", alpha=0.8)
    for name, (x, y, aff) in name_positions.items():
        ax.plot(y, x, 'o', color=affordance_colors[aff])
        ax.text(y, x, name, fontsize=6, color='black', ha='center', va='center')
    handles = [mpatches.Patch(color=col, label=label) for label, col in affordance_colors.items()]
    ax.legend(handles=handles, loc='upper right')
    ax.set_title(f"Scene: {scene_title}")
    ax.axis('off')
    plt.tight_layout()
    filename = os.path.join(output_folder, f"{scene_title[:30].replace(' ', '_')}_placement.png")
    plt.savefig(filename)
    plt.close()

    # Save Output
    output[scene_title] = {
        "matrix_base": terrain["matrix_base"],
        "matrix_patch": terrain["matrix_patch"],
        **{k: layers[k].tolist() for k in layers}
    }

# === Save Output JSON ===
output_json_path = f"StoryFiles/{STORY_ID}_scene_object_affordance_layers.json"
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"[✓] Done! Saved affordance layers to: {output_json_path}")
print(f"[✓] Visualizations saved to: {output_folder}")
