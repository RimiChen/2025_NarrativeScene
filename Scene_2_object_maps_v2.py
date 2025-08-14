import json, random, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -------- CONFIG --------
story_id = 0  # Change this for different stories
scene_summary_file = f"StoryFiles/{story_id}_scene_summaries.json"
tile_matrix_file = f"StoryFiles/{story_id}_tile_matrix_with_objects.json"
output_folder = f"StoryFiles/{story_id}_placement_visualizations_story"
output_json = f"StoryFiles/{story_id}_scene_object_affordance_layers.json"
# ------------------------

# Load data
with open(tile_matrix_file, "r", encoding="utf-8") as f:
    matrix_data = json.load(f)
with open(scene_summary_file, "r", encoding="utf-8") as f:
    scene_summaries = json.load(f)

# Create output directory
os.makedirs(output_folder, exist_ok=True)

# Object type colors
color_map = {
    "character": "red",
    "item": "blue",
    "interactive_object": "green",
    "environment_object": "orange"
}

# Mapping for object type → matrix layer key
layer_key_map = {
    "character": "matrix_character",
    "item": "matrix_item",
    "interactive_object": "matrix_interactive",
    "environment_object": "matrix_environment"
}

# Store placement and layers
all_layer_maps = {}

for scene in scene_summaries:
    scene_title = scene["scene_title"]
    base_name = scene["base"]
    base_matrix = np.array(matrix_data["base_maps"][base_name])
    H, W = base_matrix.shape

    walkable = list(zip(*np.where(base_matrix == 1)))
    random.shuffle(walkable)

    obj_entries = (
        [(n, "character") for n in scene.get("characters", [])] +
        [(n, "item") for n in scene.get("items", [])] +
        [(n, "interactive_object") for n in scene.get("interactive_objects", [])] +
        [(n, "environment_object") for n in scene.get("environment_objects", [])]
    )

    # Prepare layers
    layers = {
        "matrix_base": base_matrix.tolist(),
        "matrix_character": np.zeros((H, W), dtype=int),
        "matrix_item": np.zeros((H, W), dtype=int),
        "matrix_interactive": np.zeros((H, W), dtype=int),
        "matrix_environment": np.zeros((H, W), dtype=int),
    }

    placements = {}
    for name, obj_type in obj_entries:
        if not walkable:
            break
        x, y = walkable.pop()
        key = layer_key_map[obj_type]
        layers[key][x, y] = 1
        placements[name] = {"type": obj_type, "position": [int(x), int(y)]}

    all_layer_maps[scene_title] = layers

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(scene_title)
    ax.axis("off")
    ax.imshow(base_matrix, cmap="Greys", alpha=0.8, zorder=0)

    for name, info in placements.items():
        x, y = info["position"]
        color = color_map[info["type"]]
        ax.scatter(y, x, s=100, color=color, edgecolors='black', zorder=3)
        ax.text(y + 0.5, x, name, fontsize=8, fontweight='bold',
                color='black', ha='left', va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.8))

    handles = [mpatches.Patch(color=c, label=t.title()) for t, c in color_map.items()]
    ax.legend(handles=handles, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{scene_title.replace(' ', '_')}_placement.png"))
    plt.close()

# ✅ Convert NumPy arrays to lists before saving
serializable_layer_maps = {
    scene_title: {
        layer: arr.tolist() if isinstance(arr, np.ndarray) else arr
        for layer, arr in scene_layers.items()
    }
    for scene_title, scene_layers in all_layer_maps.items()
}

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(serializable_layer_maps, f, indent=2)
