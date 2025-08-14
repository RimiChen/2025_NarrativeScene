import json, os, random
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
SCENE_FILE = "single_scene_forest.json"
OUTPUT_DIR = "StoryFiles/single_scene_forest"
STORY_ID = "forest"
TILE_SIZE = 32
MAP_WIDTH, MAP_HEIGHT = 20, 15
TERRAIN_TYPES = ["grass", "forest", "dirt"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD SCENE ---
with open(SCENE_FILE, "r", encoding="utf-8") as f:
    scene = json.load(f)

title = scene["scene_title"]
objects = scene["environment_objects"]
relations = scene.get("spatial_relations", [])

# --- TERRAIN BASE GENERATION ---
def generate_terrain_matrix(height, width, terrain_types):
    terrain_matrix = np.random.choice(terrain_types, size=(height, width), p=[0.6, 0.3, 0.1])
    return terrain_matrix

terrain_matrix = generate_terrain_matrix(MAP_HEIGHT, MAP_WIDTH, TERRAIN_TYPES)

# --- INIT object map ---
object_matrix = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=int)
name_to_pos = {}

# --- RANDOMLY PLACE OBJECTS ---
available = list(zip(*np.where(object_matrix == 0)))
random.shuffle(available)

for name in objects:
    if not available:
        break
    y, x = available.pop()
    object_matrix[y, x] = 1
    name_to_pos[name.lower()] = (y, x)

# --- APPLY RELATIONS ---
def apply_relation(pos, relation):
    y, x = pos
    return {
        "at the left of": (y, x - 3),
        "at the right of": (y, x + 3),
        "above": (y - 3, x),
        "below": (y + 3, x),
        "on top of": (y - 1, x),
    }.get(relation, (y, x))

for rel in relations:
    src, tgt, rel_type = rel["source"], rel["target"], rel["relation"]
    src, tgt = src.lower(), tgt.lower()
    if tgt not in name_to_pos:
        continue
    tgt_pos = name_to_pos[tgt]
    new_pos = apply_relation(tgt_pos, rel_type)
    ny, nx = new_pos
    if 0 <= ny < MAP_HEIGHT and 0 <= nx < MAP_WIDTH:
        old_pos = name_to_pos.get(src)
        if old_pos:
            object_matrix[old_pos[0], old_pos[1]] = 0
        name_to_pos[src] = new_pos
        object_matrix[ny, nx] = 1

# --- SAVE JSON (terrain + object map) ---
output_json = {
    title: {
        "matrix_base": terrain_matrix.tolist(),
        "matrix_environment": object_matrix.tolist()
    }
}
with open(os.path.join(OUTPUT_DIR, f"{STORY_ID}_scene_object_affordance_layers.json"), "w") as f:
    json.dump(output_json, f, indent=2)

# --- VISUALIZATION ---
terrain_colors = {
    "grass": "#a8d08d",
    "forest": "#4b8b3b",
    "dirt": "#c2b280"
}

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title(f"Scene: {title}")
ax.set_xticks([])
ax.set_yticks([])

# Render terrain
for y in range(MAP_HEIGHT):
    for x in range(MAP_WIDTH):
        t = terrain_matrix[y, x]
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=terrain_colors.get(t, "gray")))

# Render objects with labels
for name, (y, x) in name_to_pos.items():
    ax.scatter(x + 0.5, y + 0.5, s=300, color="orange", edgecolors='black', zorder=3)
    ax.text(x + 0.6, y + 0.4, name, fontsize=9, fontweight='bold',
            color='black', ha='left', va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.8))

ax.set_xlim(0, MAP_WIDTH)
ax.set_ylim(MAP_HEIGHT, 0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{STORY_ID}_layout_with_terrain.png"))
plt.show()
