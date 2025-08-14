import json, os, random
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
SCENE_FILE = "StoryFiles/single_scene_forest.json"
OUTPUT_DIR = "StoryFiles/single_scene_forest_fixed"
STORY_ID = "forest"
MAP_WIDTH, MAP_HEIGHT = 20, 15
TERRAIN_TYPES = ["grass", "forest", "dirt"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD SCENE ---
with open(SCENE_FILE, "r", encoding="utf-8") as f:
    scene = json.load(f)

title = scene.get("scene_title", "Scene")
relations = scene.get("spatial_relations", [])

# Fix: Extract object names from relations
objects = set()
for rel in relations:
    objects.add(rel["source"].strip().lower())
    objects.add(rel["target"].strip().lower())

# --- TERRAIN GENERATION ---
def generate_terrain_matrix(h, w, types):
    return np.random.choice(types, size=(h, w), p=[0.6, 0.3, 0.1])

terrain_matrix = generate_terrain_matrix(MAP_HEIGHT, MAP_WIDTH, TERRAIN_TYPES)
object_matrix = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=int)

# --- OBJECT PLACEMENT ---
name_to_pos = {}
placed_objects = set()

def get_empty_pos():
    trials = 100
    for _ in range(trials):
        y, x = random.randint(0, MAP_HEIGHT - 1), random.randint(0, MAP_WIDTH - 1)
        if object_matrix[y, x] == 0:
            return (y, x)
    return None

def apply_relation(pos, rel):
    y, x = pos
    offset = {
        "at the left of": (0, -3),
        "at the right of": (0, 3),
        "above": (-3, 0),
        "below": (3, 0),
        "on top of": (-1, 0),
    }.get(rel, (0, 0))
    return (y + offset[0], x + offset[1])

# --- PROCESS RELATIONS FIRST ---
for rel in relations:
    source = rel["source"].strip().lower()
    target = rel["target"].strip().lower()
    rtype = rel["relation"]

    if source in placed_objects:
        continue

    if target not in placed_objects:
        tgt_pos = get_empty_pos()
        if tgt_pos:
            name_to_pos[target] = tgt_pos
            object_matrix[tgt_pos[0], tgt_pos[1]] = 1
            placed_objects.add(target)
            print(f"Placed target '{target}' at {tgt_pos}")
        else:
            print(f"⚠️ Cannot place target '{target}' — no space")

    if target in name_to_pos:
        src_pos = apply_relation(name_to_pos[target], rtype)
        y, x = src_pos
        if 0 <= y < MAP_HEIGHT and 0 <= x < MAP_WIDTH and object_matrix[y, x] == 0:
            name_to_pos[source] = (y, x)
            object_matrix[y, x] = 1
            placed_objects.add(source)
            print(f"Placed '{source}' {rtype} '{target}' → {src_pos}")
        else:
            print(f"❌ Cannot place '{source}' {rtype} '{target}' → {src_pos} (invalid)")
    else:
        print(f"⚠️ Target '{target}' not placed, skipping '{source}'")

# --- PLACE REMAINING OBJECTS ---
for name in objects:
    if name not in placed_objects:
        pos = get_empty_pos()
        if pos:
            name_to_pos[name] = pos
            object_matrix[pos[0], pos[1]] = 1
            placed_objects.add(name)
            print(f"Randomly placed '{name}' at {pos}")
        else:
            print(f"⚠️ Cannot place '{name}'")

# --- SAVE LAYER ---
output_json = {
    title: {
        "matrix_base": terrain_matrix.tolist(),
        "matrix_environment": object_matrix.tolist()
    }
}
with open(os.path.join(OUTPUT_DIR, f"{STORY_ID}_scene_object_affordance_layers.json"), "w") as f:
    json.dump(output_json, f, indent=2)

# --- SAVE OBJECT NAME-POSITION MAP ---
object_info = {
    name: {"position": [y, x]}
    for name, (y, x) in name_to_pos.items()
}
with open(os.path.join(OUTPUT_DIR, f"{STORY_ID}_objects_with_positions.json"), "w") as f:
    json.dump(object_info, f, indent=2)

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

# Draw objects and labels
for name, (y, x) in name_to_pos.items():
    ax.scatter(x + 0.5, y + 0.5, s=300, color="orange", edgecolors='black', zorder=3)
    ax.text(x + 0.6, y + 0.4, name, fontsize=9, fontweight='bold',
            color='black', ha='left', va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.8))

ax.set_xlim(0, MAP_WIDTH)
ax.set_ylim(MAP_HEIGHT, 0)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{STORY_ID}_layout_fixed.png"))
plt.show()
