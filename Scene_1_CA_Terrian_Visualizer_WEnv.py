import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from collections import defaultdict
import json
from pathlib import Path
import random

SAVE_OUT_FOLDER = "StoryFiles/"
FILE_NUMBER = 0 #"StoryFiles/"+FILE_NUMBER+"

# ---------------- Config ----------------
MAP_WIDTH, MAP_HEIGHT = 30, 20
TILE_EMPTY, TILE_BASE, TILE_PATCH, TILE_OBJECT = 0, 1, 2, 3
BASE_PROB, PATCH_PROB = 0.65, 0.5
BASE_ITER, PATCH_ITER = 4, 3
PATCH_MIN_SIZE = 20
USE_FIXED_SEED = True
BASE_SEED, PATCH_SEED = 42, 1234

# ---------------- Toggle for Story ----------------
AFFORDANCE_PATH = SAVE_OUT_FOLDER + str(FILE_NUMBER)+"_object_affordance_langchain.json"
DECISION_PATH = SAVE_OUT_FOLDER + str(FILE_NUMBER)+"_scene_generation_decisions.json"

# ---------------- Output ----------------
MATRIX_LOG_PATH = SAVE_OUT_FOLDER + str(FILE_NUMBER)+"_tile_matrix_with_objects.json"
PLACEMENT_JSON_PATH = SAVE_OUT_FOLDER + str(FILE_NUMBER)+"_object_placement_log.json"

# ---------------- Utility Functions ----------------
def initialize_map(prob, tile_val, seed):
    np.random.seed(seed)
    return np.where(np.random.rand(MAP_HEIGHT, MAP_WIDTH) < prob, tile_val, TILE_EMPTY)

def smooth_map(grid, tile_val, iterations):
    for _ in range(iterations):
        new = grid.copy()
        for y in range(MAP_HEIGHT):
            for x in range(MAP_WIDTH):
                neighbors = grid[max(0,y-1):min(MAP_HEIGHT,y+2), max(0,x-1):min(MAP_WIDTH,x+2)]
                count = np.count_nonzero(neighbors == tile_val)
                new[y,x] = tile_val if count >= 5 else TILE_EMPTY
        grid = new
    return grid

def connect_largest_region(grid, tile_val):
    labeled, num = label(grid == tile_val)
    if num == 0: return np.zeros_like(grid)
    largest = np.argmax(np.bincount(labeled.flat)[1:]) + 1
    return (labeled == largest).astype(int) * tile_val

def to_matrix_list(grid):
    return [[int(cell) for cell in row] for row in grid]

# ---------------- Load Data ----------------
with open(AFFORDANCE_PATH, "r", encoding="utf-8") as f:
    affordance_data = json.load(f)
with open(DECISION_PATH, "r", encoding="utf-8") as f:
    decision_data = json.load(f)

scene_map = {s["scene_title"]: s for s in decision_data}
scene_titles = [s["scene_title"] for s in decision_data]

# ---------------- Detect and Propagate Environmental Objects ----------------
base_to_env_objs = defaultdict(set)

for scene in affordance_data["per_scene_affordances"]:
    scene_title = scene["scene_title"]
    base = scene_map[scene_title]["chosen_base"]
    for obj in scene["objects"]:
        if obj.get("affordance_category", "").lower() == "environmental object":
            name = obj["object"].lower().replace(" ", "_")
            base_to_env_objs[base].add(name)

# ---------------- Generate Base Maps ----------------
base_maps = {}
matrix_log = {"base_maps": {}, "patch_maps": {}, "scene_maps": {}}

for i, base in enumerate(sorted({s["chosen_base"] for s in decision_data})):
    seed = BASE_SEED + i if USE_FIXED_SEED else np.random.randint(0, 9999)
    mat = initialize_map(BASE_PROB, TILE_BASE, seed)
    mat = smooth_map(mat, TILE_BASE, BASE_ITER)
    mat = connect_largest_region(mat, TILE_BASE)
    base_maps[base] = mat
    matrix_log["base_maps"][base] = to_matrix_list(mat)

# ---------------- Generate Patch Maps ----------------
patch_names = sorted({p for s in decision_data for p in s["final_patch_for_base"] if p != "<no patch>"})
patch_maps = {}
used_mask = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=bool)

for i, patch in enumerate(patch_names):
    seed = PATCH_SEED + i if USE_FIXED_SEED else np.random.randint(0, 9999)
    raw = initialize_map(PATCH_PROB, TILE_PATCH, seed)
    raw = smooth_map(raw, TILE_PATCH, PATCH_ITER)
    patch_mat = connect_largest_region(raw, TILE_PATCH)
    patch_mat = np.where(used_mask | (patch_mat == 0), TILE_EMPTY, patch_mat)
    if np.sum(patch_mat > 0) >= PATCH_MIN_SIZE:
        patch_maps[patch] = patch_mat
        used_mask |= (patch_mat > 0)
        matrix_log["patch_maps"][patch] = to_matrix_list(patch_mat)

# ---------------- Place Environmental Objects per Base ----------------
object_placements = defaultdict(dict)

for base, obj_names in base_to_env_objs.items():
    base_mask = base_maps[base]
    walkable = np.argwhere(base_mask == TILE_BASE)
    rng = random.Random(BASE_SEED + hash(base) % 10000)
    placed = set()

    for obj in sorted(obj_names):
        success = False
        for _ in range(100):
            y, x = walkable[rng.randint(0, len(walkable)-1)]
            if (x, y) not in placed:
                object_placements[base][obj] = {"x": int(x), "y": int(y)}
                placed.add((x, y))
                success = True
                break
        if not success:
            print(f"⚠️ Failed to place object '{obj}' in base '{base}'.")

# ---------------- Visualize and Record ----------------
fig, axs = plt.subplots(len(decision_data), 2, figsize=(12, 4 * len(decision_data)))

for idx, scene in enumerate(decision_data):
    title = scene["scene_title"]
    base_name = scene["chosen_base"]
    base = base_maps[base_name]
    patch_layer = np.zeros_like(base)
    obj_layer = np.zeros_like(base)



    # Build patch layer
    for patch in scene["final_patch_for_base"]:
        if patch in patch_maps:
            patch_layer = np.where(patch_maps[patch] > 0, TILE_PATCH, patch_layer)

    # Place objects from pre-computed coordinates
    for obj, coord in object_placements[base_name].items():
        obj_layer[coord["y"], coord["x"]] = TILE_OBJECT

    # Merge layers
    combined = np.where(obj_layer > 0, TILE_OBJECT,
                np.where(patch_layer > 0, TILE_PATCH, base))

    # Log matrices
    matrix_log["scene_maps"][title] = {
        "base": to_matrix_list(base),
        "patch": to_matrix_list(patch_layer),
        "objects": to_matrix_list(obj_layer),
        "combined": to_matrix_list(combined)
    }

    # Visualization (Scene Summary + Combined Map with Object Labels)
    desc = f"Scene: {title}\nBase: {base_name}\nObjects:\n" + "\n".join(
        f"• {obj} @ {coord}" for obj, coord in object_placements[base_name].items()
    )

    axs[idx, 0].text(0.01, 0.5, desc, fontsize=10)
    axs[idx, 0].axis("off")

    # Draw combined map
    axs[idx, 1].imshow(combined, cmap="tab20", vmin=0, vmax=4)
    axs[idx, 1].set_title(f"{title} (combined)")
    axs[idx, 1].axis("off")

    # Overlay object positions
    for obj, coord in object_placements[base_name].items():
        x, y = coord["x"], coord["y"]
        axs[idx, 1].scatter([x], [y], color='red', s=40, marker='o')
        axs[idx, 1].text(x + 0.5, y, obj.replace("_", " "), fontsize=7, color='black', backgroundcolor='white')


plt.tight_layout()
plt.show()

# ---------------- Save Outputs ----------------
with open(MATRIX_LOG_PATH, "w", encoding="utf-8") as f:
    json.dump(matrix_log, f, indent=2)

with open(PLACEMENT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(object_placements, f, indent=2)

print(f"✅ Matrix saved to: {MATRIX_LOG_PATH}")
print(f"✅ Object placements saved to: {PLACEMENT_JSON_PATH}")
