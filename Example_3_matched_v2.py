import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --- CONFIGURATION ---
SCENE_FILE = "StoryFiles/single_scene_forest_fixed/forest_scene_object_affordance_layers.json"
POSITION_FILE = "StoryFiles/single_scene_forest_fixed/forest_objects_with_positions.json"
IMAGE_FILE = "StoryFiles/single_scene_forest_fixed/forest_scene_objects_with_images.json"
SCENE_KEY = "Forest Scene"
ASSET_ROOT = "Data/GameTile/Assets"
TILE_SIZE = 1
IMAGE_SCALE = 1.0  # can increase for better visibility

# --- LOAD JSON DATA ---
with open(SCENE_FILE, "r") as f:
    scene_data = json.load(f)
scene = scene_data[SCENE_KEY]
matrix_base = scene["matrix_base"]

with open(POSITION_FILE, "r") as f:
    position_data = json.load(f)

with open(IMAGE_FILE, "r") as f:
    image_data = json.load(f)

# --- COMBINE DATA ---
object_data = {}
for name, pinfo in position_data.items():
    pos = pinfo.get("position")
    image_list = image_data.get(name, [])
    if pos and image_list:
        object_data[name] = {
            "position": pos,
            "image_path": image_list[0]  # use first image
        }

# --- SCENE DIMENSIONS ---
height = len(matrix_base)
width = len(matrix_base[0])

# --- PLOT SETUP ---
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, width)
ax.set_ylim(height, 0)
ax.set_xticks([])
ax.set_yticks([])

# --- DRAW BASE TERRAIN ---
terrain_colors = {
    "grass": "#a8d08d",
    "forest": "#4b8b3b",
    "dirt": "#c2b280"
}
for y in range(height):
    for x in range(width):
        terrain = matrix_base[y][x]
        color = terrain_colors.get(terrain, "#cccccc")
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=color, zorder=1))

# --- DRAW OBJECT IMAGES + LABELS ---
for name, info in object_data.items():
    y, x = info["position"]
    scene_y = y  # already top-down aligned
    full_img_path = os.path.join(ASSET_ROOT, info["image_path"])

    if os.path.exists(full_img_path):
        try:
            img = mpimg.imread(full_img_path)
            h, w = img.shape[:2]
            ratio = IMAGE_SCALE / max(h, w)
            width_px = w * ratio
            height_px = h * ratio

            ax.imshow(
                img,
                extent=(x, x + width_px, scene_y + height_px, scene_y),
                zorder=5
            )
        except Exception as e:
            print(f"⚠️ Error loading {full_img_path}: {e}")
    else:
        print(f"⚠️ Image not found: {full_img_path}")

    # Draw label beside the image
    ax.text(
        x + 0.2, scene_y + 0.2, name,
        fontsize=9, fontweight='bold',
        color="black", ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.6)
    )

# --- FINALIZE ---
plt.title("Scene: Forest Scene")
plt.tight_layout()
plt.savefig("forest_scene_mapped_fixed.png", dpi=300)
plt.show()
