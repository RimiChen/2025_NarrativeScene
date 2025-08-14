import json
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
STORY_ID = 0
TILE_SIZE = 32
IMAGE_SCALE = 3.5  # <- 🖼️ Scale multiplier for all object images (e.g., 2.0 = 64px max)

ASSET_FOLDER = "Data/GameTile/Assets"  # <- Your asset image folder
OUTPUT_FOLDER = f"StoryFiles/{STORY_ID}_scene_image_renders_scaled"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# File paths
LAYER_FILE = f"StoryFiles/{STORY_ID}_scene_object_affordance_layers_RELOCATED.json"
SUMMARY_FILE = f"StoryFiles/{STORY_ID}_scene_summaries.json"
MATCHED_OBJECTS_FILE = f"StoryFiles/{STORY_ID}_matched_objects.json"

# --- LOAD FILES ---
with open(LAYER_FILE, "r", encoding="utf-8") as f:
    scene_layers = json.load(f)
with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
    scene_summaries = json.load(f)
with open(MATCHED_OBJECTS_FILE, "r", encoding="utf-8") as f:
    object_image_map = json.load(f)

# --- CACHING IMAGE FILES ---
image_cache = {}
def load_image_scaled(image_name, base_size=TILE_SIZE, scale=IMAGE_SCALE):
    key = f"{image_name}@{base_size}x{scale}"
    if key in image_cache:
        return image_cache[key]
    
    path = os.path.join(ASSET_FOLDER, image_name)
    if not os.path.exists(path):
        print(f"⚠️ Missing asset: {image_name}")
        return None
    
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    target_size = base_size * scale
    ratio = target_size / max(w, h)
    new_size = (int(w * ratio), int(h * ratio))
    img = img.resize(new_size, resample=Image.BICUBIC)
    image_cache[key] = img
    return img

# --- DRAW EACH SCENE ---
for scene in scene_summaries:
    title = scene["scene_title"]
    layers = scene_layers[title]
    base_matrix = np.array(layers["matrix_base"])
    H, W = base_matrix.shape
    canvas = Image.new("RGBA", (W * TILE_SIZE, H * TILE_SIZE), (255, 255, 255, 255))

    # Draw base terrain as gray blocks
    for y in range(H):
        for x in range(W):
            if base_matrix[y, x] > 0:
                tile = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (220, 220, 220, 255))
                canvas.paste(tile, (x * TILE_SIZE, y * TILE_SIZE))

    # Object layers
    type_to_layer = {
        "characters": "matrix_character",
        "items": "matrix_item",
        "interactive_objects": "matrix_interactive",
        "environment_objects": "matrix_environment"
    }

    for obj_type, matrix_key in type_to_layer.items():
        matrix = np.array(layers[matrix_key])
        object_names = scene.get(obj_type, [])
        positions = np.argwhere(matrix == 1)

        for (y, x), name in zip(positions, object_names):
            key_variants = [name, name.lower().replace(" ", "_")]
            image_path = None
            for k in key_variants:
                if k in object_image_map:
                    image_path = object_image_map[k][0]
                    break
            if not image_path:
                print(f"❌ No match for: {name}")
                continue

            img = load_image_scaled(image_path, TILE_SIZE, IMAGE_SCALE)
            if img:
                # Center the image on the tile
                offset_x = x * TILE_SIZE + (TILE_SIZE - img.size[0]) // 2
                offset_y = y * TILE_SIZE + (TILE_SIZE - img.size[1]) // 2
                canvas.paste(img, (offset_x, offset_y), img)

    # Save final output
    out_path = os.path.join(OUTPUT_FOLDER, f"{title.replace(' ', '_')}.png")
    canvas.save(out_path)
    print(f"✅ Saved: {out_path}")
