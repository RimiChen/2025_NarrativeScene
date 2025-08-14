import json
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
STORY_ID = 0
TILE_SIZE = 32
ASSET_FOLDER = "Data/GameTile/Assets"  # Replace with your actual asset path
OUTPUT_FOLDER = f"StoryFiles/{STORY_ID}_scene_image_renders"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# File paths
LAYER_FILE = f"StoryFiles/{STORY_ID}_scene_object_affordance_layers_RELOCATED.json"
SUMMARY_FILE = f"StoryFiles/{STORY_ID}_scene_summaries.json"
MATCHED_OBJECTS_FILE = f"StoryFiles/{STORY_ID}_matched_objects.json"  # Rename your matched file if needed

# --- LOAD FILES ---
with open(LAYER_FILE, "r", encoding="utf-8") as f:
    scene_layers = json.load(f)
with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
    scene_summaries = json.load(f)
with open(MATCHED_OBJECTS_FILE, "r", encoding="utf-8") as f:
    object_image_map = json.load(f)

# --- CACHING IMAGE FILES ---
image_cache = {}
def load_image(image_name):
    if image_name in image_cache:
        return image_cache[image_name]
    image_path = os.path.join(ASSET_FOLDER, image_name)
    if not os.path.exists(image_path):
        print(f"⚠️ Missing: {image_name}")
        return None
    img = Image.open(image_path).convert("RGBA").resize((TILE_SIZE, TILE_SIZE))
    image_cache[image_name] = img
    return img

# --- DRAW EACH SCENE ---
for scene in scene_summaries:
    title = scene["scene_title"]
    layers = scene_layers[title]
    base_matrix = np.array(layers["matrix_base"])
    H, W = base_matrix.shape
    canvas = Image.new("RGBA", (W * TILE_SIZE, H * TILE_SIZE), (255, 255, 255, 255))

    # Render base tiles as gray blocks
    for y in range(H):
        for x in range(W):
            if base_matrix[y, x] > 0:
                tile = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (200, 200, 200, 255))
                canvas.paste(tile, (x * TILE_SIZE, y * TILE_SIZE))

    # --- Overlay objects using matched images ---
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
            if image_path:
                img = load_image(image_path)
                if img:
                    canvas.paste(img, (x * TILE_SIZE, y * TILE_SIZE), img)
            else:
                print(f"❌ No match for: {name}")

    # Save output
    out_path = os.path.join(OUTPUT_FOLDER, f"{title.replace(' ', '_')}.png")
    canvas.save(out_path)
    print(f"✅ Saved: {out_path}")
