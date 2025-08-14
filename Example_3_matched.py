import json, os

# --- CONFIG ---
PLACEMENT_FILE = "StoryFiles/single_scene_forest_fixed/forest_scene_object_affordance_layers.json"
MATCHED_OBJECTS_FILE = "StoryFiles/0_matched_objects.json"
OUTPUT_JSON = "StoryFiles/single_scene_forest_fixed/forest_scene_objects_with_images.json"
SCENE_KEY = "Forest Scene"

# --- LOAD ---
with open(PLACEMENT_FILE, "r") as f:
    placed = json.load(f)[SCENE_KEY]

with open(MATCHED_OBJECTS_FILE, "r") as f:
    matched = json.load(f)

env_matrix = placed["matrix_environment"]

# --- REVERSE MAPPING (for exact matching) ---
lowercase_match_map = {
    k.lower(): v[0] if isinstance(v, list) else v
    for k, v in matched.items()
}

# --- SCAN OBJECT LOCATIONS ---
objects_with_images = []
for y, row in enumerate(env_matrix):
    for x, val in enumerate(row):
        if val == 1:
            # Try to guess object name based on closest match
            found = False
            for obj_name in lowercase_match_map:
                if obj_name in SCENE_KEY.lower():  # just an example heuristic
                    objects_with_images.append({
                        "name": obj_name,
                        "position": [y, x],
                        "image": lowercase_match_map[obj_name]
                    })
                    found = True
                    break
            if not found:
                objects_with_images.append({
                    "name": "unknown",
                    "position": [y, x],
                    "image": "PLACEHOLDER_IMAGE_PATH.png"
                })

# --- OUTPUT ---
with open(OUTPUT_JSON, "w") as f:
    json.dump(objects_with_images, f, indent=2)

print(f"✅ Saved object list with images to: {OUTPUT_JSON}")
