import json
from collections import defaultdict

SAVE_OUT_FOLDER = "StoryFiles/"
FILE_NUMBER = 0  # Update this to match the story number

# Load files
with open(f"{SAVE_OUT_FOLDER}{FILE_NUMBER}_object_affordance_langchain.json", 'r') as f:
    object_data = json.load(f)

with open(f"{SAVE_OUT_FOLDER}{FILE_NUMBER}_scene_generation_decisions.json", 'r') as f:
    scene_decisions = json.load(f)

# Step 1: Map scene title to base+patch
scene_info = {}
for entry in scene_decisions:
    scene_title = entry["scene_title"]
    base = entry.get("chosen_base", "default")
    patch = tuple(entry.get("final_patch_for_base", []))
    scene_info[scene_title] = {"base": base, "patch": patch}

# Step 2: Extract item objects from each scene
scene_items = defaultdict(list)
for scene in object_data["per_scene_affordances"]:
    title = scene["scene_title"]
    for obj in scene.get("objects", []):
        if obj.get("category") == "Item / Collectible":
            label = obj.get("object")
            if label and label not in scene_items[title]:
                scene_items[title].append(label)

# Step 3: Group scenes by (base, patch)
grouped_scenes = defaultdict(list)
for scene_title, info in scene_info.items():
    grouped_scenes[(info["base"], info["patch"])].append(scene_title)

# Step 4: Propagate items across same (base, patch) scenes
final_output = []
for (base, patch), scenes in grouped_scenes.items():
    combined_items = set()
    for title in scenes:
        combined_items.update(scene_items.get(title, []))
    for title in scenes:
        final_output.append({
            "scene_title": title,
            "base": base,
            "patch": list(patch),
            "items": sorted(combined_items)
        })

# Save result
output_path = f"{SAVE_OUT_FOLDER}{FILE_NUMBER}_item_collectibles_expanded.json"
with open(output_path, "w") as f:
    json.dump(final_output, f, indent=2)

print(f"✅ Saved to {output_path}")
