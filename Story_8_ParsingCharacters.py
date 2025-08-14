import json
from collections import defaultdict

SAVE_OUT_FOLDER = "StoryFiles/"  # Change if needed
FILE_NUMBER = 0  # Change to 2 for Story 2

# Load files
with open(f"{SAVE_OUT_FOLDER}{FILE_NUMBER}_object_affordance_langchain.json", 'r') as f:
    object_data = json.load(f)

# Extract characters from each scene independently (no propagation)
final_output = []
for scene in object_data["per_scene_affordances"]:
    scene_title = scene["scene_title"]
    characters = [
        obj["object"].strip()
        for obj in scene.get("objects", [])
        if obj.get("category") == "Character"
    ]
    final_output.append({
        "scene_title": scene_title,
        "characters": sorted(characters)
    })

# Save result
output_path = f"{SAVE_OUT_FOLDER}{FILE_NUMBER}_character_objects_expanded.json"
with open(output_path, "w") as f:
    json.dump(final_output, f, indent=2)

print(f"✅ Character data saved to {output_path}")
