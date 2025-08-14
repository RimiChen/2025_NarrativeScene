import json
from collections import defaultdict
from pathlib import Path

SAVE_OUT_FOLDER = "StoryFiles/"
FILE_NUMBER = 0 #"StoryFiles/"+FILE_NUMBER+"

# Input files
AFFORDANCE_PATH = SAVE_OUT_FOLDER + str(FILE_NUMBER)+"_object_affordance_langchain.json"
DECISION_PATH = SAVE_OUT_FOLDER + str(FILE_NUMBER)+"_scene_generation_decisions.json"
OUTPUT_PATH = SAVE_OUT_FOLDER + str(FILE_NUMBER)+"_scene_object_propagation.json"

# Load data
with open(AFFORDANCE_PATH, "r", encoding="utf-8") as f:
    affordance_data = json.load(f)
with open(DECISION_PATH, "r", encoding="utf-8") as f:
    decision_data = json.load(f)

# Step 1: Build scene → base mapping
scene_to_base = {s["scene_title"]: s["chosen_base"] for s in decision_data}

# Step 2: Collect environmental objects (correctly)
base_to_objects = defaultdict(set)
scene_env_objects = defaultdict(list)

for scene in affordance_data["per_scene_affordances"]:
    title = scene["scene_title"]
    base = scene_to_base.get(title)
    if not base:
        continue

    for obj in scene["objects"]:
        if (obj.get("category", "").lower() == "environmental object"
                and obj.get("affordance_level") == 1):
            name = obj["object"].lower().replace(" ", "_")
            base_to_objects[base].add(name)
            scene_env_objects[title].append(name)

# Step 3: Propagate to all scenes sharing the same base
scene_to_propagated_objects = {}

for title, base in scene_to_base.items():
    propagated = sorted(base_to_objects.get(base, []))
    scene_to_propagated_objects[title] = {
        "base": base,
        "propagated_objects": propagated
    }

# Step 4: Save to JSON
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(scene_to_propagated_objects, f, indent=2)

print(f"✅ Scene-to-object propagation saved to {OUTPUT_PATH}")
