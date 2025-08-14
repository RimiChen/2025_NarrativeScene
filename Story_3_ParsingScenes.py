import json
from collections import Counter, defaultdict
from pathlib import Path

SAVE_OUT_FOLDER = "StoryFiles/"
FILE_NUMBER = 0 #"StoryFiles/"+FILE_NUMBER+"

# ------------ Configuration ------------
JSON_PATH = SAVE_OUT_FOLDER + str(FILE_NUMBER)+"_object_affordance_langchain.json"  # Or "2_object_..."
TERRAIN_KEYWORDS = ["room", "alley", "hall", "corridor", "path", "cave", "lab", "arena"]
MATERIAL_LIKE = {
    "grass", "rock", "mud", "wood", "stone", "dirt", "concrete",
    "indoor", "outdoor", "sand", "soil", "metal", "urban", "office"
}

# ------------ Load Data ------------
with open(JSON_PATH, "r", encoding="utf-8") as f:
    story_data = json.load(f)

scenes = story_data["per_scene_affordances"]

# ------------ Step 1: Collect scene info ------------
scene_decisions = []
base_to_scenes = defaultdict(list)
base_to_patches = defaultdict(set)

for scene in scenes:
    title = scene["scene_title"]

    # Step 1a: Choose Base
    raw_terrain = [
        obj["suggested_terrain"].lower()
        for obj in scene["objects"]
        if obj.get("suggested_terrain", "").lower() not in {"any", "n/a"}
    ]
    # Split multi-value entries like "indoor, concrete"
    flat_terrains = []
    for t in raw_terrain:
        flat_terrains += [x.strip() for x in t.split(",") if x.strip()]
    
    freq = Counter(flat_terrains)
    chosen_base = next((t for t, _ in freq.most_common() if t in MATERIAL_LIKE), None)
    if not chosen_base:
        chosen_base = freq.most_common(1)[0][0] if freq else "grass"

    base_to_scenes[chosen_base].append(title)

    # Step 1b: Detect patch terrain-like objects
    detected_patches = []
    for obj in scene["objects"]:
        label = obj.get("object", "").lower()
        if any(kw in label for kw in TERRAIN_KEYWORDS):
            detected_patches.append(label.replace(" ", "_"))

    for p in detected_patches:
        base_to_patches[chosen_base].add(p)

    scene_decisions.append({
        "scene_title": title,
        "suggested_terrains": list(set(flat_terrains)),
        "chosen_base": chosen_base,
        "terrain_objects_detected": detected_patches
    })

# ------------ Step 2: Propagate patch decisions per base ------------
for s in scene_decisions:
    base = s["chosen_base"]
    s["final_patch_for_base"] = sorted(base_to_patches[base]) if base_to_patches[base] else ["<no patch>"]

# ------------ Save to JSON ------------
output_path = Path(SAVE_OUT_FOLDER + str(FILE_NUMBER)+"_scene_generation_decisions.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(scene_decisions, f, indent=2)

print(f"✅ Final corrected decision summary written to: {output_path.resolve()}")
