import json
from collections import defaultdict
from difflib import get_close_matches

SAVE_OUT_FOLDER = "StoryFiles/"
FILE_NUMBER = 0  # Change to 2 for another story
SIMILARITY_THRESHOLD = 0.85  # Between 0 and 1

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

# Step 2: Extract raw object names per scene
scene_interactive_raw = defaultdict(list)
for scene in object_data["per_scene_affordances"]:
    title = scene["scene_title"]
    for obj in scene.get("objects", []):
        if obj.get("category") == "Interactive Object":
            label = obj.get("object", "").strip()
            if label:
                scene_interactive_raw[title].append(label)

# Step 3: Normalize names using fuzzy matching
def merge_similar_objects(object_lists, threshold=SIMILARITY_THRESHOLD):
    flat_list = sorted(set(obj for lst in object_lists.values() for obj in lst))
    clusters = []
    for obj in flat_list:
        matched = False
        for cluster in clusters:
            if get_close_matches(obj, cluster, n=1, cutoff=threshold):
                cluster.append(obj)
                matched = True
                break
        if not matched:
            clusters.append([obj])

    # Choose canonical names (shortest representative)
    canonical_map = {}
    for cluster in clusters:
        canonical = min(cluster, key=len)
        for item in cluster:
            canonical_map[item] = canonical
    return canonical_map

canonical_map = merge_similar_objects(scene_interactive_raw)

# Step 4: Apply normalization
scene_interactive_normalized = defaultdict(list)
for scene, objects in scene_interactive_raw.items():
    normed = sorted(set(canonical_map[obj] for obj in objects))
    scene_interactive_normalized[scene] = normed

# Step 5: Propagate across same base + patch
grouped_scenes = defaultdict(list)
for scene_title, info in scene_info.items():
    grouped_scenes[(info["base"], info["patch"])].append(scene_title)

final_output = []
for (base, patch), scenes in grouped_scenes.items():
    all_objs = set()
    for s in scenes:
        all_objs.update(scene_interactive_normalized.get(s, []))
    for s in scenes:
        final_output.append({
            "scene_title": s,
            "base": base,
            "patch": list(patch),
            "interactive_objects": sorted(all_objs)
        })

# Save result
output_path = f"{SAVE_OUT_FOLDER}{FILE_NUMBER}_interactive_objects_expanded.json"
with open(output_path, "w") as f:
    json.dump(final_output, f, indent=2)

print(f"✅ Saved with similarity merging to {output_path}")
