import re
import json

# --- Input: Structured spatial lines (example) ---
scene_description_lines = [
    "House below Tree",
    "Tree to the right of barrel",
    "Flower above Tree",
    "Tree to the left of Tree stump"
]

# --- Supported spatial relation patterns ---
relation_patterns = [
    (r"(.+?) to the right of (.+)", "at the right of"),
    (r"(.+?) to the left of (.+)", "at the left of"),
    (r"(.+?) above (.+)", "above"),
    (r"(.+?) below (.+)", "below"),
    (r"(.+?) sits atop (.+)", "on top of"),
]

# --- Parsed output ---
objects = set()
relations = []

for line in scene_description_lines:
    line = line.strip()
    matched = False
    for pattern, relation in relation_patterns:
        match = re.match(pattern, line, flags=re.IGNORECASE)
        if match:
            source = match.group(1).strip()
            target = match.group(2).strip()
            relations.append({
                "source": source,
                "target": target,
                "relation": relation
            })
            objects.add(source)
            objects.add(target)
            matched = True
            break
    if not matched:
        if line:  # fallback for singleton (no spatial relation)
            objects.add(line.strip())

# --- Construct JSON output ---
scene_json = {
    "scene_title": "Forest Scene",
    "characters": [],
    "items": [],
    "interactive_objects": [],
    "environment_objects": sorted(objects),
    "spatial_relations": relations
}

# --- Save as JSON ---
with open("single_scene_forest.json", "w", encoding="utf-8") as f:
    json.dump(scene_json, f, indent=2)

print("✅ Parsed scene saved as 'single_scene_forest.json'")
