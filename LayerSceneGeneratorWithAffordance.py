import json

STORY_ID = 3

# --- Load Inputs ---
with open("StoryFiles/"+str(STORY_ID)+"_adventure_scene_output_FIXED.json", "r", encoding="utf-8") as f:
    narrative_data = json.load(f)

with open("StoryFiles/"+str(STORY_ID)+"_object_affordance_langchain.json", "r", encoding="utf-8") as f:
    affordance_data = json.load(f)

global_affordances = affordance_data["global_object_affordances"]
scene_data = narrative_data["time_frames"]

# --- Layer Mapping Template ---
layer_labels = {
    0: "terrain",
    1: "environmental_objects",
    2: "interactive_objects",
    3: "items",
    4: "characters",
    -1: "fx"
}

# --- Build Scene Layer Structure ---
scenes = []

for scene in scene_data:
    scene_title = scene["title"]
    layout = {label: [] for label in layer_labels.values()}

    seen_objects = set()
    for rel in scene["scene_relations"]:
        if "[" not in rel or "]" not in rel:
            continue
        o1 = rel.split("[")[0].strip()
        o2 = rel.split("]")[-1].strip()
        for obj in [o1, o2]:
            if obj not in seen_objects:
                seen_objects.add(obj)
                aff = global_affordances.get(obj)
                if aff:
                    layer_key = layer_labels.get(aff["affordance_level"], "unassigned")
                    layout[layer_key].append({
                        "object": obj,
                        "category": aff["category"],
                        "suggested_terrain": aff["suggested_terrain"],
                        "confidence": aff["confidence"]
                    })

    scenes.append({
        "scene_title": scene_title,
        "layered_objects": layout
    })

# --- Save Output ---
with open("scene_layout_from_affordance.json", "w", encoding="utf-8") as f:
    json.dump(scenes, f, indent=2, ensure_ascii=False)

print("✅ Saved layered scene layout to scene_layout_from_affordance.json")
