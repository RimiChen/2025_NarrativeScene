import json
from collections import defaultdict
import os

STORY_ID = "0"  # Change this to your story ID

# === Config ===
INPUT_PATH = f"StoryFiles/output_KG_story_{STORY_ID}/{STORY_ID}_kg_data.json"
OUTPUT_PATH = f"StoryFiles/{STORY_ID}_scene_summaries.json"

# === Load KG Data ===
print(">>> Loading:", INPUT_PATH)
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    kg_data = json.load(f)

print(">>> File loaded. Top-level keys:", list(kg_data.keys()))
scene_kgs = kg_data.get("scene_kgs", {})
merged_edges = kg_data.get("merged_kg", {}).get("edges", [])

# === Build helper for scene relations (timeline) ===
scene_next = defaultdict(list)
for edge in merged_edges:
    if edge["relation"] == "precedes":
        scene_next[edge["source"]].append(edge["target"])

# === Summarize each scene ===
scene_summaries = []
for scene_title, kg in scene_kgs.items():
    print(f"Processing scene: {scene_title}")
    summary = {
        "scene_title": scene_title,
        "base": None,
        "patch": [],
        "characters": [],
        "items": [],
        "interactive_objects": [],
        "environment_objects": [],
        "spatial_relations": [],
        "next_scenes": scene_next.get(scene_title, [])
    }

    for edge in kg.get("edges", []):
        src, tgt, rel = edge["source"], edge["target"], edge["relation"]
        if src == scene_title:
            if rel == "has_base":
                summary["base"] = tgt
            elif rel == "has_patch":
                summary["patch"].append(tgt)
            elif rel == "has_character":
                summary["characters"].append(tgt)
            elif rel == "has_item":
                summary["items"].append(tgt)
            elif rel == "has_interactive_object":
                summary["interactive_objects"].append(tgt)
            elif rel == "has_environment_object":
                summary["environment_objects"].append(tgt)
        else:
            summary["spatial_relations"].append({
                "source": src,
                "target": tgt,
                "relation": rel
            })

    scene_summaries.append(summary)

# === Ensure folder and write output ===
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(scene_summaries, f, indent=2)

print(f"✅ Scene summaries saved to: {OUTPUT_PATH}")
