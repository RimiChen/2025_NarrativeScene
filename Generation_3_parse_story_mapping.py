import json
import os
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# === CONFIGURABLE ===
STORY_ID = "0"
STORY_FILE = f"StoryFiles/output_KG_story_{STORY_ID}/{STORY_ID}_kg_data.json"
EMBED_INDEX = "Data/GameTile/object_embedding_index.jsonl"
OUTPUT_FILE = f"StoryFiles/{STORY_ID}_matched_objects.json"

# === Load model and index ===
model = SentenceTransformer('all-MiniLM-L6-v2')
with open(EMBED_INDEX, "r", encoding="utf-8") as f:
    index = [json.loads(line) for line in f]

def get_embedding(text): return model.encode(text, show_progress_bar=False)
def cosine_sim(a, b): return 1 - cosine(a, b)

# === Character Heuristic ===
def is_probable_character(name):
    return name[0].isupper() and "_" not in name and len(name.split()) <= 2

# === Matching Function with optional affordance filter ===
def query_object(text, top_k=1, restrict_to_character=False):
    query_vec = get_embedding(text)
    candidates = []
    for obj in index:
        if restrict_to_character and "Characters" not in obj.get("affordance", []):
            continue
        sim_name = cosine_sim(query_vec, obj["embedding"]["detailed_name"])
        sim_group = cosine_sim(query_vec, obj["embedding"]["group"]) if obj["embedding"]["group"] else 0
        sim_super = cosine_sim(query_vec, obj["embedding"]["supercategory"]) if obj["embedding"]["supercategory"] else 0
        sim_afford = cosine_sim(query_vec, obj["embedding"]["affordance"]) if obj["embedding"]["affordance"] else 0
        total = 0.4 * sim_name + 0.3 * sim_group + 0.2 * sim_super + 0.1 * sim_afford
        candidates.append((total, obj["image_path"]))
    candidates.sort(reverse=True)
    return [c[1] for c in candidates[:top_k]]

# === Main Processing ===
with open(STORY_FILE, "r", encoding="utf-8") as f:
    story = json.load(f)

scene_kgs = story["scene_kgs"]
matched = {}

for scene, data in scene_kgs.items():
    for obj in data["nodes"]:
        if obj in matched:
            continue
        restrict = is_probable_character(obj)
        top_match = query_object(obj, top_k=1, restrict_to_character=restrict)
        matched[obj] = top_match

# === Save Output ===
with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    json.dump(matched, fout, indent=2)

print(f"[✓] Saved object-image mapping to {OUTPUT_FILE}")
