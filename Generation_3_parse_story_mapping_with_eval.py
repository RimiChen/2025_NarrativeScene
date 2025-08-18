import warnings
import json
import os
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from statistics import mean

# === CONFIGURABLE ===
STORY_ID = "1"
STORY_FILE = f"StoryFiles/output_KG_story_{STORY_ID}/{STORY_ID}_kg_data.json"
EMBED_INDEX = "Data/object_embedding_index.jsonl"

# outputs
OUTPUT_TOP1 = f"StoryFiles/{STORY_ID}_matched_objects.json"  # backward-compatible (object -> [top1_path])
OUTPUT_DETAILED = f"StoryFiles/{STORY_ID}_matched_objects_detailed.json"  # new: full scores for top-k
OUTPUT_METRICS = f"StoryFiles/{STORY_ID}_matching_metrics.json"           # new: summary metrics

# matching config
TOP_K = 3
HIGH_CONF_THRESH = 0.70  # threshold for high-confidence@1, tune as needed
WEIGHTS = {"name": 0.4, "group": 0.3, "super": 0.2, "afford": 0.1}

# === Load model and index ===
# Note: reuse your model; same as original script
model = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_sim(a, b):
    return 1 - cosine(a, b)

# Load prebuilt embedding index (expects fields: image_path, embedding.detailed_name, group, supercategory, affordance, affordance list)
with open(EMBED_INDEX, "r", encoding="utf-8") as f:
    index = [json.loads(line) for line in f]

# === Embedding helper ===
def get_embedding(text):
    return model.encode(text, show_progress_bar=False)

# === Character Heuristic (unchanged) ===
def is_probable_character(name):
    return name and name[0].isupper() and "_" not in name and len(name.split()) <= 2

# === Matching: return full rows with component scores ===
def query_object(text, top_k=TOP_K, restrict_to_character=False):
    query_vec = get_embedding(text)
    rows = []
    for obj in index:
        if restrict_to_character and "Characters" not in obj.get("affordance", []):
            continue

        emb = obj.get("embedding", {})
        # Defensive: some fields may be None
        name_vec = emb.get("detailed_name")
        group_vec = emb.get("group")
        super_vec = emb.get("supercategory")
        afford_vec = emb.get("affordance")

        sim_name = cosine_sim(query_vec, name_vec) if name_vec is not None else 0.0
        sim_group = cosine_sim(query_vec, group_vec) if group_vec is not None else 0.0
        sim_super = cosine_sim(query_vec, super_vec) if super_vec is not None else 0.0
        sim_afford = cosine_sim(query_vec, afford_vec) if afford_vec is not None else 0.0

        total = (
            WEIGHTS["name"] * sim_name +
            WEIGHTS["group"] * sim_group +
            WEIGHTS["super"] * sim_super +
            WEIGHTS["afford"] * sim_afford
        )

        rows.append({
            "image_path": obj["image_path"],
            "total_score": float(total),
            "sim_name": float(sim_name),
            "sim_group": float(sim_group),
            "sim_super": float(sim_super),
            "sim_afford": float(sim_afford),
            "weights": WEIGHTS,
            "candidate_affordances": obj.get("affordance", []),
        })

    # sort by total score desc
    rows.sort(key=lambda r: r["total_score"], reverse=True)
    return rows[:top_k]

# === Main Processing ===
with open(STORY_FILE, "r", encoding="utf-8") as f:
    story = json.load(f)

scene_kgs = story["scene_kgs"]
matched_top1 = {}       # backward compatible: object -> [top1_path]
matched_detailed = {}   # new: object -> {"restricted": bool, "candidates": [rows]}

objects_seen = set()

for scene, data in scene_kgs.items():
    for obj_name in data["nodes"]:
        if obj_name in objects_seen:
            continue
        objects_seen.add(obj_name)

        restrict = is_probable_character(obj_name)
        candidates = query_object(obj_name, top_k=TOP_K, restrict_to_character=restrict)

        matched_top1[obj_name] = [candidates[0]["image_path"]] if candidates else []
        matched_detailed[obj_name] = {
            "restricted_to_characters": bool(restrict),
            "query_text": obj_name,
            "candidates": candidates
        }

# === Save Outputs ===
os.makedirs("StoryFiles", exist_ok=True)

with open(OUTPUT_TOP1, "w", encoding="utf-8") as fout:
    json.dump(matched_top1, fout, indent=2)
print(f"[✓] Saved object->top1 image mapping to {OUTPUT_TOP1}")

with open(OUTPUT_DETAILED, "w", encoding="utf-8") as fout:
    json.dump(matched_detailed, fout, indent=2)
print(f"[✓] Saved detailed matching (top-{TOP_K}) with scores to {OUTPUT_DETAILED}")

# === Compute simple evaluation metrics from the matching run ===
def safe_mean(values):
    return float(mean(values)) if values else 0.0

top1_totals = []
top1_name = []
top1_group = []
top1_super = []
top1_afford = []
margins_12 = []  # top1 - top2
margins_13 = []  # top1 - top3
high_conf_count = 0
diversity_set = set()
char_filter_hits = 0
char_filter_total = 0

for obj, pack in matched_detailed.items():
    cands = pack["candidates"]
    if not cands:
        continue

    c1 = cands[0]
    top1_totals.append(c1["total_score"])
    top1_name.append(c1["sim_name"])
    top1_group.append(c1["sim_group"])
    top1_super.append(c1["sim_super"])
    top1_afford.append(c1["sim_afford"])
    diversity_set.add(c1["image_path"])

    if len(cands) >= 2:
        c2 = cands[1]
        margins_12.append(c1["total_score"] - c2["total_score"])
    if len(cands) >= 3:
        c3 = cands[2]
        margins_13.append(c1["total_score"] - c3["total_score"])

    if c1["total_score"] >= HIGH_CONF_THRESH:
        high_conf_count += 1

    if pack["restricted_to_characters"]:
        char_filter_total += 1
        # sanity: if we restricted to characters, top1 should be a character
        # we cannot verify ground truth, but we can check if candidate declares "Characters" affordance
        if "Characters" in c1.get("candidate_affordances", []):
            char_filter_hits += 1

num_objects = len(matched_detailed)
metrics = {
    "story_id": STORY_ID,
    "num_objects": num_objects,
    "top_k": TOP_K,
    "high_conf_threshold": HIGH_CONF_THRESH,

    "mean_top1_total": safe_mean(top1_totals),
    "mean_top1_sim_name": safe_mean(top1_name),
    "mean_top1_sim_group": safe_mean(top1_group),
    "mean_top1_sim_super": safe_mean(top1_super),
    "mean_top1_sim_afford": safe_mean(top1_afford),

    "mean_margin_top1_top2": safe_mean(margins_12),
    "mean_margin_top1_top3": safe_mean(margins_13),

    "pct_high_conf_at_1": (high_conf_count / num_objects) if num_objects else 0.0,
    "result_diversity": (len(diversity_set) / num_objects) if num_objects else 0.0,

    # character filter consistency: how often top1 is labeled as a Character when we enforced the filter
    "character_filter_total": char_filter_total,
    "character_filter_hits": char_filter_hits,
    "character_filter_hit_rate": (char_filter_hits / char_filter_total) if char_filter_total else 0.0,
}

with open(OUTPUT_METRICS, "w", encoding="utf-8") as fout:
    json.dump(metrics, fout, indent=2)
print(f"[✓] Saved matching metrics to {OUTPUT_METRICS}")
