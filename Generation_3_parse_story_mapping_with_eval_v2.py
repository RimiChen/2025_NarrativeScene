import warnings
import json
import os
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from statistics import mean

# === CONFIGURABLE ===
STORY_ID = "3"
STORY_FILE = f"StoryFiles/output_KG_story_{STORY_ID}/{STORY_ID}_kg_data.json"
EMBED_INDEX = "Data/object_embedding_index.jsonl"

# Optional expected affordance file. Leave as None if you don’t have it yet.
# Format: {"Hero": ["Characters"], "chest": ["Interactive Object","Items and Collectibles"], ...}
EXPECTED_AFFORD_FILE = f"StoryFiles/{STORY_ID}_expected_affordances.json"  # or None

# outputs
OUTPUT_TOP1 = f"StoryFiles/{STORY_ID}_matched_objects.json"                 # backward-compatible (object -> [top1_path])
OUTPUT_DETAILED = f"StoryFiles/{STORY_ID}_matched_objects_detailed.json"    # full scores, flags, margins, affordance checks
OUTPUT_METRICS = f"StoryFiles/{STORY_ID}_matching_metrics.json"             # summary metrics
# affordance predictions produced upstream
AFFORD_PRED_FILE = f"StoryFiles/{STORY_ID}_object_affordance_langchain.json"  # adjust if per-story
USE_PER_SCENE_AFFORD = True      # True = prefer per_scene_affordances; False = fallback to global
MIN_PRED_CONF = 0.0              # keep all predictions; raise to e.g. 0.5 if you want to filter low confidence


# Map prediction categories to the index's affordance tags (plural form)
CATEGORY_TO_INDEX_TAG = {
    "Character": "Characters",
    "Characters": "Characters",
    "Interactive Object": "Interactive Object",
    "Item / Collectible": "Items and Collectibles",
    "Items and Collectibles": "Items and Collectibles",
    "Environmental Object": "Environmental Object",
    "Terrain": "Terrain",
    "Effect / Ambient / Unknown": "Effect / Ambient / Unknown",
}


# matching config
TOP_K = 3
HIGH_CONF_THRESH = 0.70   # “high-confidence@1” reporting threshold (not used for review flag)
CONF_THRESH_FOR_REVIEW = 0.50  # if top1_total < this, flag as low-confidence for review
REVIEW_MARGIN_THRESH = 0.05    # if (top1 - top2) < this, flag as ambiguous
WEIGHTS = {"name": 0.5, "group": 0.3, "super": 0.1, "afford": 0.1}  # suggested rebalance



# === Load model and index ===
model = SentenceTransformer('all-MiniLM-L6-v2')


def _norm(s: str) -> str:
    return (s or "").strip()

def cosine_sim(a, b):
    return 1 - cosine(a, b)

# Load prebuilt embedding index
with open(EMBED_INDEX, "r", encoding="utf-8") as f:
    index = [json.loads(line) for line in f]


# --- Load expected affordances from LLM prediction file ---
expected_afford = {}  # object_name -> list of expected tags (aligned to index)
if os.path.exists(AFFORD_PRED_FILE):
    with open(AFFORD_PRED_FILE, "r", encoding="utf-8") as f:
        pred = json.load(f)

    # Helper: add a single object prediction into expected_afford
    def _add_expected(obj_name: str, category: str, conf: float):
        if obj_name is None or category is None:
            return
        if conf is not None and conf < MIN_PRED_CONF:
            return
        key = _norm(obj_name)
        tag = CATEGORY_TO_INDEX_TAG.get(_norm(category), _norm(category))
        if not key or not tag:
            return
        expected_afford.setdefault(key, set()).add(tag)

    if USE_PER_SCENE_AFFORD:
        for scene in pred.get("per_scene_affordances", []):
            for obj in scene.get("objects", []):
                _add_expected(obj.get("object"), obj.get("category"), obj.get("confidence", 1.0))
    else:
        for obj_name, rec in pred.get("global_object_affordances", {}).items():
            _add_expected(rec.get("object") or obj_name, rec.get("category"), rec.get("confidence", 1.0))

    # convert sets to lists
    expected_afford = {k: sorted(list(v)) for k, v in expected_afford.items()}
else:
    expected_afford = {}


# Optional: expected affordance mapping
# --- Load expected affordances from prediction file ---
expected_afford = {}  # object_name -> list of expected tags (aligned to index)

# Optional mapper from predicted category -> your index’s affordance tag form
CATEGORY_TO_INDEX_TAG = {
    "Character": "Characters",
    "Characters": "Characters",
    "Interactive Object": "Interactive Object",
    "Item / Collectible": "Items and Collectibles",
    "Items and Collectibles": "Items and Collectibles",
    "Environmental Object": "Environmental Object",
    "Terrain": "Terrain",
    "Effect / Ambient / Unknown": "Effect / Ambient / Unknown",
}

if os.path.exists(AFFORD_PRED_FILE):
    with open(AFFORD_PRED_FILE, "r", encoding="utf-8") as f:
        pred = json.load(f)
    print(f"[info] Loaded affordance predictions from {AFFORD_PRED_FILE}")

    def _add_expected(obj_name: str, category: str, conf: float):
        if obj_name is None or category is None:
            return
        if conf is not None and conf < MIN_PRED_CONF:
            return
        key = _norm(obj_name)
        tag = CATEGORY_TO_INDEX_TAG.get(_norm(category), _norm(category))
        if not key or not tag:
            return
        expected_afford.setdefault(key, set()).add(tag)

    if USE_PER_SCENE_AFFORD:
        for scene in pred.get("per_scene_affordances", []):
            for obj in scene.get("objects", []):
                _add_expected(obj.get("object"), obj.get("category"), obj.get("confidence", 1.0))
    else:
        for obj_name, rec in (pred.get("global_object_affordances", {}) or {}).items():
            _add_expected(rec.get("object") or obj_name, rec.get("category"), rec.get("confidence", 1.0))

    # sets -> lists
    expected_afford = {k: sorted(list(v)) for k, v in expected_afford.items()}
    print(f"[info] Built expected affordances for {len(expected_afford)} objects")
else:
    print(f"[warn] No affordance prediction file found at {AFFORD_PRED_FILE}")
    expected_afford = {}





def get_embedding(text):
    return model.encode(text, show_progress_bar=False)

def is_probable_character(name):
    return name and name[0].isupper() and "_" not in name and len(name.split()) <= 2

def query_object(text, top_k=TOP_K, restrict_to_character=False):
    query_vec = get_embedding(text)
    rows = []
    for obj in index:
        if restrict_to_character and "Characters" not in obj.get("affordance", []):
            continue

        emb = obj.get("embedding", {})
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

    rows.sort(key=lambda r: r["total_score"], reverse=True)
    return rows[:top_k]

# === Main Processing ===
with open(STORY_FILE, "r", encoding="utf-8") as f:
    story = json.load(f)

scene_kgs = story["scene_kgs"]
matched_top1 = {}
matched_detailed = {}
objects_seen = set()

for scene, data in scene_kgs.items():
    for obj_name in data["nodes"]:
        if obj_name in objects_seen:
            continue
        objects_seen.add(obj_name)

        restrict = is_probable_character(obj_name)
        candidates = query_object(obj_name, top_k=TOP_K, restrict_to_character=restrict)

        # Derive margins and flags
        top1 = candidates[0] if candidates else None
        top2 = candidates[1] if len(candidates) > 1 else None
        top3 = candidates[2] if len(candidates) > 2 else None

        margin_12 = float(top1["total_score"] - top2["total_score"]) if (top1 and top2) else None
        margin_13 = float(top1["total_score"] - top3["total_score"]) if (top1 and top3) else None

        low_conf_flag = (top1 is not None and top1["total_score"] < CONF_THRESH_FOR_REVIEW)
        ambiguous_flag = (margin_12 is not None and margin_12 < REVIEW_MARGIN_THRESH)

        # Expected affordance check
        # expected = expected_afford.get(obj_name, [])
        # afford_match_flag = None  # None = no expectation; True/False when expectation exists
        # if expected:
        #     top1_affs = set(top1.get("candidate_affordances", [])) if top1 else set()
        #     afford_match_flag = bool(top1_affs.intersection(set(expected)))

        # Expected affordance check (uses predictions loaded above)
        expected = expected_afford.get(_norm(obj_name), [])
        afford_match_flag = None  # None = no expectation present; True/False when expectation exists
        if expected:
            top1_affs = set(top1.get("candidate_affordances", [])) if top1 else set()
            afford_match_flag = bool(top1_affs.intersection(set(expected)))



        # needs_review if any of the conditions is concerning:
        needs_review = False
        if low_conf_flag or ambiguous_flag:
            needs_review = True
        # If you want affordance mismatch to trigger review only when expectations exist:
        if afford_match_flag is False:
            needs_review = True

        # needs_review = False
        # if low_conf_flag or ambiguous_flag:
        #     needs_review = True
        # if afford_match_flag is False:  # only triggers when we had an expectation
        #     needs_review = True


        matched_top1[obj_name] = [top1["image_path"]] if top1 else []
        # matched_detailed[obj_name] = {
        #     "restricted_to_characters": bool(restrict),
        #     "query_text": obj_name,
        #     "expected_affordances": expected,           # new
        #     "affordance_match_top1": afford_match_flag, # new: None/True/False
        #     "needs_review": bool(needs_review),         # new
        #     "review_reasons": {
        #         "low_confidence": bool(low_conf_flag),
        #         "ambiguous_margin": bool(ambiguous_flag),
        #         "affordance_mismatch": (afford_match_flag is False),
        #     },                                          # new
        #     "margins": {
        #         "top1_minus_top2": margin_12,
        #         "top1_minus_top3": margin_13
        #     },                                          # new
        #     "candidates": candidates
        # }
        matched_detailed[obj_name] = {
            "restricted_to_characters": bool(restrict),
            "query_text": obj_name,

            # NEW: affordance expectations and match info
            "expected_affordances": expected,
            "candidate_affordances_top1": top1.get("candidate_affordances", []) if top1 else [],
            "affordance_match_top1": afford_match_flag,

            "needs_review": bool(needs_review),
            "review_reasons": {
                "low_confidence": bool(low_conf_flag),
                "ambiguous_margin": bool(ambiguous_flag),
                "affordance_mismatch": (afford_match_flag is False),
            },
            "margins": {
                "top1_minus_top2": margin_12,
                "top1_minus_top3": margin_13
            },
            "candidates": candidates
        }



# === Save Outputs ===
os.makedirs("StoryFiles", exist_ok=True)

with open(OUTPUT_TOP1, "w", encoding="utf-8") as fout:
    json.dump(matched_top1, fout, indent=2)
print(f"[✓] Saved object->top1 image mapping to {OUTPUT_TOP1}")

with open(OUTPUT_DETAILED, "w", encoding="utf-8") as fout:
    json.dump(matched_detailed, fout, indent=2)
print(f"[✓] Saved detailed matching (flags, margins, top-{TOP_K}) to {OUTPUT_DETAILED}")

# === Evaluation metrics ===
def safe_mean(values):
    return float(mean(values)) if values else 0.0

top1_totals = []
top1_name = []
top1_group = []
top1_super = []
top1_afford = []
margins_12 = []
margins_13 = []
high_conf_count = 0
diversity_set = set()

needs_review_count = 0
afford_known = 0
afford_match = 0

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
        margins_12.append(c1["total_score"] - cands[1]["total_score"])
    if len(cands) >= 3:
        margins_13.append(c1["total_score"] - cands[2]["total_score"])

    if c1["total_score"] >= HIGH_CONF_THRESH:
        high_conf_count += 1

    if pack.get("needs_review"):
        needs_review_count += 1

    am = pack.get("affordance_match_top1")
    if am is not None:
        afford_known += 1
        if am:
            afford_match += 1

num_objects = len(matched_detailed)
metrics = {
    "story_id": STORY_ID,
    "num_objects": num_objects,
    "top_k": TOP_K,
    "high_conf_threshold": HIGH_CONF_THRESH,
    "conf_thresh_for_review": CONF_THRESH_FOR_REVIEW,
    "review_margin_thresh": REVIEW_MARGIN_THRESH,

    "mean_top1_total": safe_mean(top1_totals),
    "mean_top1_sim_name": safe_mean(top1_name),
    "mean_top1_sim_group": safe_mean(top1_group),
    "mean_top1_sim_super": safe_mean(top1_super),
    "mean_top1_sim_afford": safe_mean(top1_afford),

    "mean_margin_top1_top2": safe_mean(margins_12),
    "mean_margin_top1_top3": safe_mean(margins_13),

    "pct_high_conf_at_1": (high_conf_count / num_objects) if num_objects else 0.0,
    "result_diversity": (len(diversity_set) / num_objects) if num_objects else 0.0,

    "needs_review_count": needs_review_count,
    "pct_needs_review": (needs_review_count / num_objects) if num_objects else 0.0,

    "afford_expect_known": afford_known,
    "afford_expect_match": afford_match,
    "afford_expect_match_rate": (afford_match / afford_known) if afford_known else None
}

with open(OUTPUT_METRICS, "w", encoding="utf-8") as fout:
    json.dump(metrics, fout, indent=2)
print(f"[✓] Saved matching metrics to {OUTPUT_METRICS}")
