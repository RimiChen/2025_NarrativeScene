import json
import glob
import numpy as np

# Load all intersection metrics
files = sorted(glob.glob("StoryFiles/*_matching_metrics_intersection.json"))
all_metrics = []

for f in files:
    with open(f) as fp:
        data = json.load(fp)
        data["filename"] = f
        all_metrics.append(data)

# Extract metrics across stories
aff_rates = [m["afford_expect_match_rate_intersect"] for m in all_metrics]
cos_sims = [m["mean_top1_total_intersect"] for m in all_metrics]
diversities = [m["result_diversity_intersect"] for m in all_metrics]

print("=== Tile Matching Evaluation across 10 Stories ===")
print(f"Mean Affordance Match Rate: {np.mean(aff_rates):.2f} ± {np.std(aff_rates):.2f}")
print(f"Mean Top-1 Cosine Similarity: {np.mean(cos_sims):.2f} ± {np.std(cos_sims):.2f}")
print(f"Mean Diversity: {np.mean(diversities):.2f} ± {np.std(diversities):.2f}")

# Per-story summary table
print("\nPer-story results:")
print("Story | CosSim | AffordMatch | Diversity")
for m in all_metrics:
    sid = m["story_id"]
    print(f"{sid:>5} | {m['mean_top1_total_intersect']:.2f} | "
          f"{m['afford_expect_match_rate_intersect']:.2f} | "
          f"{m['result_diversity_intersect']:.2f}")
