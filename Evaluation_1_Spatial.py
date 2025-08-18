import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load Excel
df = pd.read_excel("Data/Evaluation_spatial.xlsx")

# Normalize and clean
df['Story'] = df['Story'].astype(str).str.strip()
df['verify'] = df['verify'].astype(int)

# Grouped stats
grouped = df.groupby('Story')
results = {}
total_predicates = 0
total_correct = 0

for story, group in grouped:
    total_rel = len(group)
    correct = group['verify'].sum()
    avg_per_scene = total_rel / 3  # assume 3 scenes per story

    results[f"Story_{story.zfill(2)}"] = {
        "avg_predicates_per_scene": round(avg_per_scene, 2),
        "satisfaction_rate": round(100 * correct / total_rel)
    }

    total_predicates += total_rel
    total_correct += correct

# Overall stats
overall_avg = round(total_predicates / len(grouped), 2)
overall_satisfaction = round(100 * total_correct / total_predicates)

# Binary classification metrics
y_true = [1] * len(df)  # ground truth: we assume all predicates are supposed to be satisfied
y_pred = df['verify'].tolist()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Add overall section
results["Overall"] = {
    "avg_predicates_per_scene": overall_avg,
    "satisfaction_rate": overall_satisfaction,
    "metrics": {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2)
    }
}

# Save JSON
with open("Data/spatial_satisfaction_result.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Print summary
print("Spatial Predicate Satisfaction Metrics:")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")
print("Saved to spatial_satisfaction_result.json")
