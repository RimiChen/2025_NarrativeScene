import json

with open("StoryFiles/output_KG_story_1/1_kg_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Print structure
print("Top-level type:", type(data))
if isinstance(data, dict):
    print("Top-level keys:", list(data.keys()))
    for key in data:
        print(f"\n--- {key} ---")
        print(data[key][:1] if isinstance(data[key], list) else data[key])
else:
    print("Data is not a dictionary. Sample:", data[:1])
