import json

SAVE_OUT_FOLDER = "StoryFiles/"
FILE_NUMBER = 0 #"StoryFiles/"+str(FILE_NUMBER)+"

# Load input file
with open("StoryFiles/"+str(FILE_NUMBER)+"_adventure_scene_output_FIXED.json", "r") as f:
    data = json.load(f)

# Initialize output list
kg_per_scene = []

# Retrieve mapping from predicates to spatial relations
relation_map = data["relation_mapping"]

# Process each scene
for frame in data["time_frames"]:
    scene_title = frame["title"]
    scene_relations = frame["scene_relations"]
    triples = []

    for rel in scene_relations:
        if "[" in rel and "]" in rel:
            parts = rel.split("[")
            source = parts[0].strip()
            rest = parts[1].split("]")
            predicate = rest[0].strip()
            target = rest[1].strip()

            # Only consider mapped spatial relations
            if predicate in relation_map:
                spatial_relation = relation_map[predicate]
                if spatial_relation in ["above", "below", "at the right of", "at the left of", "on top of"]:
                    triples.append({
                        "source": source,
                        "target": target,
                        "relation": spatial_relation,
                        "original_predicate": predicate
                    })

    kg_per_scene.append({
        "scene_title": scene_title,
        "triples": triples
    })

# Save to file
with open("StoryFiles/"+str(FILE_NUMBER)+"_scene_from_relations.json", "w") as f_out:
    json.dump(kg_per_scene, f_out, indent=2)

print("✅ Knowledge Graph JSON saved as scene_kg_from_relations.json")
