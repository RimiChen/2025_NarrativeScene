import json


SAVE_OUT_FOLDER = "StoryFiles/"
FILE_NUMBER = 0 #"StoryFiles/"+str(FILE_NUMBER)+"

# Load the current object propagation dictionary
with open("StoryFiles/"+str(FILE_NUMBER)+"_scene_object_propagation.json", "r") as f:
    original_data = json.load(f)

# Convert to list of dictionaries
converted_list = []
for scene_title, scene_info in original_data.items():
    entry = {"scene_title": scene_title}
    entry.update(scene_info)
    converted_list.append(entry)

# Save the new version
with open("StoryFiles/"+str(FILE_NUMBER)+"_scene_object_propagation_CONVERTED.json", "w") as f:
    json.dump(converted_list, f, indent=2)

print("Conversion complete. Output saved to StoryFiles/"+str(FILE_NUMBER)+"_scene_object_propagation_CONVERTED.json")
