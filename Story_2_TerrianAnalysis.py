import json
import time
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


STORY_ID = 0

# --- Config ---
INPUT_FILE = "StoryFiles/"+str(STORY_ID)+"_adventure_scene_output_FIXED.json"
OUTPUT_FILE = "StoryFiles/"+str(STORY_ID)+"_object_affordance_langchain.json"
# OPENAI_API_KEY = "your_openai_api_key_here"  # Replace with your API key

# llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY, temperature=0.2)
from sk import my_sk  # your OpenAI API key

# --- LLM Setup ---
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=my_sk, temperature=0.2)


# --- Prompt Template ---
template = """
You are an expert in game design and spatial reasoning. Classify each object below by its affordance level and suggest the most appropriate terrain type for placement.

Use the following affordance levels:
- 0: Terrain
- 1: Environmental Object
- 2: Interactive Object
- 3: Item / Collectible
- 4: Character
- -1: Effect / Ambient / Unknown

Return a JSON array like:
[
  {{
    "object": "object name",
    "affordance_level": 1,
    "category": "Environmental Object",
    "suggested_terrain": "concrete",
    "confidence": 0.9
  }},
  ...
]

Scene: {scene_title}
Objects:
{object_list}
"""

prompt = PromptTemplate(input_variables=["scene_title", "object_list"], template=template)
chain = LLMChain(llm=llm, prompt=prompt)

# --- Load input data ---
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# --- Process each scene ---
scene_affordances = []
global_obj_map = {}

for tf in data["time_frames"]:
    title = tf["title"]
    objects = set()

    for rel in tf["scene_relations"]:
        if "[" in rel and "]" in rel:
            pre, post = rel.split("[", 1)
            o1 = pre.strip()
            o2 = post.split("]", 1)[-1].strip()
            objects.update([o1, o2])

    sorted_objs = sorted(objects)
    print(f"🔍 Processing: {title} ({len(sorted_objs)} objects)")

    # Run LangChain
    try:
        response = chain.run({
            "scene_title": title,
            "object_list": "\n".join(sorted_objs)
        })
        parsed = json.loads(response)
    except Exception as e:
        print(f"⚠️ Error parsing scene '{title}':", e)
        parsed = []

    # Save per-scene
    scene_affordances.append({
        "scene_title": title,
        "objects": parsed
    })

    # Update global
    for obj in parsed:
        global_obj_map[obj["object"]] = obj

# --- Save output ---
output_data = {
    "global_object_affordances": global_obj_map,
    "per_scene_affordances": scene_affordances
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"✅ Saved affordance output to {OUTPUT_FILE}")
