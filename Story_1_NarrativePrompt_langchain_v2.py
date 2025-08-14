import warnings
import json
import re
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sk import my_sk  # your OpenAI API key

STORY_ID = 10

# --- LLM Setup ---
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=my_sk, temperature=0.7)

# --- Prompt Template ---
prompt_template_text = """
You are a storyteller who creates thrilling adventure stories.
Please generate an adventure story within 100 words.
Retrieve three key time frames from the story, and describe them with [Object] [Relation] [Object] type of scene descriptions.

Please tell the story in a captivating and imaginative way, engaging the reader from beginning to end.

Here is a sample format:
"In the heart of the Enchanted Forest, young Elara discovered an ancient map hidden within a hollow oak..."

Time Frame: Elara discovers the ancient map
Hollow oak [contains] ancient map
Elara [stands near] hollow oak
Sunlight [filters through] forest canopy

Time Frame: Elara faces the treacherous paths
...

After getting the time frames, generate a list that maps all the relations into the spatial categories:
"above", "below", "at the right of", "at the left of", and "on top of".
Show the mapping in the form: relation - mapped result
"""

prompt = PromptTemplate(template=prompt_template_text)
chain = LLMChain(llm=llm, prompt=prompt)

# --- Run LLM Chain ---
response = chain.run({})

# --- Extract story and lines ---
story_match = re.search(r'^(.*?)Time Frame:', response, re.DOTALL)
lines = response.splitlines()

# --- Process lines into structure ---
output_data = {
    "original_story": story_match.group(1).strip() if story_match else "",
    "time_frames": [],
    "relation_mapping": {}
}

current_frame = None
current_relations = []

for line in lines:
    line = line.strip()

    if line.startswith("Time Frame:"):
        if current_frame:
            output_data["time_frames"].append({
                "title": current_frame,
                "scene_relations": current_relations
            })
        current_frame = line.replace("Time Frame:", "").strip()
        current_relations = []
    elif "Mapping" in line and "-" not in line:
        if current_frame:
            output_data["time_frames"].append({
                "title": current_frame,
                "scene_relations": current_relations
            })
            current_frame = None
            current_relations = []
    elif current_frame and line and "-" not in line:
        current_relations.append(line)
    elif "-" in line:
        parts = line.split("-", 1)
        if len(parts) == 2:
            key = parts[0].strip()
            value = parts[1].strip()
            output_data["relation_mapping"][key] = value

# Append any remaining frame
if current_frame and current_relations:
    output_data["time_frames"].append({
        "title": current_frame,
        "scene_relations": current_relations
    })

# --- Save fixed output ---
with open("StoryFiles/"+str(STORY_ID)+"_adventure_scene_output_FIXED.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print("Fixed version saved to adventure_scene_output_FIXED.json.")