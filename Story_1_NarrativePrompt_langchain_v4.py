import warnings
import json
import re
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sk import my_sk  # your OpenAI API key

STORY_ID = 12

# --- LLM Setup ---
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=my_sk, temperature=0.7)

# --- Prompt Template ---
# prompt_template_text = """
# You are a storyteller who creates thrilling adventure stories.
# Please generate an adventure story within 100 words.
# Retrieve three key time frames from the story, and describe them with [Object] [Relation] [Object] type of scene descriptions.

# Please tell the story in a captivating and imaginative way, engaging the reader from beginning to end.

# Here is a sample format:
# "In the heart of the Enchanted Forest, young Elara discovered an ancient map hidden within a hollow oak..."

# Time Frame: Elara discovers the ancient map
# Hollow oak [contains] ancient map
# Elara [stands near] hollow oak
# Sunlight [filters through] forest canopy

# Time Frame: Elara faces the treacherous paths
# ...

# After getting the time frames, generate a list that maps all the relations into the spatial categories:
# "above", "below", "at the right of", "at the left of", and "on top of".
# Show the mapping in the form: relation - mapped result
# """
prompt_template_text = """
You are a storyteller who creates thrilling adventure stories.

Your task is to:
1. Write an engaging adventure story in under 100 words.
2. Extract three key time frames from the story.
3. For each time frame, list 3 scene descriptions in the format: [Object] [Relation] [Object]
4. Map each relation (e.g., 'contains', 'hovers above') to one of these spatial categories:
   "above", "below", "at the right of", "at the left of", "on top of"

Wrap the story between <STORY> and </STORY>.

### Example:
<STORY>
In the heart of the Enchanted Forest, young Elara discovered an ancient map hidden within a hollow oak. It led her to the legendary Crystal Cavern, rumored to grant the finder a single wish. Braving treacherous paths and wild creatures, Elara reached the cavern's shimmering entrance. Inside, she faced the Guardian, a majestic dragon. With courage and wit, she solved the Guardian’s riddle, earning her the wish. Elara wished for peace in her war-torn village. As she exited the cavern, the skies cleared, and harmony was restored, proving that bravery and hope could transform the world.
</STORY>

Time Frame: Elara discovers the ancient map  
Hollow oak [contains] ancient map  
Elara [stands near] hollow oak  
Sunlight [filters through] forest canopy  

Time Frame: Elara faces the treacherous paths  
Elara [crosses] rickety bridge  
Beasts [hide beneath] twisted trees  
Wind [howls through] mountain pass  

Time Frame: Elara solves the Guardian’s riddle  
Elara [faces] Guardian  
Guardian [guards] glowing cavern  
Elara [holds] ancient map  

Relation Mapping:  
contains - on top of  
stands near - at the left of  
filters through - above  
crosses - on top of  
hide beneath - below  
howls through - above  
faces - at the right of  
guards - on top of  
holds - at the right of
"""




prompt = PromptTemplate(template=prompt_template_text)
chain = LLMChain(llm=llm, prompt=prompt)

# --- Run LLM Chain ---
response = chain.run({})

# --- Extract story and lines ---
# story_match = re.search(r'^(.*?)Time Frame:', response, re.DOTALL)
story_match = re.search(r'<STORY>(.*?)</STORY>', response, re.DOTALL)
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
    # elif "-" in line:
    #     parts = line.split("-", 1)
    #     if len(parts) == 2:
    #         key = parts[0].strip()
    #         value = parts[1].strip()
    #         output_data["relation_mapping"][key] = value
    elif "-" in line:
        # Only keep lines that look like: relation - spatial_category
        parts = line.split("-", 1)
        if len(parts) == 2:
            key = parts[0].strip()
            value = parts[1].strip().lower()
            # Keep only allowed spatial relations
            valid_spatial = {"above", "below", "at the right of", "at the left of", "on top of"}
            if value in valid_spatial and len(key.split()) <= 3:  # Avoid full sentence as key
                output_data["relation_mapping"][key] = value


# Append any remaining frame
if current_frame and current_relations:
    output_data["time_frames"].append({
        "title": current_frame,
        "scene_relations": current_relations
    })


# --- Validate that all used relations are mapped ---
used_relations = set()
for frame in output_data["time_frames"]:
    for pred in frame["scene_relations"]:
        match = re.search(r'\[(.*?)\]', pred)
        if match:
            relation = match.group(1).strip()
            used_relations.add(relation)

missing_relations = used_relations - set(output_data["relation_mapping"].keys())
if missing_relations:
    print(f"⚠️ Missing relation mappings for: {missing_relations}")

    # Ask LLM to map the missing relations
    followup_prompt_text = f"""
    Map the following relations to one of the spatial categories:
    "above", "below", "on top of", "at the left of", "at the right of"

    Only include valid mappings in the format:
    relation - category

    Relations:
    {chr(10).join(sorted(missing_relations))}
    """

    from langchain_core.prompts import PromptTemplate as SimplePrompt
    followup_prompt = SimplePrompt(input_variables=["input"], template="{input}")
    followup_chain = LLMChain(llm=llm, prompt=followup_prompt)

    followup_response = followup_chain.run({"input": followup_prompt_text})
    print("🧩 Follow-up mapping:\n" + followup_response)

    # Parse follow-up mappings
    for line in followup_response.splitlines():
        parts = line.strip().split("-", 1)
        if len(parts) == 2:
            key = parts[0].strip()
            value = parts[1].strip().lower()
            if value in valid_spatial:
                output_data["relation_mapping"][key] = value


# --- Save fixed output ---
with open("StoryFiles/"+str(STORY_ID)+"_adventure_scene_output_FIXED.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print("Fixed version saved to adventure_scene_output_FIXED.json.")