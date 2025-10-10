## Script Index

### Narrative generation and parsing

| Script                                    | Main Functions                                                                         | Goal                                                             | Key Outputs                              |
| ----------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------- |
| `Story_1_NarrativePrompt_langchain_v2.py` | Prompts GPT-4 with a story template; extracts time frames and object–relation triples. | Generate a short adventure story with structured scene data.     | `<id>_adventure_scene_output_FIXED.json` |
| `Story_1_NarrativePrompt_langchain_v3.py` | Adds XML tagging and stricter parsing logic.                                           | Reduce parsing errors in LLM output.                             | Same as above                            |
| `Story_1_NarrativePrompt_langchain_v4.py` | Final version with stable prompt and parser for batch generation.                      | Robust narrative generation across multiple runs.                | Same as above                            |
| `Story_2_TerrianAnalysis.py`              | Classifies objects into affordance groups and suggests terrain.                        | Map nouns to terrain, environment, interactive, item, character. | `<id>_object_affordance_langchain.json`  |
| `Story_3_ParsingScenes.py`                | Determines base terrain and patch features for each scene.                             | Define terrain layout per scene.                                 | `<id>_scene_generation_decisions.json`   |

---

### Environment, interactive objects, items, and characters

| Script                                | Main Functions                                                                                  | Goal                                        | Key Outputs                                    |
| ------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------- | ---------------------------------------------- |
| `Story_4_ParsingEnvObjects.py`        | Collects environment objects by base terrain and propagates across scenes with identical bases. | Maintain consistent environmental context.  | `<id>_scene_object_propagation.json`           |
| `Story_5_ParsingEnvObjects_Format.py` | Converts propagation dictionary to list format.                                                 | Standardize format for later merging.       | `<id>_scene_object_propagation_CONVERTED.json` |
| `Story_6_ParsingItems.py`             | Extracts collectible items across base–patch pairs.                                             | Maintain stable item sets across scenes.    | `<id>_item_collectibles_expanded.json`         |
| `Story_7_ParsingInteractive.py`       | Extracts interactive objects and merges aliases using fuzzy matching.                           | Clean and unified interactive object list.  | `<id>_interactive_objects_expanded.json`       |
| `Story_8_ParsingCharacters.py`        | Identifies character entities per scene.                                                        | Build character lists for visual placement. | `<id>_character_objects_expanded.json`         |

---

### Knowledge graphs

| Script                                                 | Main Functions                                                             | Goal                                                     | Key Outputs                      |
| ------------------------------------------------------ | -------------------------------------------------------------------------- | -------------------------------------------------------- | -------------------------------- |
| `Story_9_ConstructSceneKG_spatial.py`                  | Translates narrative relations into spatial triples.                       | Normalize relations such as *left of*, *on top of*, etc. | `<id>_scene_from_relations.json` |
| `Story_10_ConstructSceneKG_objects.py`                 | Builds scene-level object graph with base, patch, and affordance grouping. | Structure object relations in scene KG.                  | Intermediate JSON                |
| `Story_11_ConstructSceneKG_combined_timeline.py`       | Merges scene and spatial graphs, adds temporal `precedes` edges.           | Construct story-level KG.                                | `<id>_scene_kg_combined.json`    |
| `Story_11_ConstructSceneKG_combined_timeline_patch.py` | Extended version with patched edges and scene cleanup.                     | Improve graph connectivity and consistency.              | `<id>_scene_kg_combined.json`    |
| `Story_12_Summarize_KG.py`                             | Summarizes scene elements, relations, and transitions.                     | Compact representation for visualization and reasoning.  | `<id>_scene_summaries.json`      |
| `ConstructSceneKG_combined.py`                         | Utility to combine all KGs from scenes and relations.                      | One-step KG builder.                                     | Combined KG JSON + visualization |
| `checkKG.py`                                           | Inspects KG JSONs for missing links or invalid relations.                  | Debug and validation tool.                               | Console output                   |

---

### Terrain synthesis and object placement

| Script                                  | Main Functions                                                                       | Goal                                     | Key Outputs                                          |
| --------------------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------- | ---------------------------------------------------- |
| `Scene_1_CA_Terrian_Visualizer_WEnv.py` | Generates base and patch terrains via Cellular Automata; overlays environment hints. | Create terrain masks and base structure. | Terrain matrices, visualization images               |
| `Scene_2_object_maps.py`                | Randomly places objects by affordance categories on terrain.                         | Basic layered placement.                 | Preview PNGs, JSON layers                            |
| `Scene_2_object_maps_v2.py`             | Refined placement and visual palette.                                                | Clearer visual outputs.                  | Improved PNGs and JSON layers                        |
| `Scene_3_object_relocate.py`            | Adjusts object positions according to spatial relations.                             | Ensure relation satisfaction.            | `<id>_scene_object_affordance_layers_RELOCATED.json` |
| `Scene_3_object_relocate_v2.py`         | Safer relocation and overlap handling.                                               | Improve placement accuracy.              | Same as above                                        |
| `Scene_4_replace_objects.py`            | Composites asset images onto grid cells.                                             | Create first rendered scenes.            | PNG renders                                          |
| `Scene_4_replace_objects_v2.py`         | Adds scaling and caching for better quality.                                         | Produce final renders.                   | PNG renders                                          |
| `Scene_4_terrian.py`                    | Renders terrain-only scenes for debugging or backdrop creation.                      | Clean terrain visual output.             | Terrain PNGs                                         |
| `LayerSceneGeneratorWithAffordance.py`  | Combines terrain, object, and affordance layers into a final layout.                 | Unified layered scene generator.         | `scene_layout_from_affordance.json`                  |

---

### Cross-modal matching and evaluation

| Script                                             | Main Functions                                                                    | Goal                                     | Key Outputs                                |
| -------------------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------ |
| `Generation_3_parse_story_mapping.py`              | Matches objects to GameTileNet asset embeddings using SentenceTransformer MiniLM. | Retrieve asset candidates for rendering. | `<id>_matched_objects.json`                |
| `Generation_3_parse_story_mapping_with_eval.py`    | Performs matching with baseline evaluation.                                       | Quantify semantic alignment.             | Matching metrics (JSON)                    |
| `Generation_3_parse_story_mapping_with_eval_v2.py` | Adds affordance consistency validation.                                           | Improve semantic coherence.              | Metrics JSON                               |
| `Generation_3_parse_story_mapping_with_eval_v3.py` | Includes fuzzy alignment and tie-breaking heuristics.                             | Refine top-k selection.                  | Metrics JSON                               |
| `Generation_4_story_eval_metrics.py`               | Aggregates all evaluation results per story.                                      | Summarize quality of matches.            | `<id>_matching_metrics.json` + summary CSV |
| `Evaluation_1_Spatial.py`                          | Calculates precision, recall, and F1 scores for spatial predicates.               | Evaluate spatial relation correctness.   | `spatial_satisfaction_result.json`         |

---

### Minimal runnable examples

| Script                           | Main Functions                                                                           | Goal                           | Key Outputs                        |
| -------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------ | ---------------------------------- |
| `Example_1_parse_text.py`        | Parses a short human-written description (e.g., "Tree above Rock") into structured JSON. | Minimal parsing demonstration. | `single_scene_forest.json`         |
| `Example_2_generate_scene.py`    | Generates terrain and object placement from parsed JSON.                                 | End-to-end prototype example.  | Scene PNG + JSON matrices          |
| `Example_2_generate_scene_v2.py` | Uses fixed color palettes and offsets.                                                   | Produce cleaner figures.       | Refined PNG and JSON               |
| `Example_3_matched.py`           | Matches story objects to image assets.                                                   | Verify object-image pipeline.  | `*_scene_objects_with_images.json` |
| `Example_3_matched_v2.py`        | Composites final visual scenes using matched assets.                                     | Render final scene snapshot.   | PNG render                         |
