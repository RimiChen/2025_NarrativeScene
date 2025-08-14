# GameTileNet
A dataset for game tiles collected from OpenGameArt.org (CC licenses), with semantic labels and other information.

---
### File Structures

- **2024-GameTileNet**/
  - **src**/
    - environment.yml
    - SliceTile.py
    - LabelSimilarityGUI.py
    - CheckTileSimilarity.py
    - LabelingConnectivityGUI.py
    - LabelingConnectivityGUI2.py
    - LabelingObjectGUI.py            
    - CreateFileList.py
    - Transform16-32.py
    - Test.py
    - DisplayImage.py
    - FindNeighbots.py
    - SelectImagePairs.py
    - StatisticsForSimilarity.py
    - LabelingSimilarityAuto.py
    - CompareSimilarityLabels.py
    - Identify16-32.py
    - FindSegmentations.py
    - ExtractObject.py
    - ExtractObjectModel.py
    - CheckTilesetGrid.py
    - LabelingCompleteness.py
    - Transform16-32.py            
  <!-- - **annotations**/
    - dataset1.csv
    - dataset2.csv -->
  - README.md

  ---
### File Descriptions

* SliceTile.py: Slice tilesets into 32x32 size, and find corresponding segements on the tileset images. Create and save the slices in folder: <tileset_name: (collection_index)_(tileset_index)>.
* environment.yml: the environment settings for conda env. 
* LabelSimilarityGUI.py: Record the similarity of two adjacent tiles.
* CheckTileSimilarity.py: Check the similarity of two adjacent tiles, and other image processing functions: filter the blank image, set similarity to zero if all side blank, etc.. 
* LabelingConnectivityGUI.py: Record tile's possible connecting directions.
* LabelingConnectivityGUI2.py: Record tile's possible connecting directions. New version.
* LabelingObjectGUI.py: Record user input for labeling Object names.            
* CreateFileList.py: Import a folder and create a file list based on files inside.
* Transform16-32.py: Tranform 16x16 tile to 32x32 tiles.
* Test.py: Mixed testing scripts
* DisplayImage.py: Dispaly tileset with tile_size grids.
* FindNeighbots.py: For each split tiles in tileset folder, find all their exist neighbors.
* SelectImagePairs.py: Select adjacent image pairs randomly.
* StatisticsForSimilarity.py: Collect the similarity annotations and computer statistic values.
* LabelingSimilarityAuto.py: Automatically decide whether the two tiles are connected.
* CompareSimilarityLabels.py: Compare the similarity labeling results.
* Identify16-32.py: Identify whetehr the tileset is 32 based or 16 based.
* FindSegmentations.py: Find related tiles by adjacency.
* ExtractObject.py: Edge detection for objects on tileset.
* ExtractObjectModel.py: Segmentation with Segment anything model.
* CheckTilesetGrid.py: Generated assigned size grid for tileset images, to check whether the tileset is 32 or 16.
* LabelingCompleteness.py: Allow users to labeling the completeness of the segmented tiles.
* Transform16-32.py: Transform 16x16 tile into 32x32.  
---
### Instructions

Conda environment for the scripts and models:

```
conda env create -f environment.yml
```

---

### Citation

bibTex:

```
@article{chen2025gametilenet,
  title={GameTileNet: A Semantic Dataset for Low-Resolution Game Art in Procedural Content Generation},
  author={Chen, Yi-Chun and Jhala, Arnav},
  journal={arXiv preprint arXiv:2507.02941},
  year={2025}
}

```
 
