# Instruction Clarification Requests in the CoDraw Dataset

This is the code accompanying the EACL 2023 submission: Instruction Clarification Requests in Multimodal Collaborative Dialogue
Games: Tasks, and an Analysis of the CoDraw Dataset.

This anonymised repository is meant only for inspection purposes during review. A fully documented repository will be publicly available afterwards.

The annotated data is also temporarily available in this repository ```data/cr_anno-main/data.tsv``` for inspection purposes during review. An official version with a proper license will also be published later for the community in another link.

The main code is under ```src/```.

Necessary data:

- CoDraw JSON file data from [the original CoDraw data repository](https://github.com/facebookresearch/CoDraw)
- scripts abs_metric.py, abs_util_orig.py and codraw_data.py from [the original CoDraw model repository](https://github.com/facebookresearch/codraw-models), used to compute scores
- AbstractScenes from [this link](https://www.microsoft.com/en-ca/download/details.aspx?id=52035)
- Incremental scenes generated from the scene string in the JSON file using the SceneRenderer.exe from AbstractScenes.
- Preprocessed embeddings using the corresponding .py scripts.

The script run.sh can be used to replicate the experiments in the paper.
