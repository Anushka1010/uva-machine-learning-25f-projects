# Classifying Skin Lesions

## Team
**Team ID:15**  
**Team Members:**  
- Brian Xu
- Caroline Xu  
- Evan Zhang

## Overview
Provide a brief introduction to the project, including the problem it addresses, the main idea, and key contributions.
Skin Lesion Detection.
This is an evaluation of the performance of EfficientNet-B0 against other popular CNNs.
- In particular, against ResNet50
- ResNet50 achieved the highest accuracy in a previous study (https://doi.org/10.48550/arXiv.2305.11125)
According to the ImageNet testing, EfficientNet should have relative (if not higher) accuracy at a greatly decreased model size.

## Usage + Setup
Currently the notebook is set up to have the dataset info stored in Google Drive at \Colab Notebooks\kaggle
Set the directories to wherever the data is stored.
The data is downloaded from: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- Click download.
- Extract from the downloaded zip file.
- Compress the HAM10000_images_part_1 and HAM10000_images_part 2 to zip files
- Ensure that DATA_DIR, zip_path_part_1, zip_path_part_2, and extract_path point to the proper directories
- Ensure that METADATA_DIR points to the directory with the HAM10000_metadata.csv file
Notebook is run with hyperparameters set to the following to match the referenced study
- Batch size: 64
- Learning rate: 1e-4
- Epochs: 5
Set epochs to 100 for full run.

## Code Demo Video
https://youtu.be/dWKxdcSMWO4