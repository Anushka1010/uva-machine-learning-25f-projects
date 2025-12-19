# Pneumonia Detection from Chest X-rays using CNNs and Grad-CAM

## Team Information
**Team ID:** Team 13  
**Team Members:**  
- Sid Luthra  
- Joe Girard  

---

## Overview
This project detects pneumonia from chest X-ray images using a convolutional neural network (CNN). A pre-trained ResNet-50 model is fine-tuned on the RSNA Pneumonia Detection Challenge dataset to perform binary classification between pneumonia and normal cases.

To improve interpretability, Grad-CAM is used to visualize regions of the X-ray images that most strongly influence the model’s predictions. The project includes DICOM image preprocessing, stratified dataset subsampling to preserve class balance, model training and evaluation, ROC analysis, and Grad-CAM visualizations.

---

## Usage
1. Download the RSNA Pneumonia Detection Challenge dataset following the instructions in `data/README.md`.
2. Upload the dataset ZIP file to your Google Drive inside a folder named **`dataset`**.
3. Open the project notebook in Google Colab.
4. Run the notebook cells in order to:
   - Mount Google Drive and load the dataset
   - Preprocess and cache DICOM images
   - Train the ResNet-50 CNN
   - Evaluate model performance
   - Generate Grad-CAM visualizations

All core results (metrics, plots, and visualizations) are produced automatically by the notebook.

---

## Setup Notes (Optional)
- The project was developed and tested in Google Colab.
- GPU acceleration is recommended for faster training.
- Preprocessing results are cached to disk to significantly reduce runtime.

