# Dataset Access Instructions

## RSNA Pneumonia Detection Challenge Dataset

This project uses the RSNA Pneumonia Detection Challenge dataset, which consists of chest X-ray images in DICOM format and associated labels for pneumonia detection.

Due to the large size of the dataset, the data is not included in this repository.

---

## How to Obtain the Dataset

1. Create or log in to a Kaggle account.
2. Visit the competition page:
   https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
3. Accept the competition rules.
4. Download the dataset ZIP file.

---

## How to Store the Dataset

After downloading the ZIP file, upload it to your Google Drive and place it inside a folder named: **`dataset`**

This folder should be located at the root level of your Google Drive. The Google Colab notebook mounts Google Drive and expects the dataset to be stored in this location.

---

## Notes
- Only `stage_2_train_images/` and `stage_2_train_labels.csv` are required to reproduce the core results.
- The code performs a stratified 40% subsample of the dataset to reduce runtime while preserving class balance.
- Preprocessed images are cached locally during execution to improve training efficiency.



