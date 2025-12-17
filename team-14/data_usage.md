# Data Usage

## Our Custom Dataset
Our custom dataset consists of labeled images of common refrigerator items captured under various conditions:
- Different lighting scenarios
- Various angles and positions
- Different levels of occlusion (hands)
- Multiple instances of the same item type

Dataset Format: YOLOv5 PyTorch format via Roboflow
- Images with bounding box annotations
- Train/validation/test splits managed through Roboflow

The dataset is managed through Roboflow and accessed via API during model training.

## Data Access
Since we access the data directly through Roboflow via API, please contact us if you would like the key to replicate the model training.