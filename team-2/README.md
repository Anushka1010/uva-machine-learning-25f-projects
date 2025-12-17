# YOLO-recycler

## Team ID and Members
Team 2: Dominic Tran (tfe2zz) and Steven Siadaty (vth3bk)

## Overview
This repo contains the pipeline we used to generate synthetic foregrounds/backgrounds from TrashNet, produce a YOLOv8-ready segmentation dataset, and train a single-class plastic segmentation model.


## Quickstart (core results)
- Create/activate the venv and install deps (defaults to CUDA 12.4 wheels; set `TORCH_INDEX=https://download.pytorch.org/whl/cpu` for CPU):  
  `bash scripts/setup_venv.sh && source .venv/bin/activate`
- The synthetic dataset used for core results is already in `data/derived/yolo_plastic/` (untar `data/derived/yolo_plastic.tar.gz` into `data/derived/` if you only see the archive).
- Train from scratch (YOLOv8s-seg, 100 epochs, 1024px):  
  `yolo train segment model=yolov8s-seg.pt data=data/derived/yolo_plastic/yolov8.yaml epochs=100 imgsz=1024 batch=8 project=runs/segment name=train_local`
- Evaluate the saved checkpoint we report (`runs/segment/train8/weights/best.pt`):  
  `yolo val segment model=runs/segment/train8/weights/best.pt data=data/derived/yolo_plastic/yolov8.yaml imgsz=1024`
- Run inference on your own images:  
  `yolo predict segment model=runs/segment/train8/weights/best.pt source='data/derived/yolo_plastic/images/val/*.png' imgsz=1024 save=True`

## Regenerate the synthetic dataset
If you want to rebuild the YOLO data rather than using `data/derived/yolo_plastic/`:
1) Download TrashNet and place class folders under `data/raw/trashnet/`.  
2) Ensure cutouts and masks exist under `data/derived/rmbg_cutouts/<class>/` and `data/derived/rmbg_masks/<class>/` (use your RMBG tool if needed; backgrounds are already under `data/raw/backgrounds/`).  
3) Run:  
```
python scripts/generate_synthetic_yolo.py \
  --classes plastic \
  --cutouts-root data/derived/rmbg_cutouts \
  --masks-root data/derived/rmbg_masks \
  --backgrounds-root data/raw/backgrounds \
  --raw-root data/raw/trashnet \
  --out-root data/derived/yolo_plastic \
  --num-augments 5 \
  --max-bg-dim 1024
```
4) Train/validate with the generated `data/derived/yolo_plastic/yolov8.yaml` as in the quickstart above.

## References
- Dataset/config: `data/derived/yolo_plastic/yolov8.yaml`, manifests in `data/derived/yolo_plastic/`.
- Training outputs: checkpoints and metrics under `runs/segment/train8/`.
- More details on data synthesis: `docs/GENERATE_SYNTHETIC_DATA.md` and `docs/SYNTHETIC_DATA_PIPELINE.md`.

## Video Demo links
- Slide Deck: https://youtu.be/YcraLptXeWg
- Code Base: https://youtu.be/L-Bif0_NOkU