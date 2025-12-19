# Generating Synthetic YOLO Segmentation Data (no raw uploads)

This guide explains how to rebuild the synthetic datasets without shipping any original data. It covers obtaining TrashNet, using the bundled backgrounds, and running the generation scripts.

## 0) Prerequisites
- Python env: see `requirements.txt` or run `bash scripts/setup_venv.sh`.
- Background images: already included in `data/raw/backgrounds/`.
- RMBG cutouts and masks: expected under `data/derived/rmbg_cutouts/` and `data/derived/rmbg_masks/` (one subfolder per class). If you need to recreate them, use your background-removal pipeline on the raw TrashNet images, then place the cutout RGBA PNGs in `rmbg_cutouts/<class>/` and corresponding masks in `rmbg_masks/<class>/` with `_rgba.png` / `_mask.png` naming (see existing files for examples).

## 1) Download TrashNet (required raw source)
1. Get the TrashNet dataset (e.g., via https://github.com/garythung/trashnet or other mirrors).
2. Place class folders under `data/raw/trashnet/`:
   ```
   data/raw/trashnet/cardboard
   data/raw/trashnet/glass
   data/raw/trashnet/metal
   data/raw/trashnet/paper
   data/raw/trashnet/plastic
   data/raw/trashnet/trash
   ```
3. Ensure images are JPG/PNG with original filenames intact.

## 2) (Optional) Rebuild cutouts/masks
If you need fresh cutouts/masks:
1. Run your RMBG tool on the raw images (per class) to produce:
   - `data/derived/rmbg_cutouts/<class>/*_rgba.png` (RGBA cutouts)
   - `data/derived/rmbg_masks/<class>/*_mask.png` (binary masks)
2. You can use `scripts/run_rmbg_batch.py` as a template if you adapt it to your RMBG model.

## 3) Generate synthetic YOLO data
Use `scripts/generate_synthetic_yolo.py` to composite cutouts onto backgrounds and emit YOLOv8-ready images/masks/labels/manifests.

Example: regenerate the glass/metal dataset (default classes):
```bash
source .venv/bin/activate  # or your env
python scripts/generate_synthetic_yolo.py \
  --classes glass metal \
  --cutouts-root data/derived/rmbg_cutouts \
  --masks-root data/derived/rmbg_masks \
  --backgrounds-root data/raw/backgrounds \
  --raw-root data/raw/trashnet \
  --out-root data/derived/yolo_glass_metal \
  --num-augments 5 \
  --max-bg-dim 1024 \
  --scale-min 0.4 --scale-max 0.85 \
  --rotation-max 20
```

Example: plastic-only synthetic set:
```bash
python scripts/generate_synthetic_yolo.py \
  --classes plastic \
  --cutouts-root data/derived/rmbg_cutouts \
  --masks-root data/derived/rmbg_masks \
  --backgrounds-root data/raw/backgrounds \
  --raw-root data/raw/trashnet \
  --out-root data/derived/yolo_plastic \
  --num-augments 5
```

Outputs for each run:
- Images: `data/derived/<out-root>/images/{train,val,test}/*.png`
- Masks: `data/derived/<out-root>/masks/{train,val,test}/*.png`
- Labels (YOLO polygon): `data/derived/<out-root>/labels/{train,val,test}/*.txt`
- Manifest: `data/derived/<out-root>/manifest.csv`
- Dataset config: `data/derived/<out-root>/yolov8.yaml`

## 4) Train / evaluate (reference)
With the generated dataset, you can train or validate using Ultralytics YOLO, e.g.:
```bash
 yolo train segment model=yolov8s-seg.pt data=data/derived/yolo_plastic/yolov8.yaml epochs=100 imgsz=1024
 yolo val segment model=runs/segment/train8/weights/best.pt data=data/derived/yolo_plastic/yolov8.yaml imgsz=1024
```

This document allows reproduction of the synthetic data without uploading any original raw images. Only backgrounds are included; TrashNet must be downloaded separately.
