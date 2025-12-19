# Full Pipeline: Rebuilding Synthetic YOLO Segmentation Data (using provided backgrounds)

This guide describes how to regenerate the synthetic datasets from scratch when you only have the repository and the included background images. No original TrashNet images or generated data are shipped.

## 0) Environment
- Create/activate a Python env and install deps: `bash scripts/setup_venv.sh` (or `pip install -r requirements.txt`).
- Ultralytics CLI will be available via the venv once installed.

## 1) Get raw TrashNet images (required)
1. Download TrashNet (e.g., https://github.com/garythung/trashnet).
2. Place class folders under `data/raw/trashnet/`:
   - `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.

## 2) Backgrounds (already provided)
- Background photos are included in `data/raw/backgrounds/` and used during compositing. No action needed.

## 3) Create cutouts and masks (foregrounds)
If you don’t have `data/derived/rmbg_cutouts/` and `data/derived/rmbg_masks/`:
1. Run your background-removal tool per class on `data/raw/trashnet/<class>/*.jpg|png`.
2. Save RGBA cutouts to `data/derived/rmbg_cutouts/<class>/*_rgba.png`.
3. Save binary masks to `data/derived/rmbg_masks/<class>/*_mask.png`.
4. (Optional) Use `scripts/run_rmbg_batch.py` as a template for batch processing.

## 4) Generate synthetic datasets
Use `scripts/generate_synthetic_yolo.py` to composite cutouts onto backgrounds and emit YOLOv8-ready images, masks, labels, and manifests.

- Glass/Metal example (default args):
```bash
source .venv/bin/activate  # if using the repo venv
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

- Plastic-only example:
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

Outputs (per `--out-root`):
- Images: `data/derived/<out-root>/images/{train,val,test}/*.png`
- Masks: `data/derived/<out-root>/masks/{train,val,test}/*.png`
- Labels (YOLO polygons): `data/derived/<out-root>/labels/{train,val,test}/*.txt`
- Manifest: `data/derived/<out-root>/manifest.csv`
- Dataset config: `data/derived/<out-root>/yolov8.yaml`

## 5) Train or validate (reference)
```bash
yolo train segment model=yolov8s-seg.pt data=data/derived/yolo_plastic/yolov8.yaml epochs=100 imgsz=1024
yolo val segment model=runs/segment/train8/weights/best.pt data=data/derived/yolo_plastic/yolov8.yaml imgsz=1024
```

This pipeline regenerates synthetic data without distributing any original TrashNet images; only the included backgrounds are bundled.
