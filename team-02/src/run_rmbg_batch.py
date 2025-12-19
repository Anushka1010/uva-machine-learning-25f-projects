#!/usr/bin/env python
"""
Batch background removal using the locally cached BRIA RMBG-2.0 ONNX model.

Outputs:
- Alpha masks: data/derived/rmbg_masks/<class>/<stem>_mask.png
- RGBA cutouts: data/derived/rmbg_cutouts/<class>/<stem>_rgba.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageEnhance


def load_session(model_path: Path) -> ort.InferenceSession:
    # Prefer CUDA if available, otherwise CPU. Disable memory pattern/reuse to
    # avoid noisy shape-mismatch warnings on dynamic inputs, and raise ORT log
    # level to error to suppress harmless warnings.
    ort.set_default_logger_severity(3)  # 0=verbose,1=info,2=warning,3=error
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.enable_mem_pattern = False
    opts.enable_mem_reuse = False
    try:
        session = ort.InferenceSession(str(model_path), providers=providers, sess_options=opts)
    except Exception as exc:
        # If CUDA provider fails to load (e.g., missing libcublas), fall back to CPU.
        logging.warning("Falling back to CPUExecutionProvider: %s", exc)
        session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
    return session


def preprocess(image: Image.Image, target: int) -> tuple[np.ndarray, tuple[int, int]]:
    orig_w, orig_h = image.size
    resized = image.resize((target, target), Image.Resampling.LANCZOS)
    arr = np.asarray(resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[None, ...]  # NCHW
    return arr, (orig_w, orig_h)


def postprocess(alpha: np.ndarray, orig_size: tuple[int, int]) -> Image.Image:
    # alpha is (1,1,H,W)
    alpha = np.squeeze(alpha)
    alpha = np.clip(alpha, 0, 1)
    alpha = (alpha * 255).astype(np.uint8)
    alpha_img = Image.fromarray(alpha, mode="L")
    if alpha_img.size != orig_size:
        alpha_img = alpha_img.resize(orig_size, Image.Resampling.LANCZOS)
    return alpha_img


def maybe_enhance(image: Image.Image) -> Image.Image:
    # Light color jitter to diversify cutouts a bit.
    rng = np.random.default_rng()
    brightness = float(rng.uniform(0.95, 1.05))
    contrast = float(rng.uniform(0.95, 1.05))
    color = float(rng.uniform(0.95, 1.05))
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(color)
    return image


def process_image(
    img_path: Path,
    sess: ort.InferenceSession,
    masks_dir: Path,
    cutouts_dir: Path,
    target: int,
) -> None:
    mask_path = masks_dir / f"{img_path.stem}_mask.png"
    cutout_path = cutouts_dir / f"{img_path.stem}_rgba.png"
    if mask_path.exists() and cutout_path.exists():
        logging.info("Skip existing outputs for %s", img_path)
        return

    image = Image.open(img_path).convert("RGB")
    arr, orig_size = preprocess(image, target)
    alpha = sess.run(None, {"pixel_values": arr})[0]
    alpha_img = postprocess(alpha, orig_size)

    masks_dir.mkdir(parents=True, exist_ok=True)
    cutouts_dir.mkdir(parents=True, exist_ok=True)

    # Save mask
    mask_path = masks_dir / f"{img_path.stem}_mask.png"
    alpha_img.save(mask_path)

    # Save cutout (apply slight jitter to RGB)
    rgba = maybe_enhance(image).convert("RGBA")
    rgba.putalpha(alpha_img)
    rgba.save(cutout_path)


def run_batch(
    classes: Iterable[str],
    raw_root: Path,
    masks_root: Path,
    cutouts_root: Path,
    model_path: Path,
    target: int,
) -> None:
    sess = load_session(model_path)
    all_imgs: List[Path] = []
    for cls in classes:
        cls_dir = raw_root / cls
        if not cls_dir.exists():
            logging.warning("Skipping missing class dir %s", cls_dir)
            continue
        imgs = sorted(p for p in cls_dir.iterdir() if p.is_file())
        all_imgs.extend(imgs)
        logging.info("Class %s: %d images", cls, len(imgs))

    for idx, img_path in enumerate(all_imgs, 1):
        cls = img_path.parent.name
        logging.info("Processing %s (%d/%d)", img_path, idx, len(all_imgs))
        process_image(
            img_path,
            sess,
            masks_root / cls,
            cutouts_root / cls,
            target,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch RMBG-2.0 inference.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["glass", "metal"],
        help="Class subdirectories under raw_root to process.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/trashnet"),
        help="Root containing per-class image folders.",
    )
    parser.add_argument(
        "--masks-root",
        type=Path,
        default=Path("data/derived/rmbg_masks"),
        help="Output root for alpha masks.",
    )
    parser.add_argument(
        "--cutouts-root",
        type=Path,
        default=Path("data/derived/rmbg_cutouts"),
        help="Output root for RGBA cutouts.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "models/rmbg2/models--briaai--RMBG-2.0/snapshots/21d558afe7050ed1d07bf7302ed77e1f440c81ab/onnx/model.onnx"
        ),
        help="Path to the RMBG-2.0 ONNX file.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Square resize dimension fed into RMBG (smaller is faster).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_batch(args.classes, args.raw_root, args.masks_root, args.cutouts_root, args.model_path, args.size)


if __name__ == "__main__":
    main()
