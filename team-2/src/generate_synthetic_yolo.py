#!/usr/bin/env python
"""
Generate synthetic composites for segmentation training using RMBG cutouts and masks.

Outputs a YOLOv8-ready dataset with images, masks, labels, and a manifest:
- Images: data/derived/yolo_glass_metal/images/{train,val,test}/*.png
- Masks:  data/derived/yolo_glass_metal/masks/{train,val,test}/*.png
- Labels: data/derived/yolo_glass_metal/labels/{train,val,test}/*.txt (polygon format)
- Manifest: data/derived/yolo_glass_metal/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance


@dataclass
class SampleMeta:
    synthetic_path: Path
    mask_path: Path
    label_path: Path
    source_image: Path
    source_mask: Path
    cutout_path: Path
    background: Path
    split: str
    scale_factor: float
    rotation_deg: float
    offset_x: int
    offset_y: int


def list_backgrounds(backgrounds_root: Path) -> List[Path]:
    backgrounds = sorted(p for p in backgrounds_root.iterdir() if p.is_file())
    if not backgrounds:
        raise FileNotFoundError(f"No backgrounds found under {backgrounds_root}")
    return backgrounds


def resize_max(image: Image.Image, max_dim: int) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_dim:
        return image
    scale = max_dim / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def choose_split(stem: str, rng: np.random.Generator) -> str:
    # Keep all augments for a source in the same split by seeding on the stem.
    local_rng = np.random.default_rng(abs(hash(stem)) % (2**32))
    r = local_rng.random()
    if r < 0.8:
        return "train"
    if r < 0.9:
        return "val"
    return "test"


def find_source_raw(stem: str, raw_dir: Path) -> Path:
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = raw_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return raw_dir / f"{stem}.jpg"  # fallback for manifest even if missing


def contour_to_yolo_line(contour: np.ndarray, img_w: int, img_h: int, class_id: int) -> str:
    points: List[str] = []
    for x, y in contour.squeeze(axis=1):
        points.append(f"{float(x) / img_w:.6f} {float(y) / img_h:.6f}")
    return f"{class_id} " + " ".join(points)


def save_label_from_mask(mask_path: Path, label_path: Path, class_id: int) -> None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logging.warning("Failed to read mask %s", mask_path)
        return
    img_h, img_w = mask.shape[:2]
    bin_mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines: List[str] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        lines.append(contour_to_yolo_line(contour, img_w, img_h, class_id))

    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def jitter_cutout(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    brightness = float(rng.uniform(0.95, 1.05))
    contrast = float(rng.uniform(0.95, 1.05))
    color = float(rng.uniform(0.95, 1.05))
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(color)
    return img


def ensure_fit(obj_w: int, obj_h: int, bg_w: int, bg_h: int) -> float:
    scale = 1.0
    if obj_w >= bg_w or obj_h >= bg_h:
        scale = 0.9 * min(bg_w / obj_w, bg_h / obj_h)
    return scale


def generate_for_cutout(
    cls: str,
    cutout_path: Path,
    mask_path: Path,
    backgrounds: List[Path],
    raw_root: Path,
    out_root: Path,
    class_to_id: Dict[str, int],
    rng: np.random.Generator,
    num_augments: int,
    max_bg_dim: int,
    scale_range: Tuple[float, float],
    rotation_range: Tuple[float, float],
) -> List[SampleMeta]:
    cutout = Image.open(cutout_path).convert("RGBA")
    base_mask = Image.open(mask_path).convert("L")
    stem = cutout_path.stem.replace("_rgba", "")
    split = choose_split(stem, rng)
    samples: List[SampleMeta] = []

    for aug_idx in range(1, num_augments + 1):
        bg_path = backgrounds[int(rng.integers(0, len(backgrounds)))]
        background = resize_max(Image.open(bg_path).convert("RGB"), max_bg_dim)
        bg_w, bg_h = background.size

        # Random scale relative to background
        base_scale = float(rng.uniform(*scale_range))
        target_size = base_scale * min(bg_w, bg_h)
        cut_w, cut_h = cutout.size
        scale_factor = target_size / max(cut_w, cut_h)
        scaled_w, scaled_h = int(cut_w * scale_factor), int(cut_h * scale_factor)
        cut_scaled = cutout.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
        mask_scaled = base_mask.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

        # Rotation
        rot_deg = float(rng.uniform(*rotation_range))
        cut_rot = cut_scaled.rotate(rot_deg, expand=True)
        mask_rot = mask_scaled.rotate(rot_deg, expand=True)
        obj_w, obj_h = cut_rot.size

        # Ensure it fits
        fit_scale = ensure_fit(obj_w, obj_h, bg_w, bg_h)
        if fit_scale < 1.0:
            obj_w, obj_h = int(obj_w * fit_scale), int(obj_h * fit_scale)
            cut_rot = cut_rot.resize((obj_w, obj_h), Image.Resampling.LANCZOS)
            mask_rot = mask_rot.resize((obj_w, obj_h), Image.Resampling.LANCZOS)
            scale_factor *= fit_scale

        # Random placement
        max_x = max(bg_w - obj_w, 0)
        max_y = max(bg_h - obj_h, 0)
        offset_x = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
        offset_y = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0

        # Composite
        composite = background.copy()
        cut_jittered = jitter_cutout(cut_rot, rng)
        composite.paste(cut_jittered, (offset_x, offset_y), mask_rot)

        mask_canvas = Image.new("L", (bg_w, bg_h), 0)
        mask_canvas.paste(mask_rot, (offset_x, offset_y))

        # Paths
        out_img_dir = out_root / "images" / split
        out_mask_dir = out_root / "masks" / split
        out_lbl_dir = out_root / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_mask_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        out_stem = f"{cls}_{stem}_aug{aug_idx}"
        img_path = out_img_dir / f"{out_stem}.png"
        out_mask_path = out_mask_dir / f"{out_stem}.png"
        out_label_path = out_lbl_dir / f"{out_stem}.txt"

        composite.save(img_path)
        mask_canvas.save(out_mask_path)
        save_label_from_mask(out_mask_path, out_label_path, class_to_id[cls])

        samples.append(
            SampleMeta(
                synthetic_path=img_path,
                mask_path=out_mask_path,
                label_path=out_label_path,
                source_image=find_source_raw(stem, raw_root / cls),
                source_mask=mask_path,
                cutout_path=cutout_path,
                background=bg_path,
                split=split,
                scale_factor=round(scale_factor, 3),
                rotation_deg=round(rot_deg, 2),
                offset_x=offset_x,
                offset_y=offset_y,
            )
        )

    return samples


def write_manifest(rows: Iterable[SampleMeta], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "synthetic_path",
                "mask_path",
                "label_path",
                "source_image",
                "source_mask",
                "cutout_path",
                "background",
                "split",
                "scale_factor",
                "rotation_deg",
                "offset_x",
                "offset_y",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.synthetic_path.as_posix(),
                    row.mask_path.as_posix(),
                    row.label_path.as_posix(),
                    row.source_image.as_posix(),
                    row.source_mask.as_posix(),
                    row.cutout_path.as_posix(),
                    row.background.as_posix(),
                    row.split,
                    row.scale_factor,
                    row.rotation_deg,
                    row.offset_x,
                    row.offset_y,
                ]
            )


def build_yaml(out_root: Path, class_names: List[str]) -> None:
    yaml_path = out_root / "yolov8.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_root}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"names: {class_names}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic YOLOv8 segmentation dataset from cutouts.")
    parser.add_argument("--classes", nargs="+", default=["glass", "metal"])
    parser.add_argument("--cutouts-root", type=Path, default=Path("data/derived/rmbg_cutouts"))
    parser.add_argument("--masks-root", type=Path, default=Path("data/derived/rmbg_masks"))
    parser.add_argument("--backgrounds-root", type=Path, default=Path("data/raw/backgrounds"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/trashnet"))
    parser.add_argument("--out-root", type=Path, default=Path("data/derived/yolo_glass_metal"))
    parser.add_argument("--num-augments", type=int, default=5)
    parser.add_argument("--max-bg-dim", type=int, default=1024, help="Max background dimension before scaling down.")
    parser.add_argument("--scale-min", type=float, default=0.4)
    parser.add_argument("--scale-max", type=float, default=0.85)
    parser.add_argument("--rotation-max", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = np.random.default_rng(args.seed)

    backgrounds = list_backgrounds(args.backgrounds_root)
    class_to_id = {cls: idx for idx, cls in enumerate(args.classes)}

    all_rows: List[SampleMeta] = []
    total = 0
    for cls in args.classes:
        cutout_dir = args.cutouts_root / cls
        mask_dir = args.masks_root / cls
        cutouts = sorted(p for p in cutout_dir.glob("*_rgba.png") if p.is_file())
        logging.info("Class %s: %d cutouts", cls, len(cutouts))
        for idx, cutout_path in enumerate(cutouts, 1):
            stem = cutout_path.stem.replace("_rgba", "")
            mask_path = mask_dir / f"{stem}_mask.png"
            if not mask_path.exists():
                logging.warning("Missing mask for %s", cutout_path)
                continue
            rows = generate_for_cutout(
                cls,
                cutout_path,
                mask_path,
                backgrounds,
                args.raw_root,
                args.out_root,
                class_to_id,
                rng,
                args.num_augments,
                args.max_bg_dim,
                (args.scale_min, args.scale_max),
                (-args.rotation_max, args.rotation_max),
            )
            all_rows.extend(rows)
            total += len(rows)
            if idx % 100 == 0:
                logging.info("Processed %d cutouts (class %s)", idx, cls)

    write_manifest(all_rows, args.out_root / "manifest.csv")
    build_yaml(args.out_root, args.classes)
    logging.info("Done. Generated %d synthetic samples.", total)


if __name__ == "__main__":
    main()
