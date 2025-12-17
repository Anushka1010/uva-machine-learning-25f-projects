"""
Regenerate YOLOv8 segmentation labels from binary masks.

Assumptions:
- Single class: 0 ("plastic")
- Masks are stored in data/derived/yolo_plastic/masks/<split>/*.png
- Images live in data/derived/yolo_plastic/images/<split>/ with matching stems.

The script binarizes masks (any pixel > 0 becomes 1), extracts external contours,
and writes YOLO-style polygon labels to the corresponding labels/<split>/*.txt.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np


def find_image_for_stem(stem: str, split_dir: Path) -> Path | None:
    """Return the first matching image path with the given stem."""
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = split_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def contour_to_yolo_line(contour: np.ndarray, img_w: int, img_h: int, class_id: int) -> str:
    """Convert a contour to a YOLO segmentation line."""
    points: List[Tuple[float, float]] = []
    for x, y in contour.squeeze(axis=1):
        points.append((float(x) / img_w, float(y) / img_h))
    flat: Iterable[str] = (f"{p[0]:.6f} {p[1]:.6f}" for p in points
    return f"{class_id} " + " ".join(flat)


def process_mask(mask_path: Path, images_dir: Path, labels_dir: Path, class_id: int) -> Tuple[int, int]:
    """Generate labels for a single mask file. Returns (num_contours_written, num_points_total)."""
    stem = mask_path.stem
    img_path = find_image_for_stem(stem, images_dir)
    if not img_path:
        logging.warning("No matching image for mask %s", mask_path)
        return 0, 0

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logging.warning("Failed to read mask %s", mask_path)
        return 0, 0

    img_h, img_w = mask.shape[:2]
    # Binarize: any non-zero becomes 1
    bin_mask = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: List[str] = []
    points_total = 0
    for contour in contours:
        if len(contour) < 3:
            continue
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        line = contour_to_yolo_line(contour, img_w, img_h, class_id)
        lines.append(line)
        points_total += len(contour)

    label_path = labels_dir / f"{stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return len(lines), points_total


def regenerate(base_dir: Path, class_id: int = 0) -> None:
    splits = ["train", "val", "test"]
    summary = {}
    for split in splits:
        masks_dir = base_dir / "masks" / split
        images_dir = base_dir / "images" / split
        labels_dir = base_dir / "labels" / split

        mask_files = sorted(masks_dir.glob("*.png"))
        written_files = 0
        total_contours = 0
        total_points = 0
        for mask_path in mask_files:
            contours_written, points = process_mask(mask_path, images_dir, labels_dir, class_id)
            total_contours += contours_written
            total_points += points
            if contours_written > 0 or points > 0:
                written_files += 1

        summary[split] = {
            "mask_files": len(mask_files),
            "label_files_written": written_files,
            "contours": total_contours,
            "points": total_points,
        }

    logging.info("Regeneration summary: %s", summary)
    print("Regeneration summary:")
    for split, stats in summary.items():
        print(
            f"  {split}: masks={stats['mask_files']}, "
            f"labels_with_content={stats['label_files_written']}, "
            f"contours={stats['contours']}, points={stats['points']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate YOLO segmentation labels from masks.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/derived/yolo_plastic"),
        help="Root directory containing masks/, images/, labels/ subfolders.",
    )
    parser.add_argument("--class-id", type=int, default=0, help="Class id to assign to all masks.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    regenerate(args.base_dir, args.class_id)


if __name__ == "__main__":
    main()
