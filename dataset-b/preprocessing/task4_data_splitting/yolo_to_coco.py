"""Regenerate COCO format from the corrected 80/20 clip-level YOLO split.

Source of truth: dataset-b/yolo_format/{images,labels}/{train,val}
Output:         dataset-b/coco_format/{images,annotations}/{train,val}

YOLO label (normalized): cls cx cy w h  ->  COCO bbox (pixel): x y w h
"""

import json
import shutil
from pathlib import Path

from PIL import Image

BASE = Path(__file__).resolve().parent.parent.parent  # dataset-b
YOLO_DIR = BASE / "yolo_format"
COCO_DIR = BASE / "coco_format"

SPLITS = ["train", "val"]
CATEGORIES = [{"id": 0, "name": "smoke"}]


def parse_yolo(txt_path: Path, width: int, height: int):
    """Return list of COCO-style [x, y, w, h] pixel bboxes."""
    boxes = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = (float(p) for p in parts)
        if w <= 0 or h <= 0:
            continue
        x = int((cx - w / 2) * width)
        y = int((cy - h / 2) * height)
        w_px, h_px = int(w * width), int(h * height)
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        boxes.append([x, y, w_px, h_px])
    return boxes


def build_split(split: str):
    img_src = YOLO_DIR / "images" / split
    lbl_src = YOLO_DIR / "labels" / split
    img_dst = COCO_DIR / "images" / split
    ann_dst = COCO_DIR / "annotations"

    img_dst.mkdir(parents=True, exist_ok=True)
    ann_dst.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    ann_id = 1

    for img_path in sorted(img_src.glob("*.jpg")):
        with Image.open(img_path) as im:
            width, height = im.size

        img_id = len(images) + 1
        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        })

        lbl_path = lbl_src / (img_path.stem + ".txt")
        for box in parse_yolo(lbl_path, width, height):
            x, y, w, h = box
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 0,
                "bbox": box,
                "area": float(w * h),
                "iscrowd": 0,
            })
            ann_id += 1

        shutil.copy2(img_path, img_dst / img_path.name)

    coco = {
        "info": {"description": "Cognitive Fire Defense - 80/20 clip-level split", "version": "1.0"},
        "licenses": [],
        "categories": CATEGORIES,
        "images": images,
        "annotations": annotations,
    }

    out = ann_dst / f"{split}.json"
    out.write_text(json.dumps(coco), encoding="utf-8")
    return len(images), len(annotations)


def main():
    if COCO_DIR.exists():
        shutil.rmtree(COCO_DIR)
    COCO_DIR.mkdir(parents=True)

    empty = 0
    for split in SPLITS:
        n_img, n_ann = build_split(split)
        print(f"{split}: images={n_img} annotations={n_ann}")


if __name__ == "__main__":
    main()