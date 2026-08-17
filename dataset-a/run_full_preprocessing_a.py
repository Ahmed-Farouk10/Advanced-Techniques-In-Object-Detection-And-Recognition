"""Full 7-Part Preprocessing & Data Cleaning Script for Dataset A (Testing Probe).
Implements the exact same rigorous cleaning, validation, format standardization,
and audit trail generation as Dataset B.
"""

import os
import glob
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageStat
import imagehash
from sklearn.cluster import KMeans

# 1. Paths
BASE_DIR = Path(__file__).resolve().parent
TEST_IMG_DIR = BASE_DIR / "dataset" / "test" / "images"
TEST_LBL_DIR = BASE_DIR / "dataset" / "test" / "labels"
PROCESSED_DIR = BASE_DIR / "processed"
TASK3_DIR = BASE_DIR / "preprocessing" / "task3_data_cleaning"
TASK2_DIR = BASE_DIR / "preprocessing" / "task2_data_understanding"
TASK4_DIR = BASE_DIR / "preprocessing" / "task4_data_transformation"
TASK6_DIR = BASE_DIR / "preprocessing" / "task6_feature_selection"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
(PROCESSED_DIR / "images").mkdir(parents=True, exist_ok=True)
(PROCESSED_DIR / "labels").mkdir(parents=True, exist_ok=True)
(PROCESSED_DIR / "annotations").mkdir(parents=True, exist_ok=True)

log_data = []  # [Step, Image, Issue Type, Detection Method, Treatment, Rationale]

print("=== Starting Dataset A Preprocessing Pipeline ===")

# Part 1: Missing Values & File Integrity Check
img_files = sorted(list(TEST_IMG_DIR.glob("*.jpg")) + list(TEST_IMG_DIR.glob("*.png")))
lbl_files = sorted(list(TEST_LBL_DIR.glob("*.txt")))

img_stems = {f.stem: f for f in img_files}
lbl_stems = {f.stem: f for f in lbl_files}

print(f"Total Test Images found: {len(img_files)}")
print(f"Total Test Labels found: {len(lbl_files)}")

valid_stems = []
for stem in sorted(set(img_stems.keys()).union(set(lbl_stems.keys()))):
    img_path = img_stems.get(stem)
    lbl_path = lbl_stems.get(stem)
    
    if not img_path:
        log_data.append(["Part 1", stem, "Missing Image", "File existence check", "Flagged label", "Unpaired annotation"])
        continue
    if not lbl_path:
        # Background image (0 annotations)
        log_data.append(["Part 1", stem, "Background Image (No Label File)", "File existence check", "Retained as hard negative", "Legitimate negative"])
        valid_stems.append(stem)
        continue
        
    # Verify image corruptness
    try:
        with Image.open(img_path) as img:
            img.verify()
        valid_stems.append(stem)
    except Exception as e:
        log_data.append(["Part 1", stem, "Corrupt Image", f"PIL.verify: {e}", "Flagged", "Unreadable file"])

print(f"Valid verified image stems: {len(valid_stems)}")

# Part 2: MD5 Exact Duplicate Check
md5_dict = {}
for stem in valid_stems:
    img_path = img_stems[stem]
    with open(img_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    if file_hash in md5_dict:
        log_data.append(["Part 2", stem, "Exact Duplicate", f"MD5 match with {md5_dict[file_hash]}", "Flagged/Retained", "Documented duplicate"])
    else:
        md5_dict[file_hash] = stem

# Part 3: Image-level & Box-level Outlier Analysis
blur_scores = []
box_areas = []
aspect_ratios = []
class_counts = {0: 0, 1: 0} # 0: fire, 1: smoke

for stem in valid_stems:
    img_path = img_stems[stem]
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]
    
    # Laplacian variance (blur)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_scores.append(blur)
    if blur < 100.0:
        log_data.append(["Part 3", stem, "Blurry Image", f"Laplacian var = {blur:.1f} (<100)", "Flagged", "Motion/defocus blur"])
        
    lbl_path = lbl_stems.get(stem)
    if lbl_path and lbl_path.exists():
        with open(lbl_path, "r") as f:
            lines = [line.strip().split() for line in f.readlines() if line.strip()]
        for line in lines:
            if len(line) >= 5:
                cls_id = int(line[0])
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                bx, by, bw, bh = map(float, line[1:5])
                area = bw * bh
                ar = bw / max(bh, 1e-6)
                box_areas.append(area)
                aspect_ratios.append(ar)
                
                # Part 4: Coordinate clipping check
                if bx < 0 or by < 0 or bx > 1 or by > 1 or bw <= 0 or bh <= 0 or (bx + bw/2) > 1.05 or (by + bh/2) > 1.05:
                    log_data.append(["Part 4", stem, "Out-of-bounds Coordinate", f"Box: {line[1:5]}", "Clipped to [0,1]", "YOLO standard normalization"])

# Area percentiles
p05 = np.percentile(box_areas, 0.5) if box_areas else 0
p995 = np.percentile(box_areas, 99.5) if box_areas else 1
for stem in valid_stems:
    lbl_path = lbl_stems.get(stem)
    if lbl_path and lbl_path.exists():
        with open(lbl_path, "r") as f:
            lines = [line.strip().split() for line in f.readlines() if line.strip()]
        for line in lines:
            if len(line) >= 5:
                bw, bh = float(line[3]), float(line[4])
                area = bw * bh
                if area < p05:
                    log_data.append(["Part 3", stem, "Micro Box Outlier", f"Area {area:.5f} < {p05:.5f}", "Flagged", "Degenerate tiny box"])
                elif area > p995:
                    log_data.append(["Part 3", stem, "Massive Box Outlier", f"Area {area:.5f} > {p995:.5f}", "Flagged", "Full frame coverage"])

# Part 5: Perceptual Hash (pHash) Near-Duplicates
phash_dict = {}
for stem in valid_stems:
    img_path = img_stems[stem]
    try:
        with Image.open(img_path) as img:
            ph = imagehash.phash(img)
        for other_ph, other_stem in phash_dict.items():
            if ph - other_ph < 4:
                log_data.append(["Part 5", stem, "Near Duplicate Frame", f"pHash diff {ph - other_ph} with {other_stem}", "Flagged", "Redundant frame"])
                break
        phash_dict[ph] = stem
    except Exception:
        pass

# Save Audit Log
df_log = pd.DataFrame(log_data, columns=["Step", "Image", "Issue Type", "Detection Method", "Treatment", "Rationale"])
df_log.to_csv(TASK3_DIR / "cleaning_log.csv", index=False)
print(f"Cleaning log saved: {len(df_log)} entries recorded in {TASK3_DIR / 'cleaning_log.csv'}")

# Part 6: Standardize Images to 640x640 & Generate COCO JSON
TARGET_SIZE = (640, 640)
coco_images = []
coco_annotations = []
coco_categories = [
    {"id": 0, "name": "fire", "supercategory": "disaster"},
    {"id": 1, "name": "smoke", "supercategory": "disaster"}
]
ann_id = 1

for idx, stem in enumerate(valid_stems, start=1):
    img_path = img_stems[stem]
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    orig_h, orig_w = img.shape[:2]
    
    # Resize to 640x640
    resized_img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    out_img_path = PROCESSED_DIR / "images" / f"{stem}.jpg"
    cv2.imwrite(str(out_img_path), resized_img)
    
    coco_images.append({
        "id": idx,
        "file_name": f"{stem}.jpg",
        "width": TARGET_SIZE[0],
        "height": TARGET_SIZE[1],
        "original_width": orig_w,
        "original_height": orig_h
    })
    
    # Process labels
    lbl_path = lbl_stems.get(stem)
    out_lbl_path = PROCESSED_DIR / "labels" / f"{stem}.txt"
    if lbl_path and lbl_path.exists():
        with open(lbl_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        
        valid_lines = []
        for l in lines:
            parts = l.split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                bx, by, bw, bh = map(float, parts[1:5])
                
                # Clip to [0, 1]
                bx = min(max(bx, 0.0), 1.0)
                by = min(max(by, 0.0), 1.0)
                bw = min(max(bw, 0.001), 1.0)
                bh = min(max(bh, 0.001), 1.0)
                
                valid_lines.append(f"{cls_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}")
                
                # COCO format: [x_min, y_min, width, height] in pixel coordinates
                abs_x = (bx - bw / 2) * TARGET_SIZE[0]
                abs_y = (by - bh / 2) * TARGET_SIZE[1]
                abs_w = bw * TARGET_SIZE[0]
                abs_h = bh * TARGET_SIZE[1]
                
                coco_annotations.append({
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": cls_id,
                    "bbox": [round(abs_x, 2), round(abs_y, 2), round(abs_w, 2), round(abs_h, 2)],
                    "area": round(abs_w * abs_h, 2),
                    "iscrowd": 0
                })
                ann_id += 1
                
        with open(out_lbl_path, "w") as f:
            f.write("\n".join(valid_lines))
    else:
        out_lbl_path.write_text("")

coco_data = {
    "images": coco_images,
    "annotations": coco_annotations,
    "categories": coco_categories
}

with open(PROCESSED_DIR / "annotations" / "test_coco.json", "w") as f:
    json.dump(coco_data, f, indent=2)

print(f"Standardized {len(coco_images)} images to 640x640.")
print(f"Generated COCO test annotations with {len(coco_annotations)} boxes.")

# Part 7: K-Means Anchor Prior Clustering (k=5)
if box_areas:
    box_wh = []
    for stem in valid_stems:
        lbl_path = lbl_stems.get(stem)
        if lbl_path and lbl_path.exists():
            with open(lbl_path, "r") as f:
                lines = [l.strip().split() for l in f.readlines() if l.strip()]
            for l in lines:
                if len(l) >= 5:
                    box_wh.append([float(l[3]), float(l[4])])
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(box_wh)
    anchors = kmeans.cluster_centers_[np.argsort(kmeans.cluster_centers_[:, 0] * kmeans.cluster_centers_[:, 1])]
    print("\nDataset A K-Means Anchor Priors (k=5):")
    for i, (w, h) in enumerate(anchors, start=1):
        print(f"  Anchor {i}: Width={w:.4f}, Height={h:.4f}, Area={w*h:.4f}")

print("\n=== Dataset A Preprocessing Completed Successfully ===")
