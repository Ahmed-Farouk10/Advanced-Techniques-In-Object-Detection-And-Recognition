"""Reproduce Esraa's 80/20 clip-level split (with empty-image handling fixed to 80/20).

Faithful to Esraa's split.ipynb algorithm:
- Group annotated clips by video sequence ID
- Clip features: brightness/blur (10-image sample), box area/small plume (all frames)
- Constraint optimization over 10,000 random assignments
- Frame sampling cap of 300 per clip
- Location stratification (all 4 locations in both splits)
"""
import os
import glob
import random
import shutil
import cv2
import numpy as np
from collections import defaultdict

random.seed(42)
np.random.seed(42)

DATA_B = r"dataset-b\raw\Boreal-Forest-Fire-Subset-A"
YOLO_DIR = r"dataset-b\yolo_format"

MAX_FRAMES_PER_CLIP = 300
OPTIMIZATION_ITERATIONS = 10000
TRAIN_RATIO = 0.80
VAL_RATIO = 0.20

locations = ["Evo", "Heinola", "Karkkila", "Ruokolahti"]

# ---- Wipe and recreate yolo_format (train/val only) ----
if os.path.exists(YOLO_DIR):
    shutil.rmtree(YOLO_DIR)
for split in ["train", "val"]:
    os.makedirs(os.path.join(YOLO_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DIR, "labels", split), exist_ok=True)

# ---- Load annotated images and group by clip ----
all_images = []
for loc in locations:
    img_dir = os.path.join(DATA_B, f"{loc}-Images")
    if os.path.exists(img_dir):
        imgs = glob.glob(os.path.join(img_dir, "*.jpg"))
        all_images.extend(imgs)
        print(f"{loc}: {len(imgs)} images")

clips = defaultdict(list)
for img_path in all_images:
    filename = os.path.basename(img_path)
    if "_frame" not in filename:
        continue
    seq_id = filename.split("_frame")[0]
    clips[seq_id].append(img_path)

for seq_id in clips:
    clips[seq_id].sort(key=lambda x: int(os.path.basename(x).split("_frame")[1].split(".")[0]))

print(f"Total clips: {len(clips)}")

# ---- Feature extraction (Esraa's method) ----
def extract_clip_features(img_paths):
    sample_paths = random.sample(img_paths, min(10, len(img_paths)))
    brightness = []
    blur = []
    for img_p in sample_paths:
        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            small = cv2.resize(img, (512, 512))
            brightness.append(np.mean(small))
            blur.append(cv2.Laplacian(small, cv2.CV_64F).var())

    box_areas = []
    small_plumes = 0
    total_boxes = 0
    for img_p in img_paths:
        lbl_p = img_p.replace("-Images", "-Labels").replace(".jpg", ".txt")
        if not os.path.exists(lbl_p):
            continue
        with open(lbl_p, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, _, _, w, h = map(float, parts)
                area = w * h
                box_areas.append(area)
                if area < 0.01:
                    small_plumes += 1
                total_boxes += 1

    first = os.path.basename(img_paths[0]).split("_")[0].lower()
    if first.startswith("evo"):
        location = "Evo"
    elif first.startswith("heinola"):
        location = "Heinola"
    elif first.startswith("karkkila"):
        location = "Karkkila"
    elif first.startswith("ruokolahti"):
        location = "Ruokolahti"
    else:
        location = "Unknown"

    return {
        "count": min(len(img_paths), MAX_FRAMES_PER_CLIP),
        "brightness": np.mean(brightness) if brightness else 0,
        "blur": np.mean(blur) if blur else 0,
        "box_area": np.mean(box_areas) if box_areas else 0,
        "small_plume_ratio": (small_plumes / total_boxes) if total_boxes > 0 else 0,
        "location": location,
    }

clip_data = {}
for seq_id, paths in clips.items():
    clip_data[seq_id] = extract_clip_features(paths)
print(f"Feature extraction done: {len(clip_data)} clips")

# ---- Constraint optimization (Esraa's scoring) ----
splits = ["train", "val"]
total_frames = sum(c["count"] for c in clip_data.values())
target_train = total_frames * TRAIN_RATIO
target_val = total_frames * VAL_RATIO
print(f"Total optimization frames: {total_frames}, target train {target_train:.0f}, val {target_val:.0f}")

best_score = float("inf")
best_split = None

for _ in range(OPTIMIZATION_ITERATIONS):
    assignment = {seq_id: random.choice(splits) for seq_id in clip_data}

    loc_splits = {"train": set(), "val": set()}
    for seq_id, split in assignment.items():
        loc_splits[split].add(clip_data[seq_id]["location"])
    if any(len(loc_splits[split]) < len(locations) for split in splits):
        continue

    counts = {"train": 0, "val": 0}
    metrics = {"train": [], "val": []}
    for seq_id, split in assignment.items():
        c = clip_data[seq_id]
        counts[split] += c["count"]
        metrics[split].append(c)

    if counts["train"] == 0 or counts["val"] == 0:
        continue

    ratio_penalty = (abs(counts["train"] / total_frames - TRAIN_RATIO)
                     + abs(counts["val"] / total_frames - VAL_RATIO))

    dist_penalty = 0
    for key in ["brightness", "blur", "box_area", "small_plume_ratio"]:
        vals = []
        for split in splits:
            split_val = (sum(c[key] * c["count"] for c in metrics[split]) / counts[split])
            vals.append(split_val)
        mean_val = np.mean(vals)
        if mean_val > 0:
            dist_penalty += np.var(vals) / (mean_val ** 2)
        else:
            dist_penalty += np.var(vals)

    for split in splits:
        small_plume_clips = sum(c["small_plume_ratio"] > 0 for c in metrics[split])
        if small_plume_clips == 0:
            dist_penalty += 100000.0

    blur_vals = []
    for split in splits:
        split_blur = sum(c["blur"] * c["count"] for c in metrics[split]) / counts[split]
        blur_vals.append(split_blur)
    dist_penalty += np.std(blur_vals) * 10.0

    score = ratio_penalty * 100000 + dist_penalty
    if score < best_score:
        best_score = score
        best_split = assignment.copy()

print(f"Best score: {best_score:.4f}")

# ---- Copy annotated clips (uniform sampling) ----
total_copied = {"train": 0, "val": 0}

def copy_pair(img_src, split):
    lbl_src = img_src.replace("-Images", "-Labels").replace(".jpg", ".txt")
    base = os.path.basename(img_src)
    img_dst = os.path.join(YOLO_DIR, "images", split, base)
    lbl_dst = os.path.join(YOLO_DIR, "labels", split, base.replace(".jpg", ".txt"))
    shutil.copy2(img_src, img_dst)
    if os.path.exists(lbl_src):
        shutil.copy2(lbl_src, lbl_dst)
    total_copied[split] += 1

for seq_id, split in best_split.items():
    paths = sorted(clips[seq_id])
    if len(paths) > MAX_FRAMES_PER_CLIP:
        indices = np.linspace(0, len(paths) - 1, MAX_FRAMES_PER_CLIP, dtype=int)
        sampled_paths = [paths[i] for i in indices]
    else:
        sampled_paths = paths
    for p in sampled_paths:
        copy_pair(p, split)

# ---- Empty images: distributed 80/20 (fixed from Esraa's 70/15/15 leftover) ----
empty_dir = os.path.join(DATA_B, "Empty-Images")
empty_images = sorted(glob.glob(os.path.join(empty_dir, "*.jpg"))) if os.path.exists(empty_dir) else []
random.shuffle(empty_images)
train_end = int(len(empty_images) * TRAIN_RATIO)
for i, p in enumerate(empty_images):
    copy_pair(p, "train" if i < train_end else "val")

print("\n=== FINAL SPLIT ===")
for split in ["train", "val"]:
    img_count = len(glob.glob(os.path.join(YOLO_DIR, "images", split, "*.jpg")))
    lbl_count = len(glob.glob(os.path.join(YOLO_DIR, "labels", split, "*.txt")))
    print(f"{split}: {img_count} images, {lbl_count} labels")

total = sum(len(glob.glob(os.path.join(YOLO_DIR, "images", s, "*.jpg"))) for s in ["train", "val"])
print(f"Total: {total} images")
print("\nSplit complete.")
