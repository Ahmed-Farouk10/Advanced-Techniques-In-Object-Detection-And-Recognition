import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# -----------------
# Cell 1: Markdown (Title)
# -----------------
title_md = """# Task 4: Video-Aware Data Splitting & Optimization

**Project:** Cognitive Fire Defense Pipeline — AIN7601  
**Goal:** Mathematically optimal 70/15/15 clip-level split to prevent temporal leakage, while balancing data distributions (blur, brightness, box area) across splits.
"""

# -----------------
# Cell 2: Code (Imports & Config)
# -----------------
imports_code = """import os
import glob
import random
import shutil
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

# Config
DATA_B_RAW = r"d:\\Masters Academy AI\\Advanced Techniques in Object Detection and Recognition\\Research Paper\\dataset-b\\raw\\Boreal-Forest-Fire-Subset-A"
YOLO_DIR = r"d:\\Masters Academy AI\\Advanced Techniques in Object Detection and Recognition\\Research Paper\\dataset-b\\yolo_format"
MAX_FRAMES_PER_CLIP = 300
OPTIMIZATION_ITERATIONS = 10000

locations = ["Evo", "Heinola", "Karkkila", "Ruokolahti"]

# Ensure YOLO dirs exist
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(YOLO_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DIR, "labels", split), exist_ok=True)
"""

# -----------------
# Cell 3: Markdown (Clip Discovery)
# -----------------
discovery_md = """## 1. Discover Clips & Empty Images
We will group images by their video prefix."""

# -----------------
# Cell 4: Code (Clip Discovery)
# -----------------
discovery_code = """all_images = []
for loc in locations:
    img_dir = os.path.join(DATA_B_RAW, f"{loc}-Images")
    if os.path.exists(img_dir):
        all_images.extend(glob.glob(os.path.join(img_dir, "*.jpg")))

empty_dir = os.path.join(DATA_B_RAW, "Empty-Images")
empty_images = glob.glob(os.path.join(empty_dir, "*.jpg")) if os.path.exists(empty_dir) else []

print(f"Total Images: {len(all_images)}")
print(f"Total Empty Images: {len(empty_images)}")

# Group by sequence ID
clips = defaultdict(list)
for img in all_images:
    filename = os.path.basename(img)
    seq_id = filename.split("_frame")[0]
    clips[seq_id].append(img)

print(f"Total Unique Clips: {len(clips)}")
"""

# -----------------
# Cell 5: Markdown (Clip Feature Extraction)
# -----------------
features_md = """## 2. Fast Clip Feature Extraction
To balance the splits, we need to know the representative properties of each clip. To avoid 4K processing bottlenecks, we randomly sample up to 10 frames per clip to estimate its Mean Brightness, Mean Blur, Mean Box Area, and Small Plume %.
"""

# -----------------
# Cell 6: Code (Features)
# -----------------
features_code = """def extract_clip_features(img_paths):
    sample_paths = random.sample(img_paths, min(10, len(img_paths)))
    brightness = []
    blur = []
    
    # Process Blur/Brightness on sample paths only (for speed)
    for img_p in sample_paths:
        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            small_img = cv2.resize(img, (512, 512))
            brightness.append(np.mean(small_img))
            blur.append(cv2.Laplacian(small_img, cv2.CV_64F).var())
            
    # Process labels on ALL paths in the clip to find rare small plumes
    box_areas = []
    small_plumes = 0
    total_boxes = 0
    for img_p in img_paths:
        lbl_p = img_p.replace("Images", "Labels").replace(".jpg", ".txt")
        if os.path.exists(lbl_p):
            with open(lbl_p, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, _, _, w, h = map(float, parts)
                        area = w * h
                        box_areas.append(area)
                        if area < 0.01:
                            small_plumes += 1
                        total_boxes += 1
                        
    return {
        "count": min(len(img_paths), MAX_FRAMES_PER_CLIP),  # We will cap sampling later
        "brightness": np.mean(brightness) if brightness else 0,
        "blur": np.mean(blur) if blur else 0,
        "box_area": np.mean(box_areas) if box_areas else 0,
        "small_plume_ratio": small_plumes / total_boxes if total_boxes > 0 else 0,
        "location": img_paths[0].split("\\\\")[-1].split("_")[0]  # e.g., 'evoDJI' -> 'evo'
    }

print("Extracting features for 31 clips...")
clip_data = {}
for seq_id, paths in tqdm(clips.items()):
    clip_data[seq_id] = extract_clip_features(paths)
"""

# -----------------
# Cell 7: Markdown (Optimization)
# -----------------
opt_md = """## 3. Stochastic Optimization Packing
We run a randomized assignment 5,000 times to find the split that minimizes deviation from the 70/15/15 ratio AND minimizes the variance of Brightness, Blur, Box Area, and Small Plumes across Train/Val/Test.
"""

# -----------------
# Cell 8: Code (Optimization)
# -----------------
opt_code = """best_score = float('inf')
best_split = None

# Calculate global targets
total_frames = sum(c["count"] for c in clip_data.values())
target_train = total_frames * 0.70
target_val = total_frames * 0.15
target_test = total_frames * 0.15

for _ in range(OPTIMIZATION_ITERATIONS):
    # Randomly assign clips to splits
    assignment = {seq: random.choice(["train", "val", "test"]) for seq in clip_data.keys()}
    
    # Stratification Check: Ensure all locations exist in all splits
    loc_splits = {"train": set(), "val": set(), "test": set()}
    for seq, split in assignment.items():
        loc_splits[split].add(clip_data[seq]["location"][:3]) # e.g. 'evo', 'hei', 'kar', 'ruo'
        
    if any(len(locs) < 4 for locs in loc_splits.values()):
        continue  # Fails stratification constraint
        
    # Calculate counts
    counts = {"train": 0, "val": 0, "test": 0}
    for seq, split in assignment.items():
        counts[split] += clip_data[seq]["count"]
        
    # If any split is empty (highly unlikely but possible), skip
    if counts["train"] == 0 or counts["val"] == 0 or counts["test"] == 0:
        continue
        
    # Ratio Penalty (MSE)
    ratio_penalty = (abs(counts["train"]/total_frames - 0.70) + 
                     abs(counts["val"]/total_frames - 0.15) + 
                     abs(counts["test"]/total_frames - 0.15))
    
    # Distribution Penalty
    metrics = {"train": [], "val": [], "test": []}
    for seq, split in assignment.items():
        metrics[split].append(clip_data[seq])
        
    dist_penalty = 0
    for key in ["brightness", "blur", "box_area", "small_plume_ratio"]:
        vals = []
        for split in ["train", "val", "test"]:
            if metrics[split]:
                # Weighted average for the split
                split_val = sum(c[key] * c["count"] for c in metrics[split]) / counts[split]
                vals.append(split_val)
            else:
                vals.append(0)
        # Variance normalization
        mean_val = np.mean(vals)
        if mean_val > 0:
            dist_penalty += np.var(vals) / (mean_val ** 2)
        else:
            dist_penalty += np.var(vals)
            
    # Add Small Plume Existence Penalty
    for split in ["train", "val", "test"]:
        if sum(c["small_plume_ratio"] for c in metrics[split]) == 0:
            dist_penalty += 100000.0

    # Add Explicit Blur Penalty
    blur_vals = []
    for split in ["train", "val", "test"]:
        if metrics[split]:
            split_blur = sum(c["blur"] * c["count"] for c in metrics[split]) / counts[split]
            blur_vals.append(split_blur)
        else:
            blur_vals.append(0)
    dist_penalty += np.std(blur_vals) * 10.0

    score = (ratio_penalty * 100000) + dist_penalty
    
    if score < best_score:
        best_score = score
        best_split = assignment
        
print(f"Optimal Split Found! Score: {best_score:.4f}")
"""

# -----------------
# Cell 9: Markdown (Execution)
# -----------------
exec_md = """## 4. Execute Split & Apply Sampling
We will now copy the files, uniformly sampling frames for clips that exceed `MAX_FRAMES_PER_CLIP`."""

# -----------------
# Cell 10: Code (Execution)
# -----------------
exec_code = """final_metrics = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
total_copied = {"train": 0, "val": 0, "test": 0}

def copy_pair(img_src, split):
    lbl_src = img_src.replace("Images", "Labels").replace(".jpg", ".txt")
    
    # Standardize names safely
    base = os.path.basename(img_src)
    img_dst = os.path.join(YOLO_DIR, "images", split, base)
    lbl_dst = os.path.join(YOLO_DIR, "labels", split, base.replace(".jpg", ".txt"))
    
    shutil.copy2(img_src, img_dst)
    if os.path.exists(lbl_src):
        shutil.copy2(lbl_src, lbl_dst)
        
    total_copied[split] += 1

# Process Clips
for seq_id, split in best_split.items():
    paths = sorted(clips[seq_id])
    
    # Uniform linear sampling if exceeding max frames
    if len(paths) > MAX_FRAMES_PER_CLIP:
        indices = np.linspace(0, len(paths)-1, MAX_FRAMES_PER_CLIP, dtype=int)
        sampled_paths = [paths[i] for i in indices]
    else:
        sampled_paths = paths
        
    # Append to final metric calculations
    cd = clip_data[seq_id]
    for _ in range(len(sampled_paths)):
        final_metrics[split]["brightness"].append(cd["brightness"])
        final_metrics[split]["blur"].append(cd["blur"])
        final_metrics[split]["box_area"].append(cd["box_area"])
        final_metrics[split]["small_plume"].append(cd["small_plume_ratio"])
        
    for p in sampled_paths:
        copy_pair(p, split)

# Process Empty Images (Random Assignment at 70/15/15)
random.shuffle(empty_images)
num_empty = len(empty_images)
train_end = int(num_empty * 0.70)
val_end = train_end + int(num_empty * 0.15)

for i, p in enumerate(empty_images):
    if i < train_end:
        copy_pair(p, "train")
    elif i < val_end:
        copy_pair(p, "val")
    else:
        copy_pair(p, "test")

print("Files successfully copied to YOLO format!")
"""

# -----------------
# Cell 11: Markdown (Paper Table)
# -----------------
table_md = """## 5. Academically Defensible Distribution Report"""

# -----------------
# Cell 12: Code (Table)
# -----------------
table_code = """from IPython.display import display, Markdown

# Calculate aggregates
data = []
for split in ["train", "val", "test"]:
    count = total_copied[split]
    ratio = count / sum(total_copied.values()) * 100
    
    b_mean = np.mean(final_metrics[split]["brightness"]) if final_metrics[split]["brightness"] else 0
    bl_mean = np.mean(final_metrics[split]["blur"]) if final_metrics[split]["blur"] else 0
    a_mean = np.mean(final_metrics[split]["box_area"]) if final_metrics[split]["box_area"] else 0
    sp_mean = np.mean(final_metrics[split]["small_plume"]) * 100 if final_metrics[split]["small_plume"] else 0
    
    data.append({
        "Split": split.capitalize(),
        "Images": count,
        "Ratio %": f"{ratio:.1f}%",
        "Mean Brightness": f"{b_mean:.1f}",
        "Mean Blur": f"{bl_mean:.1f}",
        "Mean Box Area": f"{a_mean:.3f}",
        "Small Plume %": f"{sp_mean:.1f}%"
    })

df_report = pd.DataFrame(data)

# Print as Markdown Table for the paper
md_table = df_report.to_markdown(index=False)
display(Markdown("### Distribution Metrics Across Splits"))
display(Markdown(md_table))

print("\\n" + md_table)
"""

# Assemble notebook
nb['cells'] = [
    nbf.v4.new_markdown_cell(title_md),
    nbf.v4.new_code_cell(imports_code),
    nbf.v4.new_markdown_cell(discovery_md),
    nbf.v4.new_code_cell(discovery_code),
    nbf.v4.new_markdown_cell(features_md),
    nbf.v4.new_code_cell(features_code),
    nbf.v4.new_markdown_cell(opt_md),
    nbf.v4.new_code_cell(opt_code),
    nbf.v4.new_markdown_cell(exec_md),
    nbf.v4.new_code_cell(exec_code),
    nbf.v4.new_markdown_cell(table_md),
    nbf.v4.new_code_cell(table_code)
]

os.makedirs(r"d:\Masters Academy AI\Advanced Techniques in Object Detection and Recognition\Research Paper\dataset-b\preprocessing\task4_data_splitting", exist_ok=True)
output_path = r"d:\Masters Academy AI\Advanced Techniques in Object Detection and Recognition\Research Paper\dataset-b\preprocessing\task4_data_splitting\split.ipynb"

# Write the notebook
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("split.ipynb generated successfully.")
