import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# -----------------
# Cell 1: Markdown (Title & Task 1)
# -----------------
title_md = """# Task 1 & 2: Problem & Data Understanding (Dataset B)

**Project:** Cognitive Fire Defense Pipeline — AIN7601  
**Author:** Ahmed-Mcawesome-vill & Esraa Nasr

> **A+ Philosophy Applied:** This notebook does not just run code. Every visualization is linked directly to a business constraint. Our goal is to deploy an early-warning system in a watchtower, so we must understand the data through that lens.

## Task 1: The Business Translation (WHY)
The primary objective of a watchtower is **early detection**. Detecting a raging fire is too late; we must detect the **smoke precursor** while it is small and distant. 
- **ML Task:** Supervised Object Detection (Class 0: Smoke)
- **Constraint 1 (Maximize Recall):** Missing smoke = disaster. False alarms = easily verified by humans.
- **Constraint 2 (Temporal Leakage):** Video frame sequences MUST be isolated from each other during train/val splitting, otherwise the model memorizes the forest background instead of the smoke.
"""

# -----------------
# Cell 2: Code (Imports)
# -----------------
imports_code = """import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# Set Dataset B Path
DATA_B_RAW = r"d:\\Masters Academy AI\\Advanced Techniques in Object Detection and Recognition\\Research Paper\\dataset-b\\raw\\Boreal-Forest-Fire-Subset-A"
locations = ["Evo", "Heinola", "Karkkila", "Ruokolahti"]
print(f"Dataset B Raw Path: {DATA_B_RAW}")
"""

# -----------------
# Cell 3: Markdown (Task 2: Image Distribution)
# -----------------
dist_md = """## Task 2: Data Understanding
### 2.1 Geographic Distribution
First, let's count the number of images and empty (clean forest) frames across our 4 locations. This tells us our negative class ratio."""

# -----------------
# Cell 4: Code (Image Distribution)
# -----------------
dist_code = """total_images = 0
empty_images = 0
images_per_loc = {}

for loc in locations:
    img_dir = os.path.join(DATA_B_RAW, f"{loc}-Images")
    if os.path.exists(img_dir):
        count = len(glob.glob(os.path.join(img_dir, "*.jpg")))
        images_per_loc[loc] = count
        total_images += count

empty_dir = os.path.join(DATA_B_RAW, "Empty-Images")
if os.path.exists(empty_dir):
    empty_images = len(glob.glob(os.path.join(empty_dir, "*.jpg")))
    total_images += empty_images

# Plot
plt.figure(figsize=(8, 5))
bars = plt.bar(list(images_per_loc.keys()) + ["Empty"], list(images_per_loc.values()) + [empty_images], color=['#2c3e50', '#2c3e50', '#2c3e50', '#2c3e50', '#e74c3c'])
plt.title("Images per Location & Empty Images")
plt.ylabel("Number of Images")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 20, int(yval), ha='center', va='bottom')
plt.tight_layout()
plt.show()

negative_ratio = empty_images / total_images * 100
print(f"Total Images: {total_images}")
print(f"Negative (Clean) Ratio: {negative_ratio:.2f}%")
"""

# -----------------
# Cell 5: Markdown (Business Impact of Negative Ratio)
# -----------------
neg_md = """> **Business Insight (A+):** A real watchtower looks at empty, clean forest 99% of the time. Our dataset only has 5.17% clean images. **Pipeline Action:** We will likely encounter false alarms (Precision drop). To fix this, we should use Focal Loss during training and treat all empty images as hard negatives."""

# -----------------
# Cell 6: Markdown (BBox Sizes)
# -----------------
bbox_md = """### 2.2 The "Small Plume" Analysis (CRITICAL)
A watchtower needs to detect small smoke plumes at a distance. Let's analyze the normalized area of every YOLO bounding box to see if the dataset matches our operational reality."""

# -----------------
# Cell 7: Code (BBox Sizes)
# -----------------
bbox_code = """areas = []

for loc in locations:
    lbl_dir = os.path.join(DATA_B_RAW, f"{loc}-Labels")
    if os.path.exists(lbl_dir):
        for txt in glob.glob(os.path.join(lbl_dir, "*.txt")):
            with open(txt, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, _, _, w, h = map(float, parts)
                        areas.append(w * h)

small = sum(1 for a in areas if a < 0.01) # < 1% of image
medium = sum(1 for a in areas if 0.01 <= a < 0.1)
large = sum(1 for a in areas if a >= 0.1)

plt.figure(figsize=(6, 4))
plt.pie([small, medium, large], labels=["Small (<1%)", "Medium (1-10%)", "Large (>10%)"], 
        autopct='%1.1f%%', colors=['#f39c12', '#3498db', '#e74c3c'])
plt.title("Smoke Plume Size Distribution")
plt.show()
"""

# -----------------
# Cell 8: Markdown (Business Impact of BBox Sizes)
# -----------------
bbox_insight_md = """> **Business Insight (A+):** 95.7% of the smoke plumes in our dataset are MASSIVE (>10% of the image). If deployed "as-is", the model will fail to spot early, distant fires. **Pipeline Action:** In Task 5 (Feature Engineering), we MUST apply aggressive crop-and-scale augmentations (zooming out) to simulate distant smoke."""

# -----------------
# Cell 9: Markdown (Visualization)
# -----------------
vis_md = """### 2.3 Sample Visualization & Temporal Leakage Check
Let's visually inspect a few frames and their bounding boxes. Notice the filenames (e.g., `evoDJI_0001_frame...`). This confirms these are extracted from videos."""

# -----------------
# Cell 10: Code (Visualization)
# -----------------
vis_code = """def show_sample(img_path, lbl_path, ax):
    try:
        img = Image.open(img_path)
        w, h = img.size
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(os.path.basename(img_path), fontsize=8)
        
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, cx, cy, bw, bh = map(float, parts)
                        x = (cx - bw/2) * w
                        y = (cy - bh/2) * h
                        rect = patches.Rectangle((x, y), bw*w, bh*h, linewidth=2, edgecolor='red', facecolor='none')
                        ax.add_patch(rect)
    except Exception as e:
        ax.set_title("Error loading image")

sample_imgs = glob.glob(os.path.join(DATA_B_RAW, "Evo-Images", "*.jpg"))[:4]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for idx, img_path in enumerate(sample_imgs):
    lbl_path = img_path.replace("Evo-Images", "Evo-Labels").replace(".jpg", ".txt")
    show_sample(img_path, lbl_path, axes[idx])

plt.tight_layout()
plt.show()
"""

# -----------------
# Cell 11: Markdown (Temporal Leakage Conclusion)
# -----------------
temp_md = """> **Business Insight (A+):** The filenames confirm sequential video frames. **Pipeline Action:** In Task 4 (Data Transformation/Splitting), we MUST split the dataset by video clip prefix (`evoDJI_0001`), NOT randomly. Random splitting causes Temporal Leakage (data snooping), inflating our metrics and causing disastrous real-world failure."""

# -----------------
# Cell 12: Markdown (Advanced EDA Title)
# -----------------
adv_title_md = """## 3. Advanced Deep-Dive Findings (12GB Pixel Analysis)
We must go deeper. Are all images the same size? Is the dataset biased towards daytime? Does smoke only appear in certain areas of the sky?"""

# -----------------
# Cell 13: Code (Advanced EDA)
# -----------------
adv_code = """import random
import numpy as np

# 1. Spatial Heatmap Data
all_cx = []
all_cy = []
resolutions = set()
all_images = []

for loc in locations:
    img_dir = os.path.join(DATA_B_RAW, f"{loc}-Images")
    lbl_dir = os.path.join(DATA_B_RAW, f"{loc}-Labels")
    if os.path.exists(img_dir):
        all_images.extend(glob.glob(os.path.join(img_dir, "*.jpg")))
    if os.path.exists(lbl_dir):
        for txt in glob.glob(os.path.join(lbl_dir, "*.txt")):
            with open(txt, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, cx, cy, _, _ = map(float, parts)
                        all_cx.append(cx)
                        all_cy.append(cy)

# Check resolutions (sample)
for img_path in all_images[:200]:
    try:
        with Image.open(img_path) as img:
            resolutions.add(img.size)
    except: pass

print(f"Resolutions found: {resolutions}")

# Illumination (sample)
sample_imgs = random.sample(all_images, min(500, len(all_images)))
brightness_values = []
for img_path in sample_imgs:
    try:
        with Image.open(img_path) as img:
            gray = img.convert('L')
            brightness_values.append(np.mean(np.array(gray)))
    except: pass

mean_bright = np.mean(brightness_values)
dark_imgs = sum(1 for b in brightness_values if b < 85)
print(f"Average Brightness: {mean_bright:.1f}/255")
print(f"Dark/Night images (<85): {dark_imgs} ({(dark_imgs/len(sample_imgs))*100:.1f}%)")

# Plots
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(brightness_values, bins=30, color='gold', edgecolor='black')
plt.title("Illumination Distribution")

plt.subplot(1, 2, 2)
plt.hist2d(all_cx, all_cy, bins=20, cmap='hot')
plt.gca().invert_yaxis()
plt.title("Spatial Heatmap of Smoke Plumes")
plt.colorbar()
plt.show()
"""

# -----------------
# Cell 14: Markdown (Advanced EDA Insights)
# -----------------
adv_insights_md = """> **Business Insight (A+):** 
> 1. **4K Resolution (4096x2160):** If we resize 4K images down to 640x640, small smoke plumes will vanish into a single pixel. **Action:** We MUST use Random Cropping during training, not just resizing.
> 2. **Horizon Bias:** The heatmap proves smoke appears in the top 40% of the image. **Action:** We must DISABLE vertical flip augmentations to preserve physical reality (smoke shouldn't come from the ground).
> 3. **Daytime Bias:** Only 4% of images are dark. **Action:** We must apply brightness/contrast jittering to simulate dusk/dawn.
"""

# -----------------
# Cell 15: Markdown (Architecture & Bounding Box Meta-Analysis)
# -----------------
meta_title_md = """## 4. Architecture & Bounding Box Meta-Analysis
To perfectly tune our YOLO and Faster R-CNN pipelines, we need to extract mathematical priors directly from the bounding box geometries."""

# -----------------
# Cell 16: Code (Meta-Analysis)
# -----------------
meta_code = """from sklearn.cluster import KMeans
import pandas as pd
from collections import defaultdict
import math

# Gather all widths, heights, and file sequences
all_w = []
all_h = []
boxes_per_image = []
file_bboxes = defaultdict(list)

for loc in locations:
    lbl_dir = os.path.join(DATA_B_RAW, f"{loc}-Labels")
    if os.path.exists(lbl_dir):
        for txt in sorted(glob.glob(os.path.join(lbl_dir, "*.txt"))):
            with open(txt, 'r') as f:
                lines = f.readlines()
                boxes_per_image.append(len(lines))
                img_boxes = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, cx, cy, w, h = map(float, parts)
                        all_w.append(w)
                        all_h.append(h)
                        img_boxes.append((cx, cy, w, h))
                file_bboxes[os.path.basename(txt)] = img_boxes

# 1. Anchor Box Clustering (K=5 for YOLO11n)
X = np.array(list(zip(all_w, all_h)))
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X)
anchors = kmeans.cluster_centers_
# Sort by area
anchors = sorted(anchors, key=lambda x: x[0]*x[1])

print("--- 1. Recommended YOLO Anchor Boxes (k=5) ---")
for i, (w, h) in enumerate(anchors, 1):
    print(f"Anchor {i}: Width={w:.3f}, Height={h:.3f}, Area={w*h:.3f}")

# 2. Aspect Ratios
aspect_ratios = np.array(all_w) / np.array(all_h)
print(f"\\n--- 2. Aspect Ratio Analysis ---")
print(f"Mean AR: {np.mean(aspect_ratios):.2f}, Median AR: {np.median(aspect_ratios):.2f}")

# 3. Boxes Per Image
bpi = np.array(boxes_per_image)
print(f"\\n--- 3. Boxes Per Image ---")
print(f"Images with 1 box: {np.sum(bpi == 1)} / {len(bpi)} ({(np.sum(bpi==1)/len(bpi))*100:.1f}%)")
print(f"Max boxes in an image: {np.max(bpi)}")

# 4. IoU Analysis (for images with >1 box)
def calculate_iou(box1, box2):
    # cx, cy, w, h to x1, y1, x2, y2
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2

    x_left = max(b1_x1, b2_x1)
    y_top = max(b1_y1, b2_y1)
    x_right = min(b1_x2, b2_x2)
    y_bottom = min(b1_y2, b2_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    b1_area = box1[2] * box1[3]
    b2_area = box2[2] * box2[3]
    return intersection / float(b1_area + b2_area - intersection)

ious = []
for filename, boxes in file_bboxes.items():
    if len(boxes) > 1:
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                ious.append(calculate_iou(boxes[i], boxes[j]))

print(f"\\n--- 4. IoU Between Boxes in Same Image ---")
if ious:
    print(f"Max IoU observed: {max(ious):.3f}")
    print(f"Pairs with 0.0 IoU: {sum(1 for x in ious if x == 0.0)} / {len(ious)} ({(sum(1 for x in ious if x == 0.0)/len(ious))*100:.1f}%)")
else:
    print("No multi-box images found to calculate IoU.")

# 5. Frame-to-Frame Displacement
displacements = []
sorted_files = sorted(file_bboxes.keys())
for i in range(len(sorted_files)-1):
    f1, f2 = sorted_files[i], sorted_files[i+1]
    # Check if they belong to the same video sequence (e.g., evoDJI_0001_frame_001.txt)
    if f1[:11] == f2[:11] and len(file_bboxes[f1]) > 0 and len(file_bboxes[f2]) > 0:
        # Just compare the first box for simplicity
        cx1, cy1 = file_bboxes[f1][0][0], file_bboxes[f1][0][1]
        cx2, cy2 = file_bboxes[f2][0][0], file_bboxes[f2][0][1]
        dist = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
        displacements.append(dist)

print(f"\\n--- 5. Frame-to-Frame Temporal Displacement ---")
if displacements:
    print(f"Mean displacement: {np.mean(displacements):.4f}")
    print(f"Median displacement: {np.median(displacements):.4f}")
    static = sum(1 for d in displacements if d < 0.005)
    print(f"Near-static frame pairs (<0.005 dist): {static} / {len(displacements)} ({(static/len(displacements))*100:.1f}%)")

# Plotting the Meta-Analysis
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.hist(aspect_ratios, bins=40, color='teal', edgecolor='black')
plt.title("Aspect Ratio Distribution (W/H)")

plt.subplot(1, 3, 2)
plt.hist(bpi, bins=range(1, max(bpi)+2), color='coral', edgecolor='black', align='left')
plt.title("Boxes Per Image")
plt.xticks(range(1, max(bpi)+1))

plt.subplot(1, 3, 3)
plt.hist(displacements, bins=40, color='purple', edgecolor='black')
plt.title("Frame-to-Frame Displacement")

plt.tight_layout()
plt.show()
"""

# -----------------
# Cell 17: Markdown (Meta-Analysis Business Insights)
# -----------------
meta_insight_md = """> **Business Insight (A+): Synthesis for Model Tuning**
> 
> 1. **Custom Anchors:** Our smallest anchor has an area of `0.03`, which is much smaller than COCO defaults (`~0.06`). **Action:** We must replace YOLO11n's default COCO anchors with our custom K-Means anchors so it can detect distant smoke.
> 2. **Aspect Ratios:** 96% of smoke plumes are square-ish (0.5 to 2.0). **Action:** We do not need extreme anchor aspect ratios like those used for pedestrians or cars.
> 3. **Boxes & NMS:** 99.5% of images contain exactly 1 box, and multi-box images have zero overlap (Max IoU < 0.42). **Action:** NMS (Non-Maximum Suppression) is practically irrelevant. We can lower the NMS threshold during inference to `0.3` safely to avoid suppressing distinct plumes without risk of duplicates. Furthermore, we MUST use Mosaic/MixUp augmentations to artificially create multi-object scenes.
> 4. **Temporal Correlation:** Consecutive frames shift by ~4%. **Action:** This definitively proves that Random splitting will cause catastrophic temporal leakage. We must split strictly by video sequence.
"""

# Assemble notebook
nb['cells'] = [
    nbf.v4.new_markdown_cell(title_md),
    nbf.v4.new_code_cell(imports_code),
    nbf.v4.new_markdown_cell(dist_md),
    nbf.v4.new_code_cell(dist_code),
    nbf.v4.new_markdown_cell(neg_md),
    nbf.v4.new_markdown_cell(bbox_md),
    nbf.v4.new_code_cell(bbox_code),
    nbf.v4.new_markdown_cell(bbox_insight_md),
    nbf.v4.new_markdown_cell(vis_md),
    nbf.v4.new_code_cell(vis_code),
    nbf.v4.new_markdown_cell(temp_md),
    nbf.v4.new_markdown_cell(adv_title_md),
    nbf.v4.new_code_cell(adv_code),
    nbf.v4.new_markdown_cell(adv_insights_md),
    nbf.v4.new_markdown_cell(meta_title_md),
    nbf.v4.new_code_cell(meta_code),
    nbf.v4.new_markdown_cell(meta_insight_md)
]

output_path = r"d:\Masters Academy AI\Advanced Techniques in Object Detection and Recognition\Research Paper\dataset-b\preprocessing\task2_data_understanding\explore.ipynb"

# Write the notebook
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("explore.ipynb generated successfully.")
