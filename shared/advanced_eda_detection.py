import os
import glob
import re
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter, defaultdict
import numpy as np
from sklearn.cluster import KMeans
import base64
from io import BytesIO

DATA_B_RAW = r"d:\Masters Academy AI\Advanced Techniques in Object Detection and Recognition\Research Paper\dataset-b\raw\Boreal-Forest-Fire-Subset-A"
NOTEBOOK_PATH = r"d:\Masters Academy AI\Advanced Techniques in Object Detection and Recognition\Research Paper\dataset-b\preprocessing\task2_data_understanding\explore.ipynb"

locations = ["Evo", "Heinola", "Karkkila", "Ruokolahti"]

def image_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ---------- Collect all detection data ----------
all_boxes = []
all_images = []
image_box_count = defaultdict(int)
image_to_boxes = defaultdict(list)
image_wh = {}

for loc in locations:
    img_dir = os.path.join(DATA_B_RAW, f"{loc}-Images")
    lbl_dir = os.path.join(DATA_B_RAW, f"{loc}-Labels")
    if not os.path.exists(img_dir):
        continue
    for img_path in glob.glob(os.path.join(img_dir, "*.jpg")):
        all_images.append(img_path)
        lbl_path = os.path.join(lbl_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_, cx, cy, bw, bh = map(float, parts)
                        all_boxes.append((cx, cy, bw, bh))
                        image_box_count[img_path] += 1
                        image_to_boxes[img_path].append((cx, cy, bw, bh))

print(f"Total images with annotations: {len(image_to_boxes)}")
print(f"Total bounding boxes: {len(all_boxes)}")

# ============================================================
# ANALYSIS 1: Anchor Box Clustering (k-means on w, h)
# ============================================================
print("\n=== ANALYSIS 1: Anchor Box Clustering ===")
widths = np.array([b[2] for b in all_boxes])
heights = np.array([b[3] for b in all_boxes])
wh = np.column_stack([widths, heights])

for k in [3, 5, 9]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(wh)
    centers = kmeans.cluster_centers_
    centers_sorted = centers[np.argsort(centers[:, 0] * centers[:, 1])]
    print(f"\nk={k} anchor boxes (normalized w, h):")
    for i, (w, h) in enumerate(centers_sorted):
        print(f"  Anchor {i+1}: w={w:.4f}, h={h:.4f}  (area={w*h:.6f})")

# Plot for k=5
kmeans5 = KMeans(n_clusters=5, random_state=42, n_init=10)
labels5 = kmeans5.fit_predict(wh)
centers5 = kmeans5.cluster_centers_
colors = plt.cm.tab10(np.linspace(0, 1, 5))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Scatter plot
for i in range(5):
    mask = labels5 == i
    axes[0].scatter(widths[mask], heights[mask], c=[colors[i]], s=5, alpha=0.5, label=f'Cluster {i+1}')
axes[0].scatter(centers5[:, 0], centers5[:, 1], c='red', marker='X', s=200, edgecolors='black', linewidths=1.5, zorder=10)
axes[0].set_xlabel("Normalized Width")
axes[0].set_ylabel("Normalized Height")
axes[0].set_title("K-Means Clustering of BBox Dimensions (k=5)")
axes[0].legend(markerscale=3, fontsize=7)
axes[0].grid(True, alpha=0.3)

# Cluster size pie
sizes = [np.sum(labels5 == i) for i in range(5)]
axes[1].pie(sizes, labels=[f'Cluster {i+1}' for i in range(5)], autopct='%1.1f%%',
            colors=colors)
axes[1].set_title("Cluster Distribution")

plt.tight_layout()
plt.savefig("anchor_clustering.png", dpi=120)
plt.close()

# ============================================================
# ANALYSIS 2: Aspect Ratio Distribution
# ============================================================
print("\n=== ANALYSIS 2: Aspect Ratio Distribution ===")
aspect_ratios = widths / (heights + 1e-8)

print(f"Aspect ratio (w/h):")
print(f"  Mean: {np.mean(aspect_ratios):.3f}")
print(f"  Median: {np.median(aspect_ratios):.3f}")
print(f"  Std: {np.std(aspect_ratios):.3f}")
print(f"  Wide boxes (AR>2): {np.sum(aspect_ratios > 2)} ({np.mean(aspect_ratios > 2)*100:.1f}%)")
print(f"  Square-ish (0.5<AR<2): {np.sum((aspect_ratios > 0.5) & (aspect_ratios < 2))} ({np.mean((aspect_ratios > 0.5) & (aspect_ratios < 2))*100:.1f}%)")
print(f"  Tall boxes (AR<0.5): {np.sum(aspect_ratios < 0.5)} ({np.mean(aspect_ratios < 0.5)*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(aspect_ratios, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(x=1, color='red', linestyle='--', linewidth=1.5, label='Square (1:1)')
axes[0].axvline(x=np.mean(aspect_ratios), color='orange', linestyle='--', linewidth=1.5, label=f'Mean={np.mean(aspect_ratios):.2f}')
axes[0].set_xlabel("Width/Height Ratio")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Aspect Ratio Distribution")
axes[0].legend()

axes[1].hist(np.log10(aspect_ratios), bins=50, color='darkseagreen', edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Square (log=0)')
axes[1].set_xlabel("log10(Width/Height)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Aspect Ratio (Log Scale)")
axes[1].legend()

plt.tight_layout()
plt.savefig("aspect_ratio.png", dpi=120)
plt.close()

# ============================================================
# ANALYSIS 3: Boxes Per Image
# ============================================================
print("\n=== ANALYSIS 3: Boxes Per Image ===")
counts = list(image_box_count.values())
count_dist = Counter(counts)

print(f"Images with bboxes: {len(counts)}")
print(f"  Max boxes in one image: {max(counts)}")
print(f"  Mean boxes per image: {np.mean(counts):.2f}")
print(f"  Median boxes per image: {np.median(counts):.1f}")
print(f"  Single-box images: {count_dist.get(1, 0)} ({count_dist.get(1, 0)/len(counts)*100:.1f}%)")
print(f"  Multi-box images (>=2): {sum(v for k,v in count_dist.items() if k >= 2)} ({sum(v for k,v in count_dist.items() if k >= 2)/len(counts)*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Histogram
max_n = min(10, max(counts))
axes[0].hist(counts, bins=max_n+1, color='coral', edgecolor='black', align='left', alpha=0.7)
axes[0].set_xlabel("Number of Bounding Boxes per Image")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Boxes Per Image Distribution")
axes[0].set_xticks(range(0, max_n+1))

# Categorical
cats = ['1 box', '2 boxes', '3 boxes', '4+ boxes']
cat_counts = [
    count_dist.get(1, 0),
    count_dist.get(2, 0),
    count_dist.get(3, 0),
    sum(v for k, v in count_dist.items() if k >= 4)
]
axes[1].bar(cats, cat_counts, color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
for i, v in enumerate(cat_counts):
    axes[1].text(i, v + 10, str(v), ha='center', fontweight='bold')
axes[1].set_ylabel("Frequency")
axes[1].set_title("Images by Box Count Category")

plt.tight_layout()
plt.savefig("boxes_per_image.png", dpi=120)
plt.close()

# ============================================================
# ANALYSIS 4: IoU Analysis (box overlaps in multi-box images)
# ============================================================
print("\n=== ANALYSIS 4: IoU Analysis ===")
def compute_iou(box1, box2):
    cx1, cy1, w1, h1 = box1
    cx2, cy2, w2, h2 = box2
    x1_min = cx1 - w1/2
    y1_min = cy1 - h1/2
    x1_max = cx1 + w1/2
    y1_max = cy1 + h1/2
    x2_min = cx2 - w2/2
    y2_min = cy2 - h2/2
    x2_max = cx2 + w2/2
    y2_max = cy2 + h2/2
    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

all_ious = []
for img_path, boxes in image_to_boxes.items():
    n = len(boxes)
    if n < 2:
        continue
    for i in range(n):
        for j in range(i+1, n):
            iou = compute_iou(boxes[i], boxes[j])
            all_ious.append(iou)

print(f"Multi-box images analyzed: {sum(1 for v in count_dist.values() if v >= 2)}")
print(f"Total box pairs computed: {len(all_ious)}")
print(f"  Mean IoU: {np.mean(all_ious):.4f}")
print(f"  Median IoU: {np.median(all_ious):.4f}")
print(f"  Max IoU: {np.max(all_ious):.4f}")
print(f"  Pairs with IoU > 0.5 (high overlap): {np.sum(np.array(all_ious) > 0.5)} ({np.mean(np.array(all_ious) > 0.5)*100:.2f}%)")
print(f"  Pairs with IoU > 0.0 (any overlap): {np.sum(np.array(all_ious) > 0)} ({np.mean(np.array(all_ious) > 0)*100:.2f}%)")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(all_ious, bins=30, color='mediumpurple', edgecolor='black', alpha=0.7)
axes[0].axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, label='NMS threshold (0.5)')
axes[0].set_xlabel("IoU")
axes[0].set_ylabel("Frequency (log scale)")
axes[0].set_yscale('log')
axes[0].set_title("Pairwise IoU Distribution (Multi-Box Images)")
axes[0].legend()

high_overlaps = [iou for iou in all_ious if iou > 0.01]
if high_overlaps:
    axes[1].hist(high_overlaps, bins=30, color='mediumpurple', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, label='NMS threshold (0.5)')
    axes[1].set_xlabel("IoU (non-zero overlaps only)")
    axes[1].set_ylabel("Frequency (log scale)")
    axes[1].set_yscale('log')
    axes[1].set_title("Non-Zero IoU Distribution")
    axes[1].legend()

plt.tight_layout()
plt.savefig("iou_analysis.png", dpi=120)
plt.close()

# ============================================================
# ANALYSIS 5: Frame-to-Frame Box Displacement
# ============================================================
print("\n=== ANALYSIS 5: Frame-to-Frame Box Displacement ===")
import re

# Group images by video clip prefix
video_groups = defaultdict(list)
for img_path in all_images:
    fname = os.path.basename(img_path)
    match = re.match(r'(\w+_\d+_frame)', fname)
    if match:
        video_groups[match.group(1)].append(img_path)

# Sort frames within each group and compute displacement
displacements = []
valid_clips = 0
for prefix, imgs in video_groups.items():
    if len(imgs) < 2:
        continue
    sorted_imgs = sorted(imgs)
    valid_clips += 1
    for i in range(len(sorted_imgs) - 1):
        boxes1 = image_to_boxes.get(sorted_imgs[i], [])
        boxes2 = image_to_boxes.get(sorted_imgs[i+1], [])
        if len(boxes1) != len(boxes2) or len(boxes1) == 0:
            continue
        for b1, b2 in zip(boxes1, boxes2):
            dx = b2[0] - b1[0]
            dy = b2[1] - b1[1]
            dist = np.sqrt(dx**2 + dy**2)
            displacements.append(dist)

print(f"Video clips found: {len(video_groups)}")
print(f"Clips with >=2 annotated frames: {valid_clips}")
print(f"Total frame-pair displacements computed: {len(displacements)}")
if displacements:
    print(f"  Mean displacement (norm): {np.mean(displacements):.6f}")
    print(f"  Median displacement (norm): {np.median(displacements):.6f}")
    print(f"  Max displacement (norm): {np.max(displacements):.6f}")
    print(f"  Std displacement (norm): {np.std(displacements):.6f}")
    print(f"  Near-static frames (<0.005): {np.sum(np.array(displacements) < 0.005)} ({np.mean(np.array(displacements) < 0.005)*100:.1f}%)")
else:
    print("  No valid frame pairs found for displacement analysis.")

if displacements:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(displacements, bins=50, color='teal', edgecolor='black', alpha=0.7)
    axes[0].axvline(x=np.mean(displacements), color='red', linestyle='--', label=f'Mean={np.mean(displacements):.4f}')
    axes[0].set_xlabel("Normalized Displacement (Euclidean)")
    axes[0].set_ylabel("Frequency (log scale)")
    axes[0].set_yscale('log')
    axes[0].set_title("Frame-to-Frame Box Displacement")
    axes[0].legend()

    axes[1].hist(np.log10(np.array(displacements) + 1e-8), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel("log10(Displacement)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Displacement (Log Scale)")

    plt.tight_layout()
    plt.savefig("displacement_analysis.png", dpi=120)
    plt.close()

print("\n=== ALL ANALYSES COMPLETE ===")
print("Plots saved: anchor_clustering.png, aspect_ratio.png, boxes_per_image.png, iou_analysis.png, displacement_analysis.png")
