# 🔥 Dataset A — Fire & Smoke Testing Probe Pipeline

> **Role in Research:** Unseen Zero-Shot Test Probe  
> **Source:** Fire and Smoke Detection Dataset (Independent UAV & Wilderness Capture)  
> **Key Objective:** Evaluate cross-domain visual generalization of models trained **exclusively on smoke (Dataset B)** when exposed to unseen **fire**.

---

## 🏗️ The 6-Task Preprocessing & Evaluation Architecture

Just like Dataset B, Dataset A follows a strict, standardized 6-task pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DATASET A PREPROCESSING PIPELINE                      │
├───────────────────┬─────────────────────────────────────────────────────────┤
│ Task 1: Business  │ Zero-Shot Probe Logic: Cost matrix of False Alarms vs   │
│ Logic & Strategy  │ Missed Ignitions, Strict IoU (0.50) vs Alerting (0.10). │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ Task 2: Data      │ Exploratory Data Analysis: Class distribution, flame    │
│ Understanding     │ aspect ratios, box area scales, blur & luminance.       │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ Task 3: Data      │ Strict 7-part cleaning: Missing value checks, MD5 dedup,│
│ Cleaning & Audit  │ outlier bounding boxes, coordinate clipping, audit log. │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ Task 4: Format &  │ Resolution Standardization to 640×640, normalized YOLO │
│ Transformation    │ TXT clipping, and COCO JSON annotation generation.      │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ Task 5: Feature   │ Test-Time Transformations, invariance probing against   │
│ Engineering       │ atmospheric haze, sensor noise, and illumination drift. │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ Task 6: Feature   │ K-Means Fire Anchor Clustering (k=5) & cross-dataset    │
│ Selection & Prior │ spatial prior comparison against Dataset B smoke anchors│
└───────────────────┴─────────────────────────────────────────────────────────┘
```

---

## 📊 Summary of Processed Test Set Statistics

| Metric | Dataset A (Test Probe) | Dataset B (Train/Val Baseline) |
| :--- | :--- | :--- |
| **Total Images** | **637** | **4,954** |
| **Images with Fire** | **459 (72.05%)** | 0 (0.0% — strictly held out) |
| **Ground-Truth Fire Boxes** | **995** | 0 |
| **Ground-Truth Smoke Boxes** | **896** | 4,868 |
| **Empty Background Images** | **178 (27.95%)** | 256 (5.17%) |
| **Standardized Image Size** | **$640 \times 640$** | $640 \times 640$ |
| **Dominant Object Scale** | **Small/Medium (80% area < 0.12)** | Large/Massive (95.7% area > 0.10) |
| **Cleaning Audit Flags** | **174 entries logged** | 4,139 entries logged |
| **Images Deleted** | **0 (Zero ground truth modified)** | 0 (Zero ground truth modified) |

---

## 📂 Directory Structure

```
dataset-a/
├── dataset/                  ← Raw partitions (test, train, valid)
│   └── test/
│       ├── images/           ← 637 original test images
│       └── labels/           ← 637 original YOLO annotation files
├── processed/                ← Standardized 640×640 evaluation assets
│   ├── images/               ← Resized 640×640 JPEG images
│   ├── labels/               ← Clipped, normalized YOLO labels
│   └── annotations/          ← test_coco.json for Faster R-CNN & DETR
├── preprocessing/            ← The 6-Task Preprocessing Suite
│   ├── task1_business_logic/
│   ├── task2_data_understanding/
│   ├── task3_data_cleaning/
│   ├── task4_data_transformation/
│   ├── task5_feature_engineering/
│   └── task6_feature_selection/
├── run_full_preprocessing_a.py ← Automated end-to-end execution script
└── README.md
```

---

## 🚀 How to Run the Pipeline

To re-run the entire data cleaning, standardization, audit trail, and anchor extraction pipeline:

```bash
python dataset-a/run_full_preprocessing_a.py
```
