# Cognitive Fire Defense — Revised Master Plan
### AIN7601 | Ahmed Ayman & Esraa Nasr | Spring 2026

> **Root cause of this revision:** The original plan assumed Dataset A was object detection with bounding boxes. The EDA exposed it as a **pure image classification dataset**. Simultaneously, the 12GB Boreal dataset has arrived with **existing YOLO smoke bounding box annotations**. The entire strategy is now inverted and significantly stronger — we train on smoke to predict fire *before* it ignites. This is a far more novel and academically defensible contribution.

---

## The New Research Hypothesis

> **"A model trained exclusively on smoke can generalize to detect fire early — because smoke is the leading indicator of fire ignition, not a byproduct."**

This is a **cross-domain generalization experiment** — an academically rich framing that elevates the project beyond a standard fire detection benchmark.

---

## The Revised Two-Dataset Strategy

| Dataset | Role | Task | Format | Label Status |
|---------|------|------|--------|-------------|
| **Dataset A** — Kaggle Forest Fire (~1,520 images) | **TEST SET ONLY** | Binary Image Classification → `fire` / `no-fire` | `fire/` and `nofire/` subfolders | ✅ Implicit (folder name = label) |
| **Dataset B** — Boreal Watchtower (~4,954 images, 4 locations) | **TRAIN + VAL** (Object Detection) | Smoke detection with bounding boxes → class `0` = smoke | YOLO TXT `.txt` per image | ✅ Explicit bounding boxes present |

### Why This Is Scientifically Sound
- **Dataset A → Classification role:** Since it has no bboxes, we reclassify it as a CNN classification probe. The model trained on smoke (B) will be evaluated on fire images (A) to test cross-domain smoke→fire transfer.
- **Dataset B → Detection training:** 4,698 annotated smoke images + 256 unannotated no-smoke images. Labels are in YOLO format, ready for all 4 models after format conversion.
- **The key insight:** We are measuring whether smoke-trained detectors can **localize fire** — testing them on Dataset A fire images bridges the semantic gap from precursor to event.

---

## Revised 4-Model Matrix

| # | Model | Task Type | Training Data | Evaluation Data | Academic Justification |
|---|-------|-----------|--------------|-----------------|------------------------|
| 1 | **YOLO11n** | Object Detection | Dataset B (smoke, YOLO format) | Dataset A (fire images) | Ultra-lightweight edge inference; smoke detection → fire localization transfer test |
| 2 | **Faster R-CNN** | Object Detection | Dataset B (smoke, COCO JSON) | Dataset A (fire images) | Two-stage RPN anchor precision on smoke plumes vs. YOLO's one-stage approach |
| 3 | **RT-DETR** | Object Detection | Dataset B (smoke, COCO JSON) | Dataset A (fire images) | Transformer global attention — can it generalize smoke semantics to fire? |
| 4 | **DINO** (Deformable DETR) | Object Detection | Dataset B (smoke, COCO JSON) | Dataset A (fire images) | Deformable attention on irregular shapes — smoke plumes ARE irregular; does it transfer to flame shapes? |

> **Why all 4 use the same train/test split:** This enables a controlled, fair academic comparison. The only variable is model architecture. Dataset A becomes our held-out **zero-shot fire test set**.

---

## Dataset B — Anatomy (Confirmed from file inspection)

```
Boreal-Forest-Fire-Subset-A/
├── Evo-Images/          # 931 annotated images (DJI drone videos, Finnish boreal forest)
├── Evo-Labels/          # 931 .txt files — YOLO format (class cx cy w h), class 0 = smoke
├── Heinola-Images/      # 906 annotated + 217 empty = 943 total
├── Heinola-Labels/      # 906 .txt files (annotated only)
├── Karkkila-Images/     # 1,096 annotated + 37 empty = 1,313 total
├── Karkkila-Labels/     # 1,096 .txt files
├── Ruokolahti-Images/   # 1,765 annotated + 2 empty = 1,767 total
├── Ruokolahti-Labels/   # 1,765 .txt files
├── Empty-Images/        # No-smoke negative images (shared pool)
├── Empty-Labels/        # Corresponding empty .txt files (no annotations)
└── image_counts.txt     # ✅ Already inspected
```

---

## Dataset A — Anatomy (Confirmed from file inspection)

```
Forest Fire Dataset/
├── Training/
│   ├── fire/     # ~760 fire images (JPG, ~250×250px)
│   └── nofire/   # ~760 no-fire images
└── Testing/
    ├── fire/     # ~190 fire images (used as zero-shot probe)
    └── nofire/   # ~190 no-fire images
```

**Role Decision:** Dataset A will serve as the **zero-shot cross-domain evaluation set** for all 4 object detection models. Since the models output bounding boxes (not class labels), we will adapt evaluation:

> **Evaluation Protocol for Dataset A:** If any bounding box is predicted with confidence ≥ threshold → image classified as "fire detected". Compare to ground truth folder labels. Report per-class accuracy, precision, recall.

---

## The 6-Task Pipeline — Revised for Each Dataset

### DATASET B Pipeline (Training Pipeline)

```
Task 1 → Business Logic:    Smoke = fire precursor. Maximize Recall (missed fire = death).
                            Dataset B: 4,954 watchtower images, YOLO bounding boxes.
Task 2 → Data Understanding: EDA — smoke bbox sizes, per-location distribution,
                              empty image count, duplicate frame detection.
Task 3 → Data Cleaning:     Remove corrupt images, validate bbox coords (0-1 range),
                              handle empty annotation files (treat as negative class).
Task 4 → Data Transformation: Convert ALL labels to both YOLO TXT (M1) and COCO JSON (M2/M3/M4).
                               Merge all 4 locations into unified pool.
                               Split: 70% train / 15% val / 15% test (stratified by location).
                               Resize: 640×640 for YOLO11n/RT-DETR; 800×800 for DINO.
Task 5 → Feature Engineering: Augmentations:
                               - Fog/haze simulation (watchtower scenario)
                               - Brightness/contrast jitter (day/night variation)
                               - Random crop + scale (small smoke at distance)
                               - Horizontal flip (symmetric forest scenes)
Task 6 → Feature Selection:  Lock augmentation config. Prune any augmentation that
                              causes bbox coordinate corruption. Final configs written.
```

### DATASET A Pipeline (Evaluation Pipeline — NO training)

```
Task 1 → Business Logic:    Probe whether smoke-trained model generalizes to fire.
Task 2 → Data Understanding: Count fire/nofire images in test split. Confirm no corruption.
Task 3 → Data Cleaning:     Remove corrupt images. Verify dimensions consistent.
Task 4 → Data Transformation: Resize to match each model's expected input (640×640, 800×800).
                               No annotation conversion needed (evaluation is folder-label based).
Task 5 → Feature Engineering: NO augmentation (this is the test set — no data leakage).
Task 6 → Feature Selection:   Lock eval config. Define confidence threshold sweep (0.3→0.7).
```

---

## Directory Optimization Plan

### Target Structure (Adding to current structure):

```
dataset-a/                                  
├── raw/Forest Fire Dataset/                
├── preprocessing/
│   ├── task1_business_logic/
│   ├── task2_data_understanding/
│   ├── task3_data_cleaning/
│   ├── task4_data_transformation/
│   ├── task5_feature_engineering/
│   └── task6_feature_selection/
├── processed/                      [eval images resized]
├── evaluation/                     [NEW — eval results]
│   ├── yolo11n/
│   ├── faster_rcnn/
│   ├── rt_detr/
│   └── dino/
└── README.md                       [UPDATE]

dataset-b/                                  
├── raw/Boreal-Forest-Fire-Subset-A/ 
├── preprocessing/
│   ├── task1_business_logic/
│   ├── task2_data_understanding/
│   ├── task3_data_cleaning/
│   ├── task4_data_transformation/
│   ├── task5_feature_engineering/
│   └── task6_feature_selection/
├── processed/
│   ├── unified/                    [merged 4 locations]
│   │   ├── images/train/
│   │   ├── images/val/
│   │   ├── images/test/
│   │   ├── labels/train/           [YOLO TXT]
│   │   ├── labels/val/
│   │   └── labels/test/
│   └── coco_format/               [COCO JSON for M2/M3/M4]
│       ├── train.json
│       ├── val.json
│       └── test.json
├── dino/        [training outputs]
├── faster_rcnn/ [training outputs]
├── rt_detr/     [training outputs]
└── yolo11n/     [training outputs]
```

---

## Preprocessing Scripts to Create

All scripts go in `shared/` (already exists in the directory).

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `eda_dataset_b.py` | EDA — count images, bbox stats, per-location distribution | `dataset-b/raw/` | `task2_data_understanding/` report |
| `clean_dataset_b.py` | Validate YOLO coords, remove corrupt images, pair orphan labels | `dataset-b/raw/` | Cleaned image+label list |
| `merge_and_split_b.py` | Merge 4 locations, stratified 70/15/15 split | Cleaned pairs | `dataset-b/processed/unified/` |
| `convert_to_coco.py` | Convert YOLO TXT → COCO JSON | `unified/labels/` | `processed/coco_format/train/val/test.json` |
| `resize_dataset_a.py` | Resize Dataset A test images for evaluation | `dataset-a/raw/Testing/` | `dataset-a/processed/` (640 and 800 versions) |
| `evaluate_on_dataset_a.py` | Run inference → bbox→classification → report | Any trained model | mAP, Acc, P, R, F1 per model |

---

## Training Configs (Revised)

| Param | YOLO11n | Faster R-CNN | RT-DETR | DINO |
|-------|---------|--------------|---------|------|
| Dataset | B (train/val) | B (train/val) | B (train/val) | B (train/val) |
| Eval | A (test probe) | A (test probe) | A (test probe) | A (test probe) |
| Epochs | 100 (early stop) | 50 | 50 | 50 |
| Batch | 8 | 4 | 4 | 2 + grad_accum=4 |
| Img Size | 640×640 | 640×640 | 640×640 | 800×800 |
| Classes | 1 (smoke) | 1 (smoke) | 1 (smoke) | 1 (smoke) |
| Optimizer | AdamW | SGD | AdamW | AdamW |
| LR | 0.001 | 0.005 | 0.0001 | 0.0002 |
| FP16 | ✅ | ❌ | ✅ | ✅ |
| Format | YOLO TXT | COCO JSON | COCO JSON | COCO JSON |

---

## Evaluation Framework

### Stage 1: Standard Detection (on Dataset B test split)
| Metric | Why |
|--------|-----|
| mAP@0.5 | Standard detection performance on smoke |
| mAP@0.5:0.95 | Stricter localization quality |
| Precision | False alarm rate |
| Recall | Miss rate (CRITICAL — optimize this) |
| FPS (T4) | Real-time viability |
| Model Size (MB) | Deployment constraint |

### Stage 2: Cross-Domain Transfer (on Dataset A — zero-shot fire probe)
| Metric | Formula | Why |
|--------|---------|-----|
| Fire Detection Rate | TP_fire / (TP_fire + FN_fire) | Did the model "see" fire when looking for smoke? |
| False Alarm Rate | FP_nofire / (TN_nofire + FP_nofire) | How often does it panic on clean forest? |
| Transfer Accuracy | (TP_fire + TN_nofire) / total | Overall cross-domain accuracy |

> **Conf threshold sweep:** Report at θ = 0.3, 0.5, 0.7 — lower threshold favors recall (life-safety priority).

---

## Academic Narrative (Paper Structure)

1. **Abstract:** Smoke-trained detectors as fire precursor sensors — 4-way architecture comparison
2. **Introduction:** The pre-ignition detection problem; why smoke precedes fire by 2-20 minutes
3. **Related Work:** Fire detection (YOLO family), smoke detection (separate literature), transfer learning for rare events
4. **Datasets:** Boreal Watchtower (smoke training) + Kaggle Forest Fire (fire evaluation probe)
5. **Methodology:** 6-Task Pipeline → 4 models → cross-domain evaluation protocol
6. **Results:** Detection table + transfer accuracy table + qualitative examples
7. **Discussion:** Which architecture transfers best and WHY (architectural explanation)
8. **Conclusion:** Smoke→fire generalization is viable; architectural recommendations for edge deployment

---

## Execution Order (Implementation Plan)

### PHASE 0 — Update All MD Files (TODAY)
- [x] Update `forest-fire-detection.md` (full rewrite)
- [x] Update `ain7601-pipeline/SKILL.md`
- [x] Update `ain7601-project-manager.md`
- [x] Update `project-log.md`
- [x] Update `README.md`
- [x] Create `dataset-b/preprocessing/task1_business_logic/business_logic.md`
- [x] Update `dataset-a/preprocessing/` task files if they exist

### PHASE 1 — Dataset B EDA & Cleaning
- [ ] Write + run `shared/eda_dataset_b.py`
- [ ] Write + run `shared/clean_dataset_b.py`
- [ ] Document findings in `dataset-b/preprocessing/task2_data_understanding/`

### PHASE 2 — Dataset B Processing
- [ ] Write + run `shared/merge_and_split_b.py` → 70/15/15 split
- [ ] Write + run `shared/convert_to_coco.py` → COCO JSON
- [ ] Create `dataset-b/processed/unified/` structure

### PHASE 3 — Dataset A Evaluation Prep
- [ ] Write `shared/resize_dataset_a.py` → 640 and 800 versions
- [ ] Create `dataset-a/processed/` evaluation-ready images

### PHASE 4 — Training (Google Colab / Kaggle)
- [ ] YOLO11n on Dataset B
- [ ] Faster R-CNN on Dataset B
- [ ] RT-DETR on Dataset B
- [ ] DINO on Dataset B

### PHASE 5 — Cross-Domain Evaluation
- [ ] Write `shared/evaluate_on_dataset_a.py`
- [ ] Run all 4 models on Dataset A fire test images
- [ ] Collect Transfer Accuracy, Fire Detection Rate, False Alarm Rate

### PHASE 6 — Paper
- [ ] Comparison tables
- [ ] Qualitative figures
- [ ] Final paper
