# Dataset B — Boreal Forest Fire (Watchtower/Drone)

> **AIN7601 Cognitive Fire Defense Pipeline | Spring 2026**
> **Role:** Exclusive training & validation source for all 4 models
> **4,954 images | 4,862 bounding boxes | Single class: Smoke (0)**

---

## Quick Start — Where to Look

| I want to... | Go here |
|-------------|---------|
| Understand the project from scratch | `../README.md` and `../forest-fire-detection.md` |
| See the raw data | `raw/Boreal-Forest-Fire-Subset-A/` |
| Follow the preprocessing pipeline (Tasks 1-6) | `preprocessing/` |
| Train a model | `model_training/` |
| See training results | `yolo11n/results/`, `dino/results/`, etc. |
| Use the data in YOLO format | `yolo_format/` |
| Use the data in COCO format | `coco_format/` |
| Understand all risks | `../premortem.md` |

---

## Directory Map

```
dataset-b/
├── README.md                          ← You are here
│
├── raw/                               ← Original data (never modified)
│   └── Boreal-Forest-Fire-Subset-A/
│       ├── Evo-Images/        (931 images)
│       ├── Evo-Labels/
│       ├── Heinola-Images/    (906 images)
│       ├── Heinola-Labels/
│       ├── Karkkila-Images/   (1096 images)
│       ├── Karkkila-Labels/
│       ├── Ruokolahti-Images/ (1765 images)
│       ├── Ruokolahti-Labels/
│       ├── Empty-Images/      (256 clean forest backgrounds)
│       └── Empty-Labels/
│
├── preprocessing/                     ← The 6-task pipeline (Phase 1-2)
│   ├── task1_business_logic/
│   │   ├── README.md
│   │   └── business_logic.md          ← ML translation + constraints
│   │
│   ├── task2_data_understanding/
│   │   ├── README.md
│   │   ├── explore.ipynb              ← Main EDA notebook
│   │   ├── data_understanding.md      ← Findings + business impact
│   │   └── advanced_eda_plots.png     ← Spatial/illumination analysis
│   │
│   ├── task3_data_cleaning/
│   │   ├── README.md
│   │   ├── preprocess.ipynb           ← Cleaning execution notebook
│   │   ├── cleaning_log.csv           ← 4,139 flags, 0 deletions
│   │   ├── blur_dist.png              ← Blur score distribution
│   │   └── advanced_cleaning_research.md  ← 25 techniques surveyed
│   │
│   ├── task4_data_splitting/
│   │   ├── README.md
│   │   └── split.ipynb                ← Constraint optimization split
│   │
│   ├── task5_feature_engineering/
│   │   └── README.md                  ← Augmentation design doc
│   │
│   └── task6_feature_selection/
│       └── README.md                  ← Ablation plan
│
├── model_training/                    ← Training configs (Phase 3)
│   ├── yolo11n/
│   │   ├── train.py                   ← YOLO11n training script
│   │   ├── train.ipynb                ← Demonstration notebook
│   │   ├── smoke_data.yaml            ← Dataset paths config
│   │   └── custom_hyp.yaml            ← Custom hyperparameters
│   ├── rtdetr/
│   │   ├── train_rtdetr.py
│   │   └── train.ipynb
│   ├── faster_rcnn/
│   │   ├── train_faster_rcnn.py
│   │   └── train.ipynb
│   └── dino/
│       ├── train_dino.py
│       └── train.ipynb
│
├── yolo_format/                       ← YOLO TXT format (train/val/test)
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
│
├── coco_format/                       ← COCO JSON format (train/val/test)
│   ├── images/{train,val,test}/
│   └── annotations/{train,val,test}.json
│
└── yolo11n/dino/rt_detr/faster_rcnn/  ← Training outputs (results/)
```

---

## The 6-Task Pipeline (What We Did)

| Phase | Task | Question Answered | Key Decision |
|-------|------|-------------------|-------------|
| **Phase 1** | Task 1 — Business Logic | What problem are we solving? | Maximize Recall. Split by video clip, not random frame. |
| | Task 2 — Data Understanding | What's in the data? | 95.7% large plumes. 4K resolution. 30 video clips. Daytime bias. |
| **Phase 2** | Task 3 — Data Cleaning | Is the data clean? | 4,139 flags, 0 deletions. pHash near-dupes flagged. All annotations valid. |
| | Task 4 — Data Splitting | How do we split without leakage? | Clip-level constraint optimization. 80/20 split. Zero temporal leakage. |
| | Task 5 — Feature Engineering | What augmentations do we need? | Mosaic(0.4), scale(0.9), HSV jitter, flipud=0. Fog/blur documented. |
| | Task 6 — Feature Selection | Which augmentations actually help? | Ablation study (post-training). Custom anchors for Faster R-CNN. |

---

## The 4 Models Being Trained

| # | Model | Paradigm | Framework | Format | Batch | Owner |
|---|-------|----------|-----------|--------|-------|-------|
| 1 | YOLO11n | One-stage CNN (anchor-free) | Ultralytics | YOLO TXT | 16 | Esraa |
| 2 | RT-DETR | Real-time Transformer (query-based) | Ultralytics | YOLO TXT | 8 | Ahmed |
| 3 | Faster R-CNN | Two-stage RPN (custom anchors) | torchvision | COCO JSON | 4 | Esraa |
| 4 | DINO | Deformable Attention Transformer | HuggingFace | COCO JSON | 2 | Ahmed |

---

## Key Findings (for the Paper)

1. **Anchor clustering:** Smoke plumes cluster into 5 size groups (0.03 to 0.73 area). Custom anchors injected into Faster R-CNN's RPN.
2. **Temporal leakage prevented:** pHash dedup + clip-level constraint optimization eliminates soft leakage.
3. **Small plume bias:** Only 1.3% of plumes are small (<1% area). Addressed via scale augmentations and random cropping.
4. **Daytime bias:** 96% of images are bright. HSV jitter simulates dusk/dawn.
5. **Validation integrity:** 4,139 anomalies flagged, zero ground truth modified. Full audit trail in `cleaning_log.csv`.

---

## For First-Time Students

If this is your first time taking AIN7601:

- **Start with** `../README.md` — project overview
- **Then read** `preprocessing/task1_business_logic/business_logic.md` — the "why" behind everything
- **Then open** `preprocessing/task2_data_understanding/explore.ipynb` — the EDA notebook
- **Training guides** are in `model_training/` — open the `.ipynb` files, not the `.py` files
- **Ask Ahmed** if anything is unclear — this pipeline was built with A+ philosophy (every decision linked to business impact)
