# 🔥 Cognitive Fire Defense Pipeline
> **AIN7601 — Advanced Techniques in Object Detection and Recognition | Spring 2026**  
> **Team:** Esraa Nasr & Ahmed Ayman

A cross-domain generalization experiment for forest fire surveillance. We train four architecturally distinct object detection models entirely on **smoke** (the precursor), and test their ability to detect **fire** zero-shot. This proves that lightweight edge detectors paired with advanced transformers can learn the semantic concept of a fire from smoke alone.

---

## The Cross-Domain Transfer Strategy

> **"A model trained exclusively on smoke can generalize to detect fire early — because smoke is the leading indicator of fire ignition, not a byproduct."**

```
TRAINING PHASE (Smoke Only)                  EVALUATION PHASE (Fire Only)
───────────────────────────                  ────────────────────────────
Dataset B (Watchtower)                       Dataset A (Drone/UAV)
Boreal Forest Fire dataset
4,954 images, YOLO BBoxes                    1,520 images, No BBoxes (Classification)
Classes: [Smoke]                             Classes: [Fire, No-Fire]

Model 1: YOLO11n                      ──────>  Transfer Accuracy
Model 2: Faster R-CNN + MobileNetV3   ──────>  Fire Detection Rate
Model 3: RT-DETR                      ──────>  False Alarm Rate
Model 4: Deformable DETR              ──────>  Zero-Shot Sensitivity
```

---

## Datasets

| | Dataset | Role | Camera | Format |
|-|---------|------|--------|---------|
| **A** | [Forest Fire (Drone)](https://www.kaggle.com/datasets/alik05/forest-fire-dataset) | **Test Probe** | Moving drone | Image Classification |
| **B** | Fairdata / Boreal (Watchtower) | **Train/Val** | Static panoramic | Object Detection (YOLO BBox & COCO JSON) |

> ⚠️ **Data is not stored in this repository.** See `dataset-a/README.md` and `dataset-b/README.md` for details.

---

## Models Evaluated

| # | Model | Paradigm | Lead Owner | Why This Architecture? |
|---|-------|----------|------------|------------------------|
| 1 | **YOLO11n** | One-Stage CNN (Anchor-free) | Esraa Nasr | Ultra-lightweight edge baseline; testing if pure CNNs can generalize zero-shot. |
| 2 | **Faster R-CNN** | Two-Stage RPN (Custom anchors) | Esraa Nasr | MobileNetV3-FPN backbone with $k=5$ domain-specific anchor clustering for large plumes. |
| 3 | **RT-DETR** | Real-Time Transformer (Hybrid) | Ahmed Ayman | Hybrid CNN-Transformer encoder with learned query decoder; eliminates NMS. |
| 4 | **Deformable DETR** | Sparse Attention Transformer | Ahmed Ayman | Sparse deformable attention captures irregular amorphous smoke boundaries with linear complexity. (See [Deformable DETR README](dataset-b/model_training/dino/README.md)). |

> 📊 **Full Training Results & Detailed Contribution Matrix:** See [`CHANGELOG.md`](CHANGELOG.md).

---

## Repository Structure

```
.
├── dataset-a/
│   ├── raw/
│   ├── preprocessing/      ← Evaluation pipeline tasks
│   ├── processed/          ← Resized evaluation images
│   └── evaluation/         ← Cross-domain transfer results
├── dataset-b/
│   ├── raw/
│   ├── preprocessing/      ← Training pipeline tasks
│   ├── processed/          ← Unified YOLO & COCO formats
│   └── [models]/           ← Training outputs
├── shared/                 ← Reusable Python scripts
├── docs/                   ← Course deliverables
├── paper/                  ← Academic paper drafting
├── requirements.txt
├── .gitignore
├── forest-fire-detection.md ← Full project plan
└── premortem.md            ← Risk analysis & temporal leakage mitigation
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/forest-fire-detection.git
cd forest-fire-detection
pip install -r requirements.txt
```

### 2. Execution Order

See the **Phase X** breakdown in [`forest-fire-detection.md`](forest-fire-detection.md) for the exact script execution order, from EDA to Training to Evaluation.

---

## Citation

```bibtex
@misc{nasr2026cognitivefire,
  title   = {Cognitive Fire Defense: Zero-Shot Smoke-to-Fire Generalization in Object Detection},
  author  = {Nasr, Esraa and Ayman, Ahmed},
  year    = {2026},
  note    = {AIN7601 Research Project, Spring 2026}
}
```
