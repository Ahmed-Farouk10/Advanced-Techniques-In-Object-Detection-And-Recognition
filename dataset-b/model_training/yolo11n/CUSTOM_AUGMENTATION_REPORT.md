# YOLO11n Custom Augmentation Training Report

## 1. Experiment Overview

This experiment evaluates the effect of a custom augmentation strategy on YOLO11n for early smoke detection.

The model is trained on the Boreal Forest Fire dataset (Dataset B) using smoke annotations only.

The custom augmentation strategy was designed to preserve small smoke-plume structures while improving robustness to environmental and UAV-capture variations.

---

## 2. Model

- Model: YOLO11n
- Task: Object Detection
- Target class: Smoke
- Framework: Ultralytics
- Image size: 640 × 640
- Batch size: 16
- Epochs: 100
- Device: CPU
- Pretrained weights: `yolo11n.pt`

---

## 3. Dataset

### Training Dataset

Dataset B — Boreal Forest Fire

The dataset contains UAV imagery collected from multiple Finnish locations.

The training and validation splits were created at the clip level to prevent temporal leakage between adjacent frames.

### Dataset Split

| Split | Images |
|---|---:|
| Train | 3,792 |
| Validation | 984 |

The validation set contains 38 background images with no smoke annotations.

---

## 4. Custom Augmentation Strategy

The following augmentations were enabled through `custom_hyp.yaml`.

| Augmentation | Setting | Purpose |
|---|---:|---|
| Mosaic | 0.4 | Preserve spatial context while improving scene diversity |
| Close Mosaic | 10 epochs | Disable Mosaic near the end of training for fine-tuning on natural images |
| Copy-Paste | 0.15 | Increase object/background variation |
| Scale | 0.9 | Improve robustness to different smoke-plume sizes |
| Vertical Flip | 0.0 | Disabled because vertical flipping is not physically representative of UAV forest imagery |
| HSV Hue | 0.015 | Moderate color variation |
| HSV Saturation | 0.4 | Simulate changes in scene saturation |
| HSV Value | 0.3 | Simulate lighting variation |
| Rotation | 5° | Simulate small UAV/gimbal orientation changes |
| BGR | 0.0 | Keep the standard color representation |

### Augmentations Not Activated

Fog and motion blur were considered as part of the experimental design but were not activated in this training run.

They would require additional customization of the Ultralytics augmentation pipeline beyond the native hyperparameter configuration used in this experiment.

Therefore, the reported experiment should be considered a **native YOLO11 custom augmentation experiment**, not an experiment including explicit Fog and Motion Blur transformations.

---

## 5. Training Configuration

The experiment was executed using:

```python
results = model.train(
    data="smoke_data.yaml",
    cfg="custom_hyp.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="cpu",
    project="runs",
    name="yolo11n_custom_aug",
    exist_ok=True
)