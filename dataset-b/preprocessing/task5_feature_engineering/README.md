# Task 5 — Feature Engineering (Augmentation Design)

> **Phase 2: Data Preparation | Dataset B**

## Objective

Design an augmentation pipeline that addresses every data bias discovered in Task 2, without destroying the rare small smoke plumes.

## Augmentation Design Rationale

Every augmentation parameter in `custom_hyp.yaml` traces back to a specific Task 2 finding:

| Task 2 Finding | Augmentation | Parameter | Rationale |
|---------------|-------------|-----------|-----------|
| 95.7% large plumes → model ignores small smoke | Scale | 0.9 (increased from 0.5) | Less downscaling preserves small plume features |
| 95.7% large plumes → model ignores small smoke | Mosaic | 0.4 (reduced from 1.0) | Retains spatial context; full Mosaic halves image size |
| 95.7% large plumes → model ignores small smoke | close_mosaic | 10 | Fine-tune last 10 epochs on clean data |
| 99.5% single-box → NMS untested | copy_paste | 0.15 | Paste smoke into empty backgrounds for multi-object |
| 96% daytime bias → dusk/dawn blindness | hsv_v | 0.3 | Brightness jitter simulates low-light |
| 96% daytime bias → dusk/dawn blindness | hsv_h | 0.015 | Hue variation for sky color changes |
| 96% daytime bias → dusk/dawn blindness | hsv_s | 0.4 | Saturation drop in haze/fog |
| Horizon bias (top 40%) → smoke never at ground | flipud | 0.0 | Disabled. Vertical flip teaches wrong physical prior |
| Drone platform → gimbal drift | degrees | 5.0 | Small rotation tolerance |

## Fog & Motion Blur — Documented Gap

Fog and motion blur augmentations require injecting custom `Albumentations` transforms into Ultralytics' internal `Dataset` class. This was NOT implemented to preserve framework compatibility. Native HSV jitter, scale, and Mosaic provide partial coverage. The gap is honestly documented in `train.py` with the override code included as a comment.

## Anchor Strategy

| Model | Anchor Policy | Why |
|-------|--------------|-----|
| YOLO11n | Anchor-free (Task-Aligned Assigner) | Modern architecture, cannot inject anchors |
| RT-DETR | Query-based (no anchors) | Transformer decoder learns queries |
| **Faster R-CNN** | **Custom k=5 smoke clusters** | Injected into RPN `AnchorGenerator` |
| DINO | Query-based (no anchors) | Deformable attention learns queries |

The custom anchors from Task 2:
```
[[0.15, 0.20], [0.36, 0.54], [0.62, 0.46], [0.60, 0.71], [0.78, 0.94]]
```

## Outputs

- `../../model_training/yolo11n/custom_hyp.yaml` — YOLO hyperparameter overrides
- `../../model_training/yolo11n/smoke_data.yaml` — Dataset path config (shared by RT-DETR)
- `../../model_training/yolo11n/train.py` — Training script with honest Albumentations documentation

## For First-Time Students

"Feature engineering" in classical ML means creating new columns from existing ones (like `MonthlyCharges / Tenure`). In computer vision, it means designing AUGMENTATIONS — transforms applied to images during training that force the model to learn invariant features. This task is about choosing WHICH transforms to apply and WHY. The actual transforms happen on-the-fly in GPU memory during `model.train()` — no images are permanently modified.
