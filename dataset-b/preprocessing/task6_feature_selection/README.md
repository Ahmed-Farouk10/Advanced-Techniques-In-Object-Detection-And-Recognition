# Task 6 — Feature Selection (Augmentation Ablation)

> **Phase 2: Data Preparation | Dataset B**  
> **Status:** Designed, pending training results for execution

## Objective

Determine WHICH augmentations from Task 5 actually improve model performance — and which are neutral or harmful. Prune the augmentation pipeline to the minimal effective set.

## Why This Matters

Task 5 designed augmentations based on data analysis (WHAT should work). Task 6 validates them experimentally (WHAT DOES work). Without this, you're guessing.

## Ablation Study Design

Train YOLO11n multiple times, each time disabling ONE augmentation group. Measure mAP delta.

| Ablation | Disable | Hypothesis | Expected mAP Δ |
|----------|---------|-----------|---------------|
| Baseline | Nothing (full `custom_hyp.yaml`) | Best performance | — |
| No Mosaic | `mosaic: 0.0` | Mosaic improves generalization but may hurt small plumes | -2 to -5 mAP |
| No HSV Jitter | `hsv_h: 0, hsv_s: 0, hsv_v: 0` | HSV critical for day/night invariance | -1 to -3 mAP on dark images |
| No Copy-Paste | `copy_paste: 0.0` | Since 99.5% single-box, minor impact | -0.5 to -1 mAP |
| No Scale | `scale: 0.5` (default) | Default downscaling destroys small plumes | -3 to -6 APsmall |
| Default COCO Augs | Use YOLO default config | Custom config is better | -2 to -4 mAP |
| Close Mosaic Early | `close_mosaic: 0` | Mosaic should be disabled before final epochs | Compare mAP@final |

## Output Table

```
| Ablation | mAP@0.5 | APsmall | APmedium | APlarge | Training Time |
|----------|---------|---------|----------|---------|---------------|
| Baseline (full custom) | XX.X | X.X | XX.X | XX.X | X min |
| No Mosaic | XX.X | X.X | XX.X | XX.X | X min |
| ... | | | | | |
```

## Feature Selection for Faster R-CNN

Beyond augmentation ablation, the custom anchor test is a form of feature selection:

| Variant | Anchor Source | Hypothesis |
|---------|--------------|-----------|
| Default RPN | COCO anchors (32², 64², 128², 256², 512²) | One-size-fits-all |
| Custom RPN | k=5 smoke clusters from Task 2 | Domain-specific anchors improve smoke recall |

## Deliverables (Pending)

- Ablation execution notebook
- mAP comparison table with deltas
- Recommendation: which augmentations to keep, which to drop
- Justification for each pruning decision

## For First-Time Students

Feature selection in classical ML means picking which columns to keep (variance threshold, mutual information, RFE). In object detection, it means ablating (removing) augmentations to see which ones actually help. The concept is the same: "does this feature/augmentation improve performance, or is it just noise?"
