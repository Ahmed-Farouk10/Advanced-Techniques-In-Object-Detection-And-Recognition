# Experiments — Cognitive Fire Defense

## E01: Smoke Domain Validation (Within-Domain)
- Tests: Establishes within-domain performance baseline
- Setup: All four models evaluated on held-out Boreal validation split (926 images)
- Procedure: Standard COCO evaluation protocol
- Metrics: mAP@0.5, mAP@0.5:0.95, AP_small/medium/large, Precision, Recall, F1, FPS
- Baselines: Kim & Muminov (2023) 86.4% AP@50; Chetoui & Akhloufi (2024) 92.6% mAP@50
- Note: Baselines are within-domain (smoke→smoke). Our scores may differ due to different splits and single-class design.

## E02: Zero-Shot Fire Transfer (Cross-Domain)
- Tests: C01, C02 — does smoke-trained model detect fire without fire training?
- Setup: All four smoke-trained models evaluated on Kaggle Forest Fire Dataset (760 fire / 760 no-fire)
- Procedure: No parameter updates. Any detected bounding box > τ counts as a detection.
- Metrics: Fire Detection Rate, False Positive Rate, Sensitivity curve (τ ∈ [0.1, 0.9])
- Expected outcome: At least one model exceeds random-chance detection rate

## E03: Augmentation Ablation
- Tests: Which augmentations improve performance and by how much?
- Setup: Best-performing model from E01 retrained 7 times with individual augmentations disabled
- Metrics: mAP delta per ablation
- Output: Ablation table mapping augmentation → performance contribution

## E04: Temporal Leakage Quantification
- Tests: C03 — how much does random splitting inflate metrics?
- Setup: Train best model on random split. Compare mAP against clip-level split.
- Expected outcome: Random split produces 15--25% inflated mAP

## E05: Anchor Ablation (Faster R-CNN Only)
- Tests: C04 — do domain-specific anchors beat COCO anchors?
- Setup: Faster R-CNN trained twice: once with custom k=5 anchors, once with default COCO anchors
- Metrics: mAP delta
- Expected outcome: Custom anchors yield measurable improvement

## E06: Feature-Level Analysis
- Tests: Provides mechanistic evidence for why transfer succeeds or fails
- Procedure: Extract backbone features from smoke and fire bounding boxes. Compute cosine similarity, t-SNE, Grad-CAM attention maps.
- Output: Feature space visualization, per-layer similarity heatmap, attention overlay on fire images
