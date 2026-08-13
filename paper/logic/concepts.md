# Key Concepts — Cognitive Fire Defense

## Visual Prototype Transfer
The hypothesis that object detectors learn a visual prototype from training data that can transfer to semantically related but visually distinct phenomena. Smoke and fire share low-level features (turbulent fluid motion, semi-transparency, upward trajectory, irregular boundaries) despite differing in color, temperature, and chemical composition. If transfer occurs, it is because the backbone encodes these shared features.

## Temporal Leakage
The inflation of validation metrics caused by placing adjacent frames from a continuous video into different dataset splits. The model learns to recognize static scene context (trees, horizon, camera angle) rather than the dynamic object of interest (smoke). Detected frame-to-frame displacement in our data: mean 0.041 normalized units, 95.2% of frames >0.005 displacement.

## Constraint Optimization Split
A formulation of dataset splitting as a constrained optimization problem over 31 indivisible video clips. Minimizes ratio deviation from 70/15/15 while penalizing distribution imbalance (blur variance, small plume absence) and enforcing geographic stratification (≥1 clip per location per split).

## Perceptual Hash (pHash)
A hashing algorithm where visually similar images produce similar bit strings, unlike cryptographic hashes where one bit difference produces entirely different output. pHash Hamming distance <5 detects functionally identical frames that differ by 1-2 pixels due to drone vibration, lighting flicker, or JPEG compression.

## Zero-Imputation Policy
The rule that no bounding box annotation may be synthesized or hallucinated. Unlike tabular data where missing values can be imputed from distributional statistics, a missing spatial annotation represents missing ground truth that cannot be recovered. All cleaning operations either delete corrupt files or flag issues without modifying ground truth.

## Anchor Clustering
k-means clustering applied to the (width, height) dimensions of all bounding boxes to find the optimal anchor sizes for a specific detection domain. Our k=5 clusters produce anchors matching the observed smoke size distribution (areas: 0.03 to 0.73) rather than the COCO default (areas: 0.03 to 0.50). Only used by Faster R-CNN; YOLO11n, RT-DETR, and DINO are anchor-free.

## Distribution Imbalance Penalty
Terms added to the split optimizer's loss function that penalize undesirable distributions: blur standard deviation across splits (prevents evaluation on unrepresentatively clean data) and zero-small-plume penalty (weight=10.0, prevents the early-detection evaluation from being impossible).

## Single-Class Design
Training models on a single class (smoke) rather than joint training on smoke+fire or multi-class smoke/fire/background. This isolates the transfer experiment: any performance on fire comes from visual generalization, not from labeled fire training data.
