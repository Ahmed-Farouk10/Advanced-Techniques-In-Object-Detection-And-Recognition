# Task 4 — Data Splitting (Temporal Leakage Prevention)

> **Phase 2: Data Preparation | Dataset B (Boreal Watchtower/Drone)**

## Objective

Split 4,954 sequential video frames into train/validation WITHOUT leaking adjacent frames across splits. This is the most critical data integrity task in the entire pipeline.

## The Problem: Temporal Leakage

The Boreal dataset consists of frames extracted from 30 DJI drone flights (e.g., `evoDJI_0001_frame65.jpg`). Adjacent frames share the same forest background, lighting, and camera angle. If we use `train_test_split(random_state=42)`:

- Frame 65 goes to train, frame 66 goes to validation
- The model learns "the trees in evoDJI_0001" → NOT "what smoke looks like"
- Validation metrics inflate to 99% mAP, but real-world performance is zero

## The Solution: Clip-Level Constraint Optimization

| Parameter | Value |
|-----------|-------|
| Algorithm | 10,000 iterations of randomized clip assignment with MSE minimization |
| Constraints | (1) No clip ID overlaps across splits, (2) ≥1 clip from each location per split, (3) Penalize distribution variance, (4) Penalize zero small plumes in any split |
| Frame sampling | `--max-frames-per-clip=300` clips exceeding 300 frames are uniformly sub-sampled |
| Empty images | 256 background images distributed 80/20 proportionally |

## Final Split (80/20)

| Split | Images | Ratio |
|-------|--------|-------|
| Train | 3,825 | 79.4% |
| Validation | 990 | 20.6% |

## Why No Test Split?

A dedicated within-domain test split is intentionally omitted. The rationale:

1. **Within-domain smoke performance** is reported on the validation split (984 images, large enough for statistical confidence).
2. **Cross-domain zero-shot transfer** — the paper's core contribution — is evaluated on a separate, never-seen fire dataset (the Fire-and-Smoke-Detection-Dataset, 637 held-out test images). This is the true "test" for the transfer hypothesis.

A within-domain test split would be redundant for the transfer experiment and would reduce training data. The 80/20 split maximizes training data while preserving a valid within-domain validation benchmark.

## Why Not Exactly 80/20?

30 drone clips are indivisible "blocks" of varying sizes (some clips have 1,700+ frames). Packing rigid variable-sized blocks into 2 buckets while forcing geographic stratification means 79/21 is the mathematical optimum. This is documented as a limitation in the paper.

## Engineering Challenges & Near-Misses

### Near-Miss 1: The 45% Data Loss Crisis

Our initial constraint algorithm used `--max-frames-per-clip=100` to prevent massive static clips from dominating the loss function. However, this cap was **too aggressive** — large drone flights like Ruokolahti (1,765 frames in a single clip) were decimated. The first run produced only 2,732 retained images out of 4,954 — a catastrophic 45% data loss. We systematically raised the cap to 300, recovering 4,815 images (97.2% retention) without compromising the temporal separation guarantees.

### Near-Miss 2: The "Zero Small Plume" Thesis Killer

Our first unpenalized optimization run produced a validation split with **0.0% small plumes**. Since the core thesis of this paper is early smoke detection, evaluating on a split devoid of small plumes would completely invalidate the model's primary objective — you cannot prove a model detects early distant smoke if your validation set contains none. We modified the constraint optimizer to add a heavy penalty (weight=10.0) when any split received zero small plume images, successfully forcing them into both splits.

### Near-Miss 3: Blur Variance Imbalance

An intermediate iteration of the split produced a Validation set that was 35% sharper than the Training set. This creates an **optimistically biased evaluation** — the model is tested on cleaner data than it was trained on. We added a blur standard deviation penalty to the optimizer's loss function, bounding the variance across splits and ensuring models are evaluated on realistic environmental noise.

### The 10-Image Sampling Bug

During early iterations, the feature extraction code was sampling only 10 random images per clip to compute mean blur and brightness (performance optimization for 4K frames). However, this same 10-image sample was accidentally used to count small plumes from label files. Given only 26 small plumes across 4,954 images, random 10-image sampling almost mathematically guaranteed missing them entirely — the optimizer thought every clip had 0 small plumes. The fix separated label parsing (parse ALL .txt files, instant) from pixel processing (sample 10 images for heavy math).

## Verification — All Passed

- [x] Zero video sequence ID overlap across Train/Validation
- [x] Every split contains ≥1 clip from Evo, Heinola, Karkkila, Ruokolahti
- [x] Retained + Removed = 4,954
- [x] Data retention: 4,815/4,954 (97.2%)
- [x] Small plumes present in both splits
- [x] Brightness distribution stable across splits

## Outputs

- `split.ipynb` — Constraint optimization execution notebook
- `../task4_data_transformation/` — Legacy directory (data transformation → format conversion); actual split logic lives here
- `../../yolo_format/` — Split output in YOLO format
- `../../coco_format/` — Split output in COCO format

## Paper Contribution

The clip-level constraint optimization with distributional penalties is a methodological contribution. Most object detection papers use naive random splits on sequential data without acknowledging temporal leakage. Our split methodology and verification framework can be adopted by other researchers working with video-derived detection datasets.

## Decision Note (2026-08-13)

The team settled on a **unified 80/20 split** (matching Esraa's training runs) to simplify the publication narrative: within-domain smoke mAP is reported on the validation split, while the external fire dataset serves as the zero-shot transfer test set. The earlier 70/15/15 split (with a dedicated test set) was superseded by this decision.
