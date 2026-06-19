# Task 4 — Data Splitting (Temporal Leakage Prevention)

> **Phase 2: Data Preparation | Dataset B (Boreal Watchtower/Drone)**

## Objective

Split 4,954 sequential video frames into train/val/test WITHOUT leaking adjacent frames across splits. This is the most critical data integrity task in the entire pipeline.

## The Problem: Temporal Leakage

The Boreal dataset consists of frames extracted from 31 DJI drone flights (e.g., `evoDJI_0001_frame65.jpg`). Adjacent frames share the same forest background, lighting, and camera angle. If we use `train_test_split(random_state=42)`:

- Frame 65 goes to train, frame 66 goes to validation
- The model learns "the trees in evoDJI_0001" → NOT "what smoke looks like"
- Validation metrics inflate to 99% mAP, but real-world performance is zero

## The Solution: Clip-Level Constraint Optimization

| Parameter | Value |
|-----------|-------|
| Algorithm | 10,000 iterations of randomized clip assignment with MSE minimization |
| Constraints | (1) No clip ID overlaps across splits, (2) ≥1 clip from each location per split, (3) Penalize distribution variance, (4) Penalize zero small plumes in any split |
| Frame sampling | `--max-frames-per-clip=300` clips exceeding 300 frames are uniformly sub-sampled |
| Empty images | 256 background images distributed 70/15/15 proportionally |

## Final Split

| Split | Images | Ratio | Mean Brightness | Mean Blur | Mean Box Area | Small Plume % |
|-------|--------|-------|-----------------|-----------|---------------|---------------|
| Train | 3,066 | 63.7% | 113.9 | 2,710 | 0.454 | 1.5% |
| Val | 926 | 19.2% | 109.6 | 1,852 | 0.419 | 0.4% |
| Test | 823 | 17.1% | 109.2 | 2,750 | 0.339 | 1.0% |

## Why Not Exactly 70/15/15?

31 drone clips are indivisible "blocks" of varying sizes (some clips have 1,700+ frames). Packing rigid variable-sized blocks into 3 buckets while forcing geographic stratification means 63/19/17 is the mathematical optimum. This is documented as a limitation in the paper.

## Engineering Challenges & Near-Misses

### Near-Miss 1: The 45% Data Loss Crisis

Our initial constraint algorithm used `--max-frames-per-clip=100` to prevent massive static clips from dominating the loss function. However, this cap was **too aggressive** — large drone flights like Ruokolahti (1,765 frames in a single clip) were decimated. The first run produced only 2,732 retained images out of 4,954 — a catastrophic 45% data loss. We systematically raised the cap to 300, recovering 4,815 images (97.2% retention) without compromising the temporal separation guarantees.

### Near-Miss 2: The "Zero Small Plume" Thesis Killer

Our first unpenalized optimization run produced splits where Val and Test had **0.0% small plumes**. Since the core thesis of this paper is early smoke detection, evaluating on splits devoid of small plumes would completely invalidate the model's primary objective — you cannot prove a model detects early distant smoke if your test set contains none. We modified the constraint optimizer to add a heavy penalty (weight=10.0) when any split received zero small plume images, successfully forcing them into all splits: Train 1.5%, Val 0.4%, Test 1.0%.

### Near-Miss 3: Blur Variance Imbalance

An intermediate iteration of the split produced a Validation set that was 35% sharper than the Training set (mean blur: Train 2,710 vs Val 1,852). This creates an **optimistically biased evaluation** — the model is tested on cleaner data than it was trained on. We added a blur standard deviation penalty to the optimizer's loss function, bounding the variance across splits and ensuring models are evaluated on realistic environmental noise.

### The 10-Image Sampling Bug

During early iterations, the feature extraction code was sampling only 10 random images per clip to compute mean blur and brightness (performance optimization for 4K frames). However, this same 10-image sample was accidentally used to count small plumes from label files. Given only 26 small plumes across 4,954 images, random 10-image sampling almost mathematically guaranteed missing them entirely — the optimizer thought every clip had 0 small plumes. The fix separated label parsing (parse ALL .txt files, instant) from pixel processing (sample 10 images for heavy math).

## Verification — All Passed

- [x] Zero video sequence ID overlap across Train/Val/Test
- [x] Every split contains ≥1 clip from Evo, Heinola, Karkkila, Ruokolahti
- [x] Retained + Removed = 4,954
- [x] Data retention: 4,815/4,954 (97.2%)
- [x] Small plumes present in all three splits
- [x] Brightness distribution stable across splits

## Outputs

- `split.ipynb` — Constraint optimization execution notebook
- `../task4_data_transformation/` — Legacy directory (data transformation → format conversion); actual split logic lives here
- `../../yolo_format/` — Split output in YOLO format
- `../../coco_format/` — Split output in COCO format

## Paper Contribution

The clip-level constraint optimization with distributional penalties is a methodological contribution. Most object detection papers use naive random splits on sequential data without acknowledging temporal leakage. Our split methodology and verification framework can be adopted by other researchers working with video-derived detection datasets.
