# Dataset B Preprocessing — Change Report

## Overview

This report documents the preprocessing and data-splitting changes applied to Dataset B (Boreal Forest Fire Subset A).

The main objective was to establish a reliable training and validation pipeline while preventing temporal leakage caused by the sequential nature of the drone video frames.

---

## Task 3 — Data Cleaning

### Dataset Integrity

The working dataset was verified to contain:

- 4,954 images
- 4,954 corresponding labels
- 9,909 total files

All image-label pairs passed the initial file integrity checks.

### Exact Duplicate Detection

MD5 hashing was applied to identify byte-identical images.

Result:

- Images checked: 4,954
- Exact duplicate images detected: 0
- Unique images retained: 4,954

No files were removed during the MD5 check.

### Bounding Box Statistics

The dataset contains 4,862 annotated bounding boxes.

Computed statistics:

| Metric | Value |
|---|---:|
| Area P0.01 | 0.000288 |
| Area P99.5 | 0.896247 |
| AR Q1 | 0.7358 |
| AR Q3 | 1.0201 |
| AR Lower Bound | 0.3093 |
| AR Upper Bound | 1.4465 |

The area distribution was handled using percentile-based bounds because bounding-box areas are strongly right-skewed.

Aspect-ratio outliers were detected using IQR-based bounds.

### Cleaning Findings

| Issue | Count | Treatment |
|---|---:|---|
| Blurry images | 431 | Flagged |
| Aspect-ratio outliers | 368 | Flagged |
| Area outliers | 26 | Flagged |
| Box count shifts | 193 | Flagged |
| Near duplicates | 3,389 | Flagged |

Quality-related issues were flagged rather than automatically deleting potentially useful ground-truth data.

### Near-Duplicate Analysis

Perceptual hashing (pHash) was used to analyze consecutive frames.

Results:

- Consecutive comparisons: 4,923
- Minimum pHash distance: 0
- Maximum pHash distance: 42
- Mean distance: 5.35
- Median distance: 4.00
- Distance < 5: 3,389 frames

Near-duplicate frames were treated as temporal redundancy rather than corrupted data.

They were therefore flagged and handled through frame sampling rather than destructive deletion.

---

# Task 4 — Temporal Train/Validation Splitting

## Motivation

Dataset B consists of sequential drone video frames. Random image-level splitting can cause adjacent frames from the same flight to appear in both training and validation sets.

This creates temporal leakage because neighboring frames share highly similar:

- Background
- Camera viewpoint
- Illumination
- Forest structure
- Smoke appearance

Therefore, the split was performed at the clip level.

## Clip-Level Splitting

The dataset contains:

- 30 unique video clips
- 4 locations:
  - Evo
  - Heinola
  - Karkkila
  - Ruokolahti

No clip was divided between Train and Validation.

The target ratio was approximately:

- Train: 80%
- Validation: 20%

Because clips are indivisible blocks, the final ratio does not exactly equal 80/20.

## Final Split

| Split | Clips | Frames | Ratio |
|---|---:|---:|---:|
| Train | 22 | 3,613 | 79.25% |
| Validation | 8 | 946 | 20.75% |
| Total retained | 30 | 4,559 | 100% |

All four locations are represented in both Train and Validation.

### Small-Plume Distribution

| Split | Clips containing small plumes |
|---|---:|
| Train | 9 |
| Validation | 1 |

Small-plume presence was explicitly considered during split optimization because early smoke detection is a primary objective of the project.

---

## Frame Sampling

A maximum of 300 frames per clip was introduced:

```text
MAX_FRAMES_PER_CLIP = 300
