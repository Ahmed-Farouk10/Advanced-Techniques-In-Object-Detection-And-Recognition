# Advanced Data Cleaning Research for Object Detection Test Probes

> **Module:** Dataset A — Task 3 Data Cleaning Research

---

## 1. Zero-Imputation Policy in Evaluation Sets

In academic research, test set cleaning must follow a strict **zero-imputation policy**:
1. Never fabricate or hallucinate bounding boxes for missing annotations.
2. Never delete ground-truth fire annotations because they are small, blurry, or difficult.
3. Every modification (e.g., coordinate clipping of points outside $[0,1]$ by $<2\%$) must be logged with an explicit mathematical justification in `cleaning_log.csv`.

---

## 2. The 7-Part Cleaning Pipeline on Dataset A

| Step | Technique | Purpose | Result on Dataset A |
| :--- | :--- | :--- | :--- |
| **Part 1** | PIL File Verification & Stem Match | Detect 0-byte corrupt images or dangling labels | **637 / 637 Valid** (0 corrupt) |
| **Part 2** | MD5 Exact Deduplication | Flag bit-for-bit identical test images | 0 duplicate files |
| **Part 3** | Laplacian Variance Blur Analysis | Flag motion blur ($\text{Var} < 100$) | 42 blurry frames logged |
| **Part 4** | Percentile Box Area Outliers | Flag degenerate $(<0.5\%)$ and screen-filling $(>99.5\%)$ boxes | 12 outlier boxes logged |
| **Part 5** | YOLO Coordinate Validation | Clip out-of-bounds coordinates to $[0, 1]$ | 8 coordinate adjustments logged |
| **Part 6** | Perceptual Hash (pHash) Clustering | Flag redundant video frames (Hamming distance $< 4$) | 112 near-duplicate frames logged |
| **Part 7** | Audit Trail Serialization | Export all actions to CSV | **174 entries logged** in `cleaning_log.csv` |
