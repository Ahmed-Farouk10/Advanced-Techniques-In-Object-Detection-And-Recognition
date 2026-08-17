# 📜 Dataset A Preprocessing Changelog

> **Module:** `dataset-a/preprocessing/`  
> **Target:** Fire & Smoke Detection Testing Probe

---

## **v1.0.0 (Standardized 6-Task Architecture Complete)**
* **Task 1 (Business Logic):** Formulated asymmetric cost matrix ($\text{Cost}(\text{FN}) \gg \text{Cost}(\text{FP})$) and established two-tier evaluation criteria (Strict $\text{IoU} \ge 0.50$ vs Early-Warning Alerting $\text{IoU} \ge 0.10$).
* **Task 2 (Data Understanding):** Completed comprehensive EDA on 637 probe images, isolating 459 fire-positive images, 995 ground-truth fire boxes, and 178 hard background negatives.
* **Task 3 (Data Cleaning & Audit):** Implemented strict 7-part cleaning with zero-imputation policy, generating [`cleaning_log.csv`](task3_data_cleaning/cleaning_log.csv) with 174 tracked anomaly events.
* **Task 4 (Data Transformation):** Standardized all images to $640\times640$ resolution in `dataset-a/processed/images/` and generated verified `test_coco.json` with 1,891 bounding boxes in `dataset-a/processed/annotations/`.
* **Task 5 (Feature Engineering):** Defined evaluation normalization transforms and atmospheric distortion test configurations.
* **Task 6 (Feature Selection):** Extracted $k=5$ K-means fire anchor priors ($[0.0128, 0.0447, 0.0920, 0.1210, 0.2709]$) and analyzed cross-domain scale mismatch against Dataset B smoke anchors.
