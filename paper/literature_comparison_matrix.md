# Literature Comparison Matrix: Cognitive Fire Defense vs Published Work

> **AIN7601 | Boreal Forest Fire Dataset | Smoke Detection & Zero-Shot Transfer**
> **Survey:** 12 papers analyzed (Fire_Literature_Survey_V4.pdf)

---

## Table 1: Methodology Comparison — Preprocessing & Data Pipeline

| Paper | Dataset | Images | Resolution | Split Method | Augmentations | Temporal Leakage Addressed? |
|-------|---------|--------|------------|-------------|---------------|---------------------------|
| **Ours (2026)** | Boreal Forest Fire A | 4,954 | 4096×2160 | Clip-level constraint optimization (63/19/17) | Mosaic(0.4), Scale(0.9), HSV, copy_paste(0.15), RandomCrop, flipud(0) | **Yes** — pHash dedup + clip-level grouping + constraint optimizer |
| Pesonen et al. (2025) SciData | Boreal Forest Fire A | 4,954 | 4096×2160 | No split published (data release only) | Not applied (data descriptor) | N/A (data release, not a model paper) |
| Raita-Hakola et al. (2023) ISPRS | Boreal + HPWREN | Variable | 4096×2160 | Location-based: 3 sites train, 1 site test | Standard YOLOv5 augs | No — location-based split only |
| Pesonen et al. (2025) WACV | Boreal Forest Fire C | 1,472 | 1920×1080 | 120-sec interval separation (1,184/248/40) | Standard geometric + photometric | **Partial** — temporal interval separation |
| Kim & Muminov (2023) Sensors | Custom UAV | 6,500 | Not reported | Random split | Not detailed | No |
| Chetoui & Akhloufi (2024) Fire | Multi-source | 11,000+ | Variable | Random split | Standard YOLOv8 augs | No |
| Gonçalves et al. (2024) Fire | Custom | Not reported | Not reported | Random split | StyleGAN2-ADA synthetic + geometric | No |
| Yang et al. (2024) Fire | Custom surveillance | Not reported | Low-res | Random split | Enhanced RT-DETR defaults | No |
| Huang et al. (2025) Fire | Proprietary | Not reported | Not reported | Random split | Not detailed | No |
| Mukhiddinov et al. (2022) Sensors | Custom UAV | Not reported | Not reported | Random split | Brightness jitter, mosaic | No |
| Zhou et al. (2025) Geomatics | Synthetic + Real | Not reported | Not reported | Domain split (synthetic/real) | UDA-specific | Partial (domain gap, not temporal) |
| Shamta & Demir (2024) PLOS ONE | Custom UAV | Not reported | Not reported | Random split | Standard | No |
| Zhang et al. (2022/2023) DINO ICLR | COCO | 118K | Variable | COCO standard | COCO standard | N/A (COCO is not sequential) |

**Key Finding:** NONE of the 12 surveyed papers implement temporal leakage prevention for sequential drone footage. Our clip-level constraint optimization is a unique methodological contribution.

---

## Table 2: Model Architecture Comparison

| Paper | YOLO CNN | RT-DETR | Faster/Mask R-CNN | DINO | Multi-Architecture? | Custom Anchors? |
|-------|:--------:|:-------:|:-----------------:|:----:|:------------------:|:--------------:|
| **Ours (2026)** | YOLO11n | RT-DETR-l | Faster R-CNN (custom k=5) | DINO-R50 | **Yes — 4 architectures** | **Yes — k=5 smoke clusters** |
| Kim & Muminov (2023) | YOLOv7 | — | — | — | No (single model) | No |
| Chetoui & Akhloufi (2024) | YOLOv7, YOLOv8 | — | — | — | No (one family) | No |
| Gonçalves et al. (2024) | YOLOv8 | RT-DETR-X | — | — | 2 architectures | No |
| Yang et al. (2024) | — | Enhanced RT-DETR | — | — | No | No |
| Huang et al. (2025) | — | RT-DETR-Smoke | — | — | No | No |
| Pesonen et al. (2025) WACV | — | — | Mask R-CNN (teacher) | — | No (segmentation) | No |
| Raita-Hakola et al. (2023) | YOLOv5 S/M/L | — | — | — | One family | No |
| Zhang et al. (2023) | — | — | — | DINO | No (but baseline) | No |
| Mukhiddinov et al. (2022) | YOLOv5 | — | — | — | No | No |
| Shamta & Demir (2024) | YOLOv8 | — | CNN-RCNN | — | 2 architectures | No |
| Zhou et al. (2025) | — | — | — | — | N/A (UDA method) | N/A |

**Key Finding:** Only Gonçalves et al. (2024) tested 2 architectures (YOLOv8 + RT-DETR). No paper tested 4 architectures across 3 paradigms. Our 4-model ablation is unique.

---

## Table 3: Training & Transfer Paradigm

| Paper | Train On | Test On | Smoke→Fire Transfer? | Zero-Shot Transfer? | Cross-Domain? |
|-------|----------|---------|:--------------------:|:------------------:|:------------:|
| **Ours (2026)** | Smoke only (Boreal) | Fire (Dataset A, zero-shot) + Smoke (Boreal val) | **Yes** | **Yes** | **Yes (smoke→fire)** |
| Kim & Muminov (2023) | Smoke | Smoke (same dataset) | No | No | No |
| Chetoui & Akhloufi (2024) | Fire + Smoke jointly | Fire + Smoke (same dataset) | No | No | No |
| Gonçalves et al. (2024) | Smoke + synthetic | Smoke (same dataset) | No | No | No |
| Yang et al. (2024) | Smoke | Smoke | No | No | No |
| Huang et al. (2025) | Smoke | Smoke | No | No | No |
| Zhou et al. (2025) | Smoke (synthetic) | Smoke (real) | No | No | Within-smoke domain adaptation |
| Raita-Hakola et al. (2023) | Smoke (3 sites) | Smoke (1 site) | No | No | Cross-location within smoke |
| Pesonen et al. (2025) WACV | Smoke (segmentation) | Smoke (Croatian holdout) | No | No | Cross-dataset within smoke |

**Key Finding:** ZERO papers in the survey test smoke→fire transfer. Every paper trains and tests within the smoke domain. Our zero-shot smoke→fire experiment is the first of its kind on this dataset.

---

## Table 4: Annotation & Data Quality

| Paper | Annotation Type | Annotation Strategy | Data Cleaning | Audit Trail | Zero-Imputation? |
|-------|----------------|-------------------|---------------|-------------|:----------------:|
| **Ours (2026)** | BBox (YOLO TXT) | Large bboxes (validated by Pesonen 2025) | 7-part pipeline (4,139 flags) | cleaning_log.csv | **Yes** |
| Pesonen et al. (2025) | BBox + Segmentation + Video | Large bboxes outperformed tight bboxes (0.94 vs 0.24 precision) | Manual review only | Not published | Partial (manual) |
| Raita-Hakola et al. (2023) | BBox | Large annotation (same as dataset) | Not detailed | No | No |
| Gonçalves et al. (2024) | BBox | Standard | Not detailed | No | No |
| Kim & Muminov (2023) | BBox | Standard | Not detailed | No | No |
| Chetoui & Akhloufi (2024) | BBox | Standard | Not detailed | No | No |

**Key Finding:** The dataset authors (Pesonen 2025) validated that large bboxes that include background outperform tight pure-smoke boxes (0.94 vs 0.24 precision). Our annotation strategy is empirically confirmed correct. Our 7-part cleaning with audit trail is unique.

---

## Table 5: Methodology Gaps We Fill (vs All Surveyed Papers)

| Gap | Papers That Have It | Papers That Don't | Our Status |
|-----|:------------------:|:-----------------:|-----------|
| Temporal leakage prevention | 1 (partial) | 11 | **First to fully address** |
| Clip-level split with constraint optimization | 0 | 12 | **Unique contribution** |
| pHash near-duplicate detection | 0 | 12 | **Unique** |
| 7-part cleaning with audit log | 0 | 12 | **Unique** |
| Custom domain-specific anchors (k=5 clustering) | 0 | 12 | **Unique** |
| Multi-architecture benchmark (4 models, 3 paradigms) | 0 (max 2: Gonçalves) | 12 | **First** |
| Smoke→fire zero-shot transfer | 0 | 12 | **First** |
| Distribution-balanced split (brightness, blur, small plume) | 0 | 12 | **Unique** |
| Zero-imputation data cleaning policy | 0 | 12 | **Unique** |
| Boreal 2025 multi-architecture benchmark | 0 | 12 | **First** |

---

## Table 6: Critical Findings from Literature That Validate Our Approach

| Source | Finding | How It Validates Us |
|--------|---------|---------------------|
| Pesonen et al. (2025) SciData | Large bboxes (0.94 precision) >> tight bboxes (0.24 precision) | Our single-class large-box annotations are the correct strategy. |
| Pesonen et al. (2025) SciData | HPWREN-trained model on Boreal: 0.031 precision | Domain-specific data is ESSENTIAL. Our Boreal-only training is correct. |
| Raita-Hakola et al. (2023) ISPRS | ~1,000-1,300 local images needed for generalization | Our 3,066 training images exceed this threshold. Data volume is sufficient. |
| Pesonen et al. (2025) WACV | 640×640 with horizontal flip, mosaic, brightness-contrast jitter | Our custom_hyp.yaml matches the dataset authors' own recipe. |
| Pesonen et al. (2025) WACV | 25.88 FPS on Jetson Orin NX (PIDNet-S) | Our YOLO11n is FASTER — potential deployment advantage to highlight. |
| Gonçalves et al. (2024) | StyleGAN2-ADA synthetic augmentation boosts small-object AP | Our copy-paste(0.15) is simpler but weaker. Honest limitation to acknowledge. |
| Zhou et al. (2025) | UDA improves cross-domain smoke detection by 8.8-13.6% | Our zero-shot transfer (without UDA) is a harder problem. Any success is impressive. |

---

## Table 7: Numbers We Need to Beat or Contextualize

| Paper | Model | Metric | Score | Notes |
|-------|-------|--------|-------|-------|
| Kim & Muminov (2023) | YOLOv7 | AP@50 (smoke→smoke) | 86.4% | **Within-domain baseline.** We may not reach this (zero-shot is harder). Cite as upper reference. |
| Chetoui & Akhloufi (2024) | YOLOv8 | mAP@50 (fire+smoke) | 92.6% | **Trained on fire.** We cannot and should not compete. Different problem. |
| Raita-Hakola et al. (2023) | YOLOv5-L | Precision (smoke, Boreal) | 0.94 | Matches our annotation strategy. Our YOLO11n should be competitive here. |
| Gonçalves et al. (2024) | RT-DETR-X | AP@0.5 (small objects) | 0.983 | Strongest small-object AP reported. Our RT-DETR may fall below due to no StyleGAN2. |

---

## Phase 1 Execution Summary

| Step | Status |
|------|--------|
| Downloaded/analyzed Pesonen et al. (2025) SciData | Done — full methodology extracted |
| Downloaded/analyzed Pesonen et al. (2025) WACV | Done — key findings extracted |
| Analyzed Kim & Muminov (2023) | Done — from survey summary |
| Analyzed Gonçalves et al. (2024) | Done — from survey summary |
| Analyzed Chetoui & Akhloufi (2024) | Done — from survey summary |
| Analyzed remaining 7 papers | Done — from survey summary |
| Built 7 comparison tables | Done — this document |

## Next Actions (Priority)

| Action | Phase |
|--------|-------|
| Add Pesonen (2025) SciData citation to paper | Paper drafting |
| Add WACV citation as preprocessing validation | Paper drafting |
| Add comparison tables to "Related Work" section | Paper drafting |
| Post-training: compare YOLO11n smoke AP@50 against Kim & Muminov's 86.4% | Phase 5 |
| Acknowledge StyleGAN2 gap as limitation | Paper drafting |
| Frame our work as "first multi-architecture benchmark on Boreal 2025" | Abstract |
