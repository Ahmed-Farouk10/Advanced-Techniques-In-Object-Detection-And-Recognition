# Rules.md — Q1 Journal Compliance Checklist

> **AIN7601 Cognitive Fire Defense Pipeline**
> **Target:** Q1 Journal (Remote Sensing MDPI, Fire MDPI, or equivalent)
> **Purpose:** Every rule must be satisfied before submission. No violations allowed.

---

## Section 1: Reproducibility (CRITICAL — Desk Reject If Missing)

| ID | Rule | Status | Action Required | Deadline |
|----|------|--------|-----------------|----------|
| R1.1 | Code publicly available on GitHub with Zenodo DOI | Not started | Archive repo + generate DOI via Zenodo after submission | Before submission |
| R1.2 | Exact split assignments released (image → train/val/test mapping) | Partial | Publish `split_assignments.json` mapping every image file to its split | Before training |
| R1.3 | All hyperparameters documented in config files (not inline code) | Partial | `custom_hyp.yaml` done. Missing: Faster R-CNN config, DINO config | Before training |
| R1.4 | Random seeds documented for all stochastic operations (Python, NumPy, PyTorch, data loading) | Not done | Document seeds used. Run 3-seed validation (Phase 8) | After training |
| R1.5 | Environment: `requirements.txt` with pinned versions, OR Dockerfile | Partial | Verify `requirements.txt` includes ultralytics, torch, torchvision, transformers, albumentations with versions | Before training |
| R1.6 | Dataset instructions: how to download, verify checksum, reproduce split | Not done | Add to README: Kaggle/Fairdata links + file count verification | Before submission |
| R1.7 | All `build_*.py` scripts that generate notebooks must be published | Done | `build_clean_nb.py`, `build_split_nb.py`, `build_train_nbs.py` in shared/ | Ongoing |
| R1.8 | No hardcoded absolute paths in published code | Partial | `smoke_data.yaml` uses absolute path. Fix to relative before release. | Before submission |

---

## Section 2: Boreal Dataset Compliance

| ID | Rule | Status | Action Required | Deadline |
|----|------|--------|-----------------|----------|
| R2.1 | Dataset cited correctly: Pesonen et al. (2025) Scientific Data, Vol 12, 1419 | Not done | Add to paper references and README | Before submission |
| R2.2 | Preprocessing follows authors' recommended settings (640×640, mosaic, horizontal flip, brightness-contrast jitter) — validated by Pesonen et al. (2025b) WACV | Done | Our `custom_hyp.yaml` matches. Cite WACV paper as validation. | Already done |
| R2.3 | Subset A limitation acknowledged: smoke bounding boxes only, no fire annotations, no segmentation masks | Not done | Add to paper limitations section | Paper drafting |
| R2.4 | Geographic limitation disclosed: 4 Finnish sites (Evo, Heinola, Karkkila, Ruokolahti) — boreal biome only | Not done | Add to paper limitations | Paper drafting |
| R2.5 | Controlled burn conditions disclosed (not real wildfire emergencies) | Not done | Add to paper methodology | Paper drafting |
| R2.6 | No redistribution of dataset images (license compliance) | Done | README states data not stored in repo. Download links provided. | Already done |
| R2.7 | Official dataset statistics (4,954 images, 4 locations, 30 clips) match our counts | Done | Our Task 2 EDA confirmed all counts | Already done |
| R2.8 | Dataset B's "Subset A" naming convention used consistently | Partial | Some docs call it "Boreal" — standardize | Before submission |

---

## Section 3: Methodology Integrity (Q1 Minimum Standard)

| ID | Rule | Status | Action Required | Deadline |
|----|------|--------|-----------------|----------|
| R3.1 | Train/val/test split performed BEFORE any data transformation or augmentation fitting | Done | Clip-level split (Task 4) done before any processing. Augmentations are on-the-fly, not pre-computed. | Already done |
| R3.2 | Validation set used ONLY for early stopping and hyperparameter monitoring, NEVER for model selection | Must enforce | Document: val used for loss monitoring + early stopping. Final model selection is done on val. Test set held out completely. | During training |
| R3.3 | Test set evaluated EXACTLY ONCE — after all 4 models are fully trained and hyperparameters locked | Must enforce | Test set not touched until all training + ablation complete. | Before evaluation |
| R3.4 | No test-time augmentation (TTA) unless explicitly reported and compared fairly across all models | Must document | Default: no TTA. If TTA used, apply uniformly across models and disclose. | Evaluation phase |
| R3.5 | Splits are at IMAGE level to prevent same-image objects leaking across splits | Exceeded | Clip-level split (stricter than image-level: prevents same-scene leakage) | Already done |
| R3.6 | Empty/negative images distributed proportionally across splits | Done | 256 background images distributed 70/15/15 | Already done |
| R3.7 | All data cleaning/filtering decisions logged with audit trail | Done | `cleaning_log.csv` (4,139 entries) with Step, Image, Issue Type, Detection Method, Treatment, Rationale | Already done |
| R3.8 | Zero-imputation policy: no hallucinated bounding boxes, no synthesized annotations | Done | 0 images deleted, 2 marginal coordinate auto-corrections (<2% margin) | Already done |

---

## Section 4: Comparison & Baseline Rules

| ID | Rule | Status | Action Required | Deadline |
|----|------|--------|-----------------|----------|
| R4.1 | Compare against published baselines that used the SAME dataset (Boreal Forest Fire Subset A) | Partial | Pesonen et al. (2025), Raita-Hakola et al. (2023) used Boreal data. Identify their metrics. Kim & Muminov, Chetoui & Akhloufi used different datasets — compare as within-domain baselines. | After training |
| R4.2 | NEVER compare our zero-shot smoke→fire numbers against papers trained on fire directly | Must enforce | Create separate table: "Within-Domain Comparison" vs "Cross-Domain Transfer." Never mix. | Paper drafting |
| R4.3 | Include naive lower bound (e.g., HSV color threshold, motion detector, or HOG+SVM) | Not done | Phase 10.3 — required for Q1 | Summer |
| R4.4 | Include supervised upper bound (model trained on fire directly, or pseudo-labeled on Dataset A) | Not done | Phase 10.3 — required for Q1 | Summer |
| R4.5 | If no published baseline exists on Boreal dataset, state: "We present the first multi-architecture benchmark on this dataset" | Verify | Literature survey confirms no prior multi-architecture benchmark. Statement is truthful. | Paper drafting |
| R4.6 | Report statistical significance (p-value or 95% CI) between model pairs | Not done | Phase 8: 3-seed runs needed for confidence intervals | Summer |

---

## Section 5: Reporting Standards

| ID | Rule | Status | Action Required | Deadline |
|----|------|--------|-----------------|----------|
| R5.1 | Report mAP@0.5 AND mAP@0.5:0.95 for all models | Pending training | Phase 5 evaluation | After training |
| R5.2 | Report AP_small, AP_medium, AP_large separately (critical for early detection thesis) | Pending training | Phase 5 evaluation | After training |
| R5.3 | Precision-Recall curves for all 4 models on same plot | Pending training | Phase 5 | After training |
| R5.4 | F1-score vs confidence threshold curve | Pending training | Phase 5 | After training |
| R5.5 | Model size (parameters, FLOPs) and inference speed (FPS, latency ms) | Pending training | Phase 5 — include in paper Table 4 | After training |
| R5.6 | Failure analysis: qualitative examples where each model fails, with explanation | Not done | Phase 7 | After training |
| R5.7 | Confusion matrix or error breakdown per model | Pending training | Phase 5 | After training |
| R5.8 | Training and validation loss curves for all models | Pending training | Phase 4 automatically saves | During training |
| R5.9 | All tables must be self-contained: captions explain FINDINGS, not just describe content | Must enforce | Paper drafting convention | Paper drafting |

---

## Section 6: Ethical & Broader Impact

| ID | Rule | Status | Action Required | Deadline |
|----|------|--------|-----------------|----------|
| R6.1 | Broader impact statement: climate change adaptation, wildfire prevention, environmental monitoring | Not done | Add to paper conclusion | Paper drafting |
| R6.2 | UAV/drone usage disclosed: controlled burns, NOT surveillance or military application | Not done | Add to methodology section | Paper drafting |
| R6.3 | No claim of "solving" wildfire detection — honest about limitations (daytime only, boreal only, controlled burns) | Must enforce | Limitations section already planned. Add these 3 specific limitations. | Paper drafting |
| R6.4 | Dual-use assessment: smoke detection has no meaningful weaponization risk (unlike e.g., facial recognition) — state this briefly | Not done | Add in broader impact paragraph | Paper drafting |
| R6.5 | Data ethics: dataset collected via authorized controlled burns by Finnish research institutions. No privacy concerns (unpopulated forest areas). | Not done | Add to methodology | Paper drafting |

---

## Section 7: Novelty Verification (Anti-Rejection)

| ID | Rule | Status | Action Required | Deadline |
|----|------|--------|-----------------|----------|
| R7.1 | Novelty claim verified against literature survey (12 papers analyzed) | Done | Survey confirms 3 gaps we fill | Already done |
| R7.2 | Novelty claim is SPECIFIC, not vague — "first multi-architecture benchmark on Boreal 2025 for smoke→fire zero-shot transfer" | Must enforce | Revise abstract to use this exact phrasing | Paper drafting |
| R7.3 | No overclaiming: we are NOT claiming SOTA fire detection | Must enforce | Add explicit statement: "We do not claim state-of-the-art fire detection, as our models are trained exclusively on smoke." | Paper drafting |
| R7.4 | Primary contribution is METHODOLOGICAL (pipeline rigor, temporal leakage prevention) + SCIENTIFIC (visual prototype transfer), not numerical mAP edge | Must enforce | Frame introduction and abstract around these two contributions | Paper drafting |
| R7.5 | Negative results reported honestly | Must enforce | If DINO fails on zero-shot transfer, report it as a finding about deformable attention overfitting to smoke texture | Phase 7 |
| R7.6 | Every contribution claim in the abstract is supported by an experiment in the paper | Must verify | Cross-check abstract vs results before submission | Paper drafting |

---

## Section 8: Anti-Rejection Checklist (Common Desk Reject Causes)

| ID | Rejection Cause | Prevention Strategy | Status |
|----|----------------|---------------------|--------|
| R8.1 | No code released | GitHub release + Zenodo DOI (R1.1) | Not done |
| R8.2 | Single dataset, no generalization evidence | Frame as "first benchmark on Boreal 2025" rather than "general solution." Add external validation in Phase 10.1. | Partial |
| R8.3 | No ablation study | 7 ablation studies designed (Phase 10.2) | Pending training |
| R8.4 | Overclaiming (e.g., "solves wildfire detection") | Explicit disclaimer: smoke-only training, daytime only, boreal only (R7.3) | Must enforce |
| R8.5 | No comparison to published baselines | Table comparing against Kim & Muminov, Chetoui & Akhloufi, Pesonen, Raita-Hakola (R4.1) | After training |
| R8.6 | Unclear contribution — "applied existing models to new data" | Contributions reframed as methodological + scientific, not applied (R7.4) | Paper drafting |
| R8.7 | Single seed, no error bars | 3-seed runs planned (R4.6) | Summer |
| R8.8 | Missing related work — survey is shallow | 16 citations in draft + 6 more from survey = 22 total | Paper drafting |
| R8.9 | Data leakage not addressed | Clip-level split with mathematical justification (R3.5) | Done |
| R8.10 | No ethical statement | Broader impact + drone ethics (R6.1-R6.5) | Paper drafting |
| R8.11 | Figures low quality or unreadable | Use vector format (PDF/EPS) for all plots. Minimum 300 DPI for raster. | Paper drafting |
| R8.12 | Self-plagiarism — copied text from own prior work | This is the first paper on this project. No prior publications. | Not an issue |
| R8.13 | Poor English — grammar/spelling errors | Proofread by a native English speaker before submission. Use Grammarly. | Before submission |
| R8.14 | Wrong journal scope — paper doesn't fit | Target MDPI Remote Sensing or Fire. Both publish wildfire detection. Confirm scope before submission. | Before submission |

---

## Compliance Summary

| Section | Total Rules | Passed | Pending | Failing |
|---------|-----------|--------|---------|---------|
| 1. Reproducibility | 8 | 2 | 6 | 0 |
| 2. Boreal Dataset | 8 | 4 | 4 | 0 |
| 3. Methodology Integrity | 8 | 7 | 1 | 0 |
| 4. Comparisons & Baselines | 6 | 1 | 5 | 0 |
| 5. Reporting Standards | 9 | 0 | 9 | 0 |
| 6. Ethics & Impact | 5 | 0 | 5 | 0 |
| 7. Novelty Verification | 6 | 2 | 4 | 0 |
| 8. Anti-Rejection | 14 | 2 | 11 | 1 |
| **TOTAL** | **64** | **18** | **45** | **1** |

**High-priority violations to fix NOW:**
- R8.4: Ensure no overclaiming in paper abstract
- R4.2: Separate within-domain vs cross-domain comparison tables
- R1.8: Fix absolute paths before code release
