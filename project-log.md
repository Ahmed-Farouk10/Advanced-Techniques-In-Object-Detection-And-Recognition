# AIN7601 Project Log

> **Living Project Docstrings:** This file tracks every major action, configuration change, and adaptation made by the AI agents on this project.

## Log Entries

### [2026-06-14] Project Manager Initialization & Directory Restructure
**Action Taken:**
- Migrated legacy `data/` and `results/` structures into a highly organized, dataset-first architecture (`dataset-a` and `dataset-b`).
- Enforced a uniform 6-Task Preprocessing Pipeline inside both dataset directories.
- Downloaded `alik05/forest-fire-dataset` into `dataset-a/raw/Forest Fire Dataset/`.
- Created custom AG Kit Agent (`ain7601-project-manager.md`) and Pipeline Skill (`ain7601-pipeline/SKILL.md`).
- Initialized persistent memory (`MEMORY.md`) to guarantee all AI sessions address the user as "Ahmed" and enforce project constraints.

**Outcome / Lessons Learned:**
- Directory limits and GitHub constraints handled via `.gitignore`.
- Raw drone data isolated successfully. Ready for Task 1 execution.

### [2026-06-14] Task 1 Execution (Dataset A) & A+ Student Rules
**Action Taken:**
- Upgraded `ain7601-project-manager.md` with "A+ Student" philosophy (Explain WHY, link to business impact, show critical thinking).
- Executed Task 1 by generating `dataset-a/preprocessing/task1_business_logic/business_logic.md`.
- Defined drone battery constraints, max-recall requirement for life-safety, and edge throughput limits.
- Translated ML Task: Object Detection with bounding box dependency.

**Outcome / Lessons Learned:**
- By forcing a focus on Business Logic, we established that Dataset A *must* have bounding boxes to be viable, and our metric tuning must prioritize Recall over Precision due to the severe asymmetric cost of false negatives in forest fires.

### [2026-06-14] Task 2 Execution (Dataset A) & Fatal Flaw Discovery
**Action Taken:**
- Executed custom EDA script (`shared/eda_dataset_a.py`) against `dataset-a/raw/Forest Fire Dataset/`.
- Verified class distribution (balanced 760/760) and image dimensions (~250x250).
- Documented findings in `dataset-a/preprocessing/task2_data_understanding/data_understanding.md`.

**Outcome / Lessons Learned:**
- **CRITICAL FLAW:** Discovered Dataset A contains ZERO bounding box annotations. It is an Image Classification dataset, not an Object Detection dataset.
- **A+ Philosophy Applied:** Instead of blindly passing the data to Task 3, we linked the missing bounding boxes to the business constraint (Drones must localize fires using coordinates).
- **Adaptation Required:** We must now choose to either Manually Annotate, Pseudo-Label via Zero-Shot model, or pivot to a new Dataset A. This showcases true "solutions over models" thinking.

### [2026-06-18] Phase 0 Execution & Strategic Pivot
**Action Taken:**
- Analyzed the mismatch between the original plan (Dataset A detection, Dataset B detection) and the actual data (Dataset A classification, Dataset B detection).
- Proposed and locked a major strategic pivot: **Cross-Domain Smoke→Fire Transfer Learning**.
- Rewrote `forest-fire-detection.md` entirely to document the new 4-model zero-shot transfer plan.
- Rewrote `README.md` to reflect the updated architecture.
- Created `premortem.md` outlining the critical risk of temporal leakage in Dataset B's sequential video frames, and defined the required test cases.
- Updated `business_logic.md` for both Dataset A (Zero-Shot Probe role) and Dataset B (Smoke Training role).
- Updated agent `SKILL.md` and `ain7601-project-manager.md` to enforce the new structural roles and constraints.

**Outcome / Lessons Learned:**
- **A+ Philosophy Applied:** The project is now fundamentally more robust and academically novel. Instead of two unrelated detection benchmarks, it is a cohesive experiment proving that models can learn fire semantics purely from smoke pre-cursors.
- Phase 0 is complete. Next is Phase 1 (Dataset B EDA & Cleaning).

### [2026-06-18] Task 1 & 2 Execution (Dataset B)
**Action Taken:**
- Wrote and executed `shared/eda_dataset_b.py` to parse all 4,954 images and YOLO bounding boxes.
- Created `dataset-b/preprocessing/task2_data_understanding/data_understanding.md`.
- Evaluated bounding box areas and discovered a massive bias: 95.7% of smoke plumes take up >10% of the image.
- Identified sequential video frame naming (e.g., `evoDJI_0001_frame...`), proving temporal leakage risk.

**Outcome / Lessons Learned:**
- **A+ Philosophy Applied:** Linked the large plume bias directly to the business problem (watchtowers need to detect *early, distant* smoke, which is small). Decided to heavily rely on scale/crop augmentations in Task 5 to simulate distant smoke.
- Mapped the video sequence discovery to Task 4 (Splitting), setting a hard rule that splits must happen at the video-clip level, not random frame level.
- Rewrote `explore.ipynb` in `task2_data_understanding/` to programmatically build a notebook containing all Dataset B EDA code, integrated with A+ business insight markdown cells.

### [2026-06-18] Advanced Task 2 Deep Dive (12GB Pixel Analysis)
**Action Taken:**
- Ran `shared/advanced_eda.py` to analyze spatial coordinates, image resolutions, and pixel illumination across the 12GB dataset.
- Generated `advanced_eda_plots.png` showing the spatial horizon bias and brightness distribution.
- Appended findings to `data_understanding.md`.

**Outcome / Lessons Learned:**
- **A+ Philosophy Applied:** Discovered that images are 4K (4096x2160). Linked this to the massive risk of downsampling small plumes into oblivion if resized to 640x640. Mandated Random Cropping for Task 5.
- Identified that smoke primarily appears in the top 40% (horizon). Mandated disabling Vertical Flip augmentations to preserve physical reality.
- Identified daytime illumination bias. Mandated brightness/contrast jittering.

