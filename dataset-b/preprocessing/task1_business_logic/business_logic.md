# Task 1: Business Logic & Constraints (Dataset B)

> **Dataset B (Boreal Forest Watchtower Imagery)**  
> **Phase:** Problem Understanding & Business Translation

## 1. The Business Problem (WHY)
Detecting a fire once flames are visible is often too late to prevent widespread destruction. The true goal of an early-warning system is to detect the **precursor** to fire: Smoke.

By deploying object detection models on footage from stationary watchtowers in the Boreal forest, we aim to detect smoke plumes at varying distances.

## 2. ML Translation
- **Business Task:** Early Fire Precursor Detection via Watchtower 
- **ML Task:** Supervised Object Detection
- **Target Classes:** `Smoke` (Class `0`)

## 3. Project Constraints (The Rules of the Game)

### Constraint 1: Maximize Recall over Precision
- **Business Link:** The cost of missing a smoke plume (False Negative) is a full-blown forest fire. The cost of a False Positive (e.g., misclassifying a cloud as smoke) is low (human verification).
- **Technical Action:** We will tune our confidence thresholds to prioritize Recall.

### Constraint 2: Generalization to Fire (Cross-Domain Transfer)
- **Business Link:** The ultimate objective is not just to detect smoke, but to prove that learning the visual semantics of smoke allows the model to predict the presence of a fire.
- **Technical Action:** Dataset B will be used strictly for **training** (and standard validation). Dataset A (which contains fire without smoke labels) will be used as the **test probe** to measure this transfer learning capability.

### Constraint 3: Handling Temporal Leakage
- **Business Link:** Over-optimistic models cost lives because they fail in production. If we leak consecutive video frames into train and val sets, the model learns the background, not the smoke.
- **Technical Action:** We MUST split our dataset at the video/clip level (e.g., `evoDJI_0001` vs `evoDJI_0007`), NOT the frame level.

## 4. Expected Outcomes
- **Dataset Size:** ~4,954 images across 4 locations (Evo, Heinola, Karkkila, Ruokolahti).
- **Success Criteria:** > 85% Recall on the Dataset B validation set, and a measurable "Fire Detection Rate" > 0% when tested zero-shot on Dataset A.
