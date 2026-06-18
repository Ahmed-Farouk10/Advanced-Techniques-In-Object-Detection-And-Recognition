# Task 1: Business Logic & Constraints (Dataset A)

> **Dataset A (Drone/UAV Imagery)**  
> **Phase:** Problem Understanding & Business Translation

## 1. The Business Problem (WHY)
Traditional object detection requires explicitly labeled bounding boxes for the target object. However, what if a drone flies over a fire and the model has only ever seen smoke? 

The goal of this dataset in our project is NOT to train an object detection model. The EDA revealed this dataset contains **no bounding boxes** — it is purely a binary classification dataset (`fire` vs `nofire`). 

We are pivoting its role: Dataset A will act as a **zero-shot cross-domain evaluation probe**.

## 2. ML Translation
- **Business Task:** Verify if a smoke-trained watchtower model can detect fire from a drone.
- **ML Task:** Zero-Shot Transfer Evaluation (Bounding Box to Classification Mapping)
- **Target Classes:** `fire` vs `nofire`

## 3. Project Constraints (The Rules of the Game)

### Constraint 1: Bounding Box to Classification Mapping
- **Technical Action:** Since our models (trained on Dataset B) output bounding boxes, and Dataset A only has folder-level labels, we define the following mapping:
  - If the model predicts *at least one bounding box* with confidence > θ, classify the image as `Detected`.
  - Compare `Detected` against the `fire`/`nofire` ground truth folder.

### Constraint 2: No Data Leakage
- **Technical Action:** Dataset A must be strictly quarantined from the training process. No images from Dataset A will ever be used to update model weights.

### Constraint 3: Sensitivity Sweeping
- **Business Link:** In a life-safety scenario, we want to know *how confident* the model is when it sees fire for the first time.
- **Technical Action:** We will evaluate the Transfer Accuracy at multiple confidence thresholds (e.g., θ = 0.1, 0.3, 0.5, 0.7) to plot a sensitivity curve.

## 4. Expected Outcomes
- We expect the models to trigger bounding boxes on the `fire` images, despite only being trained on `smoke`.
- We expect to measure a **Fire Detection Rate** and a **False Alarm Rate** for each of the 4 models.
