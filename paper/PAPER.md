---
title: "Smoke Before Fire: Can Object Detectors Find a Wildfire Before the Flame Is Visible?"
authors: ["Esraa Nasr ElSayed", "Ahmed Ayman"]
year: "2026"
venue: "Masters Academy AI — AIN7601 Advanced Techniques in Object Detection and Recognition"
doi: ""
domain: "Computer Vision — Object Detection"
keywords: ["Wildfire Detection", "Smoke Detection", "Object Detection", "YOLO11n", "Faster R-CNN", "RT-DETR", "DINO", "Zero-Shot Transfer", "Temporal Leakage", "Constraint Optimization"]
abstract: "Wildfires destroy approximately 4.5 million square kilometers of land annually, yet most detection systems identify fires only after flames become visible. Smoke precedes flame in nearly every ignition scenario. This paper asks whether object detection models trained exclusively on smoke bounding boxes can generalize to fire detection without exposure to fire labels. Four architectures—YOLO11n (one-stage CNN), Faster R-CNN (two-stage RPN with domain-specific anchors), RT-DETR (hybrid CNN-transformer), and DINO (end-to-end transformer with deformable attention)—are trained on 3,066 smoke-annotated 4K drone images from Finnish boreal forests and evaluated zero-shot on a held-out fire dataset. The training pipeline incorporates a constraint-optimized, clip-level data split that eliminates temporal leakage across sequential video frames, a problem unaddressed in all twelve papers surveyed. A seven-part data cleaning protocol identified 4,139 anomalies while maintaining strict zero-imputation integrity. Domain-specific anchor clusters were computed from 4,862 smoke bounding boxes and injected into the Faster R-CNN region proposal network. This work establishes the first multi-architecture benchmark on the Boreal Forest Fire 2025 dataset and provides the first controlled experiment isolating smoke-to-fire visual prototype transfer in object detection."
controls: |
  The WHY RULE governs every decision in this paper: every choice of dataset, model, split, augmentation, and metric must be traceable to evidence from the data or from published literature. No parameter is set by convention alone.
  Data integrity: 4,954 images, zero ground truth modified, 4,139 anomalies flagged with full audit trail in cleaning_log.csv.
  Reproducibility: All preprocessing scripts (build_clean_nb.py, build_split_nb.py, build_train_nbs.py) and config files published alongside this paper.
  Limitations are documented honestly: daytime bias, single geographic region, unimplemented fog/motion-blur augmentations, limited small-plume evaluation sample.
---
