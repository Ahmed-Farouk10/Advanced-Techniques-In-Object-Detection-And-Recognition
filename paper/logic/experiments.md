## E01: Micro View Performance Analysis
- Verifies: [C01]
- Setup: Dataset A (Drone/UAV Imagery), T4 GPU edge simulation.
- Procedure: Train both YOLO11n and MobileNet-Faster-RCNN on Dataset A. Measure inference FPS and recall on dynamic test set.
- Metrics: Precision, Recall, F1-Score, FPS
- Expected outcome: YOLO11n achieves significantly higher FPS with comparable recall.
- Baselines: MobileNet-Faster-RCNN
- Dependencies: []

## E02: Macro View False Positive Suppression
- Verifies: [C02]
- Setup: Dataset B (Static Watchtower Imagery).
- Procedure: Evaluate false positive rate on images containing visual mimics (sunsets, fog, clouds) using Swin-T vs EfficientDet-D0+OWLv2.
- Metrics: Precision, Recall, False Positive Rate, IoU
- Expected outcome: EfficientDet+OWLv2 pipeline flags significantly fewer false positives than Swin-T.
- Baselines: Swin-T
- Dependencies: []

## E03: Memory Complexity Evaluation
- Verifies: [C01, C02]
- Setup: All four models.
- Procedure: Measure model size and FLOPs.
- Metrics: Model Size (MB), FLOPs
- Expected outcome: EfficientDet and YOLO11n show the lowest memory footprint.
- Baselines: Swin-T, Faster R-CNN
- Dependencies: []
