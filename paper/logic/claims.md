## C01: YOLO11n outperforms Faster R-CNN on edge hardware
- Statement: YOLO11n achieves higher FPS than MobileNet-Faster-RCNN on drone video while maintaining acceptable recall.
- Status: Proposed
- Falsification criteria: Faster-RCNN maintains real-time FPS on edge hardware with better recall than YOLO11n.
- Proof: [E01, E03]
- Evidence basis: Not yet collected.
- Interpretation: One-stage detectors are fundamentally better suited for fast drone deployment than two-stage detectors.
- Dependencies: []
- Tags: [edge, performance]

## C02: Semantic Pipeline suppresses distant false positives better than monolithic Swin-T
- Statement: The EfficientDet+OWLv2 pipeline eliminates more visual mimics (e.g. sunsets, clouds) at long distances than Swin-T.
- Status: Proposed
- Falsification criteria: Swin-T achieves a lower false positive rate than the semantic pipeline on Dataset B.
- Proof: [E02, E03]
- Evidence basis: Not yet collected.
- Interpretation: Zero-shot textual reasoning can filter out-of-distribution visual mimics without requiring extensive targeted training data.
- Dependencies: []
- Tags: [semantic, false-positives]
