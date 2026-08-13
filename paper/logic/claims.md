# Claims — Cognitive Fire Defense

## C01: Smoke-to-fire visual prototype transfer exists
- Statement: Object detectors trained exclusively on smoke bounding boxes will detect fire above a naive baseline, because both phenomena share low-level visual prototypes (turbulent fluid dynamics, high-frequency texture, semi-transparency against sky).
- Status: Proposed, pending training results
- Falsification: All four models achieve fire detection rates indistinguishable from random chance (50% at any threshold)
- Proof: Phase 6 — zero-shot fire evaluation on Dataset A
- Evidence basis: Not yet collected
- Interpretation: If C01 holds, cognitive-science question answered: smoke visual features transfer without language. If falsified, meaningful negative result — smoke and fire features do not overlap.

## C02: Transfer magnitude varies by architecture
- Statement: Transformer-based architectures (RT-DETR, DINO) exhibit different zero-shot transfer behavior than CNN-based architectures (YOLO11n, Faster R-CNN) due to differences in how global vs. local features are encoded.
- Status: Proposed, pending training results
- Falsification: All four models show statistically indistinguishable transfer performance
- Proof: Phase 6 comparison
- Evidence basis: Not yet collected

## C03: Clip-level splitting prevents metric inflation from temporal leakage
- Statement: A random split of sequential video frames inflates validation mAP by 15--25% compared to our clip-level constraint-optimized split, because the model learns static scene context rather than smoke features.
- Status: To be tested via ablation study 7
- Falsification: Random split and clip-level split produce statistically indistinguishable validation metrics
- Proof: Ablation 7
- Evidence basis: Not yet collected

## C04: Domain-specific anchors improve smoke localization over default COCO anchors
- Statement: Faster R-CNN with custom k=5 smoke anchors achieves higher mAP on smoke validation than identical architecture with default COCO anchors (32², 64², 128², 256², 512²).
- Status: To be tested via ablation study 6
- Falsification: Custom anchors and COCO anchors produce statistically indistinguishable mAP
- Proof: Ablation 6
- Evidence basis: Not yet collected
