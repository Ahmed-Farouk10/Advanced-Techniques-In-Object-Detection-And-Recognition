## Observations
- O1: Traditional forest fire detection models suffer from bottlenecks when run on edge hardware like drones.
- O2: Lightweight edge models trigger too many false alarms due to confusing clouds or glare for fire.
- O3: Severe class imbalance exists in forest environments (massive background vs. tiny smoke patches).

## Gaps
- G1: Need a system that can process video on edge drones rapidly without requiring enterprise GPUs.
- G2: Need a mechanism to eliminate environmental false positives at long distances without training heavy monolithic models on vast cloud datasets.

## Key Insight
- By splitting the workload into an asymmetric Edge-Cloud hybrid architecture—a fast "Sentinel" (e.g., YOLO11n) for reflexes and a zero-shot "Commander" (e.g., OWLv2) for semantic triage—the system can achieve high accuracy while remaining within compute limits.

## Assumptions
- Assumption 1: Drones and watchtowers have access to edge hardware comparable to T4 constraints.
- Assumption 2: Zero-shot semantic filtering via OWLv2 can distinguish visual mimics (clouds vs smoke) better than standard CNNs without explicit training on those mimics.
