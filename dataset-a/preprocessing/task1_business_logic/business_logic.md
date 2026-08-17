# Task 1 — Business Logic & Evaluation Strategy (Dataset A)

> **Phase:** Evaluation Probe Design | **Dataset:** Dataset A (Fire & Smoke Test Probe)

---

## 1. Problem Definition & Operational Context

In wilderness wildfire surveillance, unmanned aerial vehicles (UAVs) must operate under severe uncertainty:
* **The Training Assumption:** The model is trained exclusively on early plume smoke formations from boreal restoration burns (Dataset B).
* **The Operational Test:** When deployed over an active wildfire zone, can the detector recognize unannotated **fire** without generating unacceptable false alarm cascades?

---

## 2. Asymmetric Cost Matrix: False Alarms vs. Missed Fires

In early warning systems, prediction errors carry vastly unequal operational consequences:

| Outcome | Real-World Scenario | Cost & Impact |
| :--- | :--- | :--- |
| **True Positive (TP)** | Flame detected correctly | **High Benefit:** Immediate dispatch of suppression assets before crowning. |
| **False Negative (FN)** | Unseen fire undetected | **Catastrophic Failure:** Uncontrolled fire expansion, loss of habitat, life, and infrastructure. |
| **False Positive (FP)** | Background tree/glare flagged | **Moderate Cost:** Operator spends 2 seconds reviewing drone telemetry feed. |
| **True Negative (TN)** | Clean forest ignored | **Routine:** Normal patrol continues without bandwidth drain. |

**Core Optimization Principle:**  
$$\text{Cost}(\text{FN}) \gg \text{Cost}(\text{FP})$$

Therefore, the evaluation pipeline prioritizes **Recall over Precision**, especially at low confidence thresholds ($0.05 \le \tau \le 0.10$).

---

## 3. Two-Tier Evaluation Protocols

1. **Strict Zero-Shot Transfer ($\text{IoU} \ge 0.50$):**
   * Requires tight spatial overlap with ground-truth fire bounding boxes.
   * Evaluates whether learned feature representations encode precise geometric fire boundaries.
2. **Relaxed Early-Warning Alerting ($\text{IoU} \ge 0.10$):**
   * Requires any meaningful spatial overlap with the fire anomaly.
   * Mirrors real-world drone surveillance where flagging the vicinity of a flame is sufficient to alert human incident commanders.
