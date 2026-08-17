# Task 6: Feature Selection & Anchor Prior Analysis

> **Objective:** Extract spatial anchor priors from Dataset A and analyze the structural discrepancy with Dataset B.

---

## 1. K-Means Fire Anchor Clustering ($k=5$)

K-means clustering was executed on normalized bounding box dimensions $(w, h)$ across all ground-truth fire targets in Dataset A:

| Anchor Prior | Normalized Width | Normalized Height | Area Coverage | Morphology Profile |
| :---: | :---: | :---: | :---: | :--- |
| **Anchor 1** | **0.1006** | **0.1272** | **0.0128 (1.28%)** | Small localized ignition point |
| **Anchor 2** | **0.1542** | **0.2901** | **0.0447 (4.47%)** | Vertical flame front |
| **Anchor 3** | **0.3198** | **0.2875** | **0.0920 (9.20%)** | Medium cluster fire |
| **Anchor 4** | **0.2203** | **0.5492** | **0.1210 (12.10%)**| Tall crowning tree fire |
| **Anchor 5** | **0.6355** | **0.4263** | **0.2709 (27.09%)**| Wide surface fire line |

---

## 2. Comparison: Fire Anchors (Dataset A) vs. Smoke Anchors (Dataset B)

```
Dataset B Smoke Anchors:  [0.03,  0.18,  0.34,  0.51,  0.73]  ← (Massive Plumes)
                                ▲      ▲      ▲      ▲
                                │      │      │      │  (Severe Domain Mismatch)
                                ▼      ▼      ▼      ▼
Dataset A Fire Anchors:   [0.01,  0.04,  0.09,  0.12,  0.27]  ← (Compact Flames)
```

### Strategic Conclusion:
* **Faster R-CNN Anchor Miss:** Faster R-CNN's RPN anchors were configured on large smoke clusters (up to Area 0.73), making them structurally ill-suited to produce high-IoU region proposals for tight flame centers (Area 0.01–0.04).
* **Why Deformable DETR Succeeded:** Deformable DETR does not rely on rigid anchor scales; its multi-scale deformable keypoints dynamically adjust to compact flame clusters, achieving **58.82% detection rates**.
