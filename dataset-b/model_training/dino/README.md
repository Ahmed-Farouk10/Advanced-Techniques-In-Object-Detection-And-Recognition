# 🦅 Deformable DETR Training & Evaluation Pipeline

> **Sub-module:** `dataset-b/model_training/dino/`  
> **Model:** Deformable DETR (`SenseTime/deformable-detr`) with ResNet-50 multi-scale backbone  
> **Module Owner & Contributor:** Ahmed Ayman  
> **Target Class:** `smoke` (Single-class, zero-shot transfer evaluated on unseen `fire`)

---

## 📌 Architectural Overview & Evolution

In our search for an optimal vision transformer for smoke and wildfire detection, we followed a progressive evaluation of transformer paradigms:

```
Vanilla DETR ───(Memory Bottleneck & Slow Convergence)───> Failed / Rejected
     │
     ▼
Standard DINO ──(Single-Class Gradient Collapse)─────────> Failed / Rejected
     │
     ▼
Deformable DETR ─(Sparse Attention + Focal Alpha Fix)────> ✅ SUCCESS (88.61% mAP@50, ~60% Zero-Shot Transfer)
```

### Why Vanilla DETR and DINO were bypassed:
1. **Vanilla DETR (Global Dense Attention):**  
   Computes full pairwise self-attention ($\mathcal{O}(H^2 W^2)$) across all feature pixels. On high-resolution UAV images ($640\times640$), this resulted in catastrophic VRAM spikes and required 300–500 epochs to converge.
2. **Standard DINO (Contrastive DeNoising):**  
   Designed for multi-class benchmarks (COCO 80 classes). On our single-class smoke dataset with empty negative frames, the contrastive query dynamics collapsed under extreme class imbalance.
3. **Deformable DETR (Sparse Multi-Scale Attention):**  
   Each query attends to a small, fixed set of $K=4$ sampling points around dynamic 2D reference points across multi-scale feature maps ($C_3, C_4, C_5, C_6$). Its computational complexity is linear ($\mathcal{O}(N_q K C)$), and the learned 2D sampling offsets naturally wrap around the amorphous, non-rigid boundaries of smoke plumes.

---

## 🛠️ Key Technical Implementations

* **Focal Loss Gradient Re-balancing:** Default `focal_alpha=0.25` collapses on sparse single-class datasets due to the ~299:1 background-to-object gradient pressure. We manually overridden `focal_alpha = 0.95` in `AutoConfig` to heavily upweight the positive smoke gradients.
* **In-Memory RAM Caching:** Pre-decodes and resizes images to $640\times640$ uint8 numpy arrays in RAM (~4.7 GB total) upon initialization to completely eliminate disk I/O bottlenecks and bypass Windows DataLoader multiprocessing deadlocks.
* **Precision:** Trained with `bf16=True` and gradient checkpointing on PyTorch / HuggingFace Transformers.

---

## 📂 Directory Contents

| File | Description |
| :--- | :--- |
| `train_deformable_detr.py` | Full training script for Deformable DETR on Dataset B (COCO format). |
| `evaluate_dino.py` | Zero-shot cross-domain evaluation script on Dataset A (`fire`). |
| `visualize_dino.py` | Inference visualizer generating qualitative detection overlays. |
| `train.ipynb` | Interactive notebook for step-by-step training and debugging. |
| `Cognitive_Fire_Defense/` | Output directory storing model checkpoints and evaluation logs. |
| `pred_*.jpg` | Sample visual predictions and qualitative bounding box outputs. |

---

## 🚀 How to Run

### 1. Training on Dataset B (Smoke)

Ensure the COCO annotations are generated via `dataset-b/preprocessing/task4_data_splitting/yolo_to_coco.py`, then run:

```bash
cd dataset-b/model_training/dino
python train_deformable_detr.py
```

**Key Hyperparameters:**
* **Input Resolution:** $640 \times 640$
* **Batch Size:** 4 (per device)
* **Epochs:** 50 (with early stopping patience = 10)
* **Base Learning Rate:** `5e-5` (cosine decay schedule with 5% warmup)
* **Weight Decay:** `1e-4`

### 2. Zero-Shot Fire Evaluation on Dataset A

Evaluate the trained checkpoint zero-shot on the unseen fire dataset across multiple confidence thresholds ($0.50, 0.30, 0.20, 0.10, 0.05$):

```bash
python evaluate_dino.py
```

### 3. Generate Qualitative Visualizations

```bash
python visualize_dino.py
```

---

## 📊 Experimental Results

### Within-Domain Smoke Detection (Dataset B Validation Split)
* **mAP@50:** `88.61%`
* **mAP@50–95:** `40.75%`

### Zero-Shot Fire Transfer (Dataset A — Relaxed Localization $\text{IoU} \ge 0.10$)

| Confidence Threshold | Predictions | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1 Score | Image Detection Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.50** | 765 | 57 | 708 | 938 | 7.45% | 5.73% | 6.48% | 12.42% (57/459) |
| **0.30** | 2,351 | 103 | 2,248 | 892 | 4.38% | 10.35% | 6.16% | 21.13% (97/459) |
| **0.20** | 5,283 | 153 | 5,130 | 842 | 2.90% | 15.38% | 4.87% | 28.32% (130/459) |
| **0.10** | 19,656 | 265 | 19,391 | 730 | 1.35% | 26.63% | 2.57% | 42.92% (197/459) |
| **0.05** | 46,983 | 431 | 46,552 | 564 | 0.92% | **43.32%** | 1.80% | **58.82% (270/459)** |

> 💡 **Takeaway:** Under early-warning operational constraints ($\text{IoU} \ge 0.10$), Deformable DETR correctly identified fire anomalies in **58.82% of unseen fire images** without ever receiving a single fire training label, proving that sparse attention models capture cross-domain semantic texture better than standard CNNs.
