# Roadmap: From Master's Project to A-Tier Paper

> **AIN7601 — Cognitive Fire Defense Pipeline**  
> **Goal:** Transform current work into a competitive conference submission

---

## Current State (Complete)

| # | Deliverable | Status |
|---|------------|--------|
| 1 | Business logic (Dataset B) | Done |
| 2 | EDA + advanced EDA (Task 2) | Done |
| 3 | Data cleaning (Task 3) | Done |
| 4 | Data splitting + constraint optimization (Task 4) | Done |
| 5 | Augmentation config (Task 5) | Done |
| 6 | Premortem (14 risks, 15 tests) | Done |
| 7 | Research document (25 cleaning techniques) | Done |
| 8 | YOLO11n train.py | Done |

---

## Phase 3: Training Infrastructure (This Session)

| # | Ticket | Who | Time |
|---|--------|-----|------|
| 1 | `shared/yolo_to_coco.py` — convert YOLO TXT → COCO JSON | Antigravity | 1h |
| 2 | `rtdetr/train_rtdetr.py` — RT-DETR training script | Ahmed (antigravity) | 30m |
| 3 | `dino/train_dino.py` — DINO HuggingFace training script | Ahmed (antigravity) | 1h |
| 4 | `faster_rcnn/train_faster_rcnn.py` — with custom k=5 RPN anchors | Esraa | 1h |
| 5 | YOLO11n train.py — execute training | Esraa | 30-45m |

---

## Phase 4: Model Training Execution

| # | Model | Input Size | Batch | Device | Epochs | Owner |
|---|-------|-----------|-------|--------|--------|-------|
| 1 | YOLO11n | 640 | 16 | T4 | 100 | Esraa |
| 2 | RT-DETR | 640 | 8 | T4 | 100 | Ahmed |
| 3 | Faster R-CNN (custom anchors) | 800×1333 | 4 | T4 | 100 | Esraa |
| 4 | DINO (HuggingFace) | 800 | 2 | T4 | 50 | Ahmed |

**Output per model:** `results.csv` (epoch-level metrics), `best.pt` (or `pytorch_model.bin`), `train_loss.png`

---

## Phase 5: Evaluation on Dataset B (Smoke Validation)

| # | Analysis | Output |
|---|----------|--------|
| 1 | Per-model mAP@0.5, mAP@0.5:0.95 on val split | Numbers |
| 2 | APsmall, APmedium, APlarge per model | Table |
| 3 | Precision-Recall curves (4 models on same plot) | Figure |
| 4 | F1 confidence curve (4 models on same plot) | Figure |
| 5 | Confusion matrix per model | Figure |
| 6 | Inference speed (FPS) per model on T4 | Table |
| 7 | Model size (params, FLOPs, disk) | Table |

---

## Phase 6: Zero-Shot Evaluation on Dataset A (Fire Test Probe)

| # | Analysis | Output |
|---|----------|--------|
| 1 | Rewrite Dataset A for zero-shot probe role | Tasks 1-2 docs |
| 2 | Run inference of all 4 smoke-trained models on Dataset A fire images | Detection json |
| 3 | Fire Detection Rate: any bbox > threshold → fire | Number per model |
| 4 | False Positive Rate on no-fire images | Number per model |
| 5 | Confidence threshold sweep (0.1 to 0.9) for sensitivity curve | Figure |
| 6 | Per-model attention map visualization (Grad-CAM) on fire images | Figures |

---

## Phase 7: Feature-Level Proof (Makes It A-Tier)

| # | Analysis | What It Proves |
|---|----------|---------------|
| 1 | Extract backbone features from smoke bboxes vs fire bboxes | Are smoke/fire features similar in latent space? |
| 2 | Cosine similarity matrix (smoke frames vs fire frames per model) | Quantifies the transfer |
| 3 | t-SNE of backbone features colored by smoke/fire | Visual proof of feature overlap |
| 4 | GLCM texture of smoke bboxes vs fire bboxes | Are textures actually similar? |
| 5 | Canny edge density per bbox (smoke vs fire) | Do the models key on edges? |
| 6 | Attention map overlay on Dataset A images | Where does DINO look when it detects fire? |
| 7 | Per-model failure analysis: which fire images get zero detections? | Night? Small? Occluded? |

---

## Phase 8: Statistical Rigor (Makes It Defensible)

| # | Task |
|---|------|
| 1 | Run each model 3× with different random seeds (42, 100, 2026) |
| 2 | Report mean ± std for all metrics |
| 3 | Add upper bound: train one model directly on fire (or pseudo-annotations) |
| 4 | Add lower bound: simple HSV threshold-based fire detector |
| 5 | Statistical test: is the smoke→fire mAP significantly > random? |

---

## Phase 9: Paper Writing

| Section | Content |
|---------|---------|
| Abstract | Smoke→fire transfer, 4-model ablation, key finding |
| Introduction | Wildfire early detection, why smoke is the precursor |
| Related Work | Smoke detection, fire detection, zero-shot transfer, temporal leakage |
| Methodology | Datasets, 6-task pipeline, splitting, augmentations, models, anchors |
| Results | Tables from Phase 5 + 6 |
| Discussion | Feature analysis (Phase 7), failure analysis, why transfer works/fails |
| Conclusion | Contributions, limitations, deployment path |
| References | ~30-40 papers |

**Figures to produce:**
1. Dataset B sample images with bboxes
2. Anchor clustering plot (Task 2)
3. Split distribution table (Task 4)
4. mAP comparison bar chart (4 models, smoke val)
5. PR curves (4 models, same plot)
6. Fire detection rate bar chart (zero-shot)
7. Confidence sweep curve
8. t-SNE feature space (smoke vs fire)
9. Grad-CAM attention maps (fire images)
10. Failure case examples

---

## Phase 10: Summer — A-Tier Upgrade Path

### 10.0 What A-Tier Reviewers Demand (The Checklist)

A typical CVPR/ICCV/WACV reviewer evaluates on:

| Criterion | Weight | What They Look For |
|-----------|--------|-------------------|
| **Novelty** | 30% | Is the idea new, or an incremental tweak? Does it challenge an assumption? |
| **Technical Quality** | 25% | Are experiments rigorous? Ablations? Statistical significance? Upper/lower bounds? |
| **Clarity** | 15% | Can a PhD student in a different subfield understand the paper? |
| **Reproducibility** | 15% | Code, hyperparams, seeds, dataset splits — can someone else replicate? |
| **Impact** | 15% | Does this change how people think about the problem? Or just another 0.3% mAP gain? |

**Common rejection reasons to avoid:**

| Rejection Reason | How We Prevent It |
|-----------------|-------------------|
| "Incremental — just applied existing models to new data" | We don't apply models. We ask: *does smoke visual semantics transfer to fire across architectures?* That's a research QUESTION, not a model run. |
| "No comparison to SOTA" | Add upper bound (fire-trained model), lower bound (HSV baseline), and compare against published smoke detection baselines. |
| "Single dataset — no evidence of generalization" | Add 1-2 external fire datasets (Phase 10.1). |
| "Metrics are cherry-picked — one seed, no error bars" | 3-seed runs with mean ± std (Phase 8). |
| "No insight into WHY — just numbers" | Phase 7 provides the mechanism layer. |
| "Overclaims — 'we solved wildfire detection'" | Honest limitations section: "Our zero-shot transfer works on daytime aerial fire but fails on night and ground-level fires. Transfer is partial, not perfect." |
| "No code, no hyperparams, unreproducible" | Every script is version-controlled. `build_clean_nb.py`, `build_split_nb.py`. Release on GitHub. |
| "Related work is shallow — missing key papers" | 30-40 citations minimum, covering smoke detection (classical + DL), fire detection, zero-shot transfer, temporal leakage, domain adaptation, and each architecture's original paper. |

### 10.1 Add External Validation Datasets

A single dataset pair (Boreal → Forest Fire) is insufficient for A-tier generalizability claims. Minimum: 2-3 distinct fire test sets.

| Dataset | Source | Images | Format | Fire Type | Acquisition |
|---------|--------|--------|--------|-----------|-------------|
| **VisiFire** | Bilkent University | ~1,000 video clips | Frame-level fire/smoke labels | Ground-level, CCTV | Public |
| **FLAME** | Northern Arizona University | ~2,000 frames | Fire segmentation masks | Aerial drone, prescribed burns | Public |
| **Corsican Fire Dataset** | University of Corsica | ~1,500 images | Fire/no-fire classification | Ground + aerial, visible + IR | Public (request) |
| **FireNet** | Custom compilation | ~500 images | Bounding boxes | Mixed (urban, forest, industrial) | Public |
| **FiSmo** | University of Tokyo | ~5,000 frames | Bounding boxes + segmentation | Ground-level, multi-condition | Public (request) |

**Evaluation protocol per dataset:**
1. Run zero-shot inference of all 4 smoke-trained models
2. If dataset has bboxes → report mAP
3. If dataset has classification labels → report Fire Detection Rate + FPR
4. If dataset has segmentation masks → convert to bbox IoU via minimal bounding rectangle
5. Aggregate: "Across N external datasets, smoke-trained models achieve X% mean detection rate"

### 10.2 Comprehensive Ablation Study

Reviewers expect to see: "We removed X, and Y happened. Therefore X matters."

| Ablation | Question Being Answered | Expected Finding |
|----------|------------------------|-----------------|
| **Disable Mosaic** (set to 0) | Does multi-image augmentation help smoke detection? | mAP likely drops 3-5% but small plume AP may increase |
| **Disable HSV jitter** | Does simulating dusk/dawn matter? | mAP on bright val images unchanged; crucial for dark test images |
| **Disable copy-paste** | Does synthetic multi-object help? | Minor drop since 99.5% images are single-box |
| **Default COCO anchors vs custom k=5** (Faster R-CNN only) | Do domain-specific anchors improve smoke detection? | Expected 2-4 mAP gain on custom anchors |
| **Random split vs clip-level split** | How much does temporal leakage inflate metrics? | Random split likely shows 15-25% inflated mAP (paper's key warning to community) |
| **With vs without empty images** | Do 256 negatives reduce false positives? | FPR should drop by 10-20% with negatives |
| **Scale augmentation ablation** | Does preserving 4K textures via crop vs resize matter? | mAP on small plumes should be higher with crop strategy |

**Output:** One large table: `Ablation Study Results`

```
| Ablation | YOLO11n mAPΔ | RT-DETR mAPΔ | Faster R-CNN mAPΔ | DINO mAPΔ |
|----------|-------------|-------------|-------------------|-----------|
| No Mosaic | -3.2 | -2.8 | -1.5 | -4.1 |
| No HSV   | -0.5 | -0.3 | -0.1 | -0.8 |
| ...      |       |       |                   |           |
```

### 10.3 Upper Bound & Lower Bound Baselines

Without bounds, your zero-shot numbers are meaningless. You need:

**Upper Bound (Oracle/Supervised):**
- Train one model (YOLO11n or RT-DETR) directly on fire images with bboxes
- If Dataset A has no bboxes → generate pseudo-labels using a pretrained fire detector (e.g., existing Fire-YOLO checkpoint) OR manually annotate 100-200 fire images
- This shows: "The ceiling for fire detection on this dataset is X% mAP. Our zero-shot method achieves Y% of the ceiling."
- Without this, a reviewer says: "62% mAP on fire — is that good or bad? What's the maximum possible?"

**Lower Bound (Naive Baseline):**
- HSV color-based fire detector: threshold on red/orange hue + saturation channels
- OR: simple frame differencing (motion detector) — fires flicker, forests don't
- OR: HOG + SVM classifier trained on 50 fire/50 no-fire from Dataset A (classical ML baseline)
- This shows: "Even our worst zero-shot model (X% detection rate) outperforms the naive baseline (Y%)."

**Table:**
```
| Method | Fire Detection Rate | FPR | Notes |
|--------|-------------------|-----|-------|
| HSV threshold (lower bound) | 34.2% | 42.1% | Color-based, no learning |
| Ours (YOLO11n, zero-shot from smoke) | 62.3% | 18.7% | No fire training |
| Ours (RT-DETR, zero-shot from smoke) | 58.1% | 21.4% | No fire training |
| ... | | | |
| Fire-trained YOLO11n (upper bound) | 94.7% | 4.2% | Trained on fire directly |
```

### 10.4 Cross-Modality Experiment (Multispectral Transfer)

Wildfire detection in production uses both visible (RGB) and infrared (thermal/IR) cameras. Does smoke→fire transfer work across modalities?

**Experiment 1: RGB Smoke → IR Fire**
- If any external dataset provides IR imagery (Corsican has visible+IR pairs), train on RGB smoke, test on IR fire
- Hypothesis: RGB smoke textures don't transfer to IR (different spectral signature). Failure is a finding.

**Experiment 2: RGB Smoke → Multispectral Fire**
- Use a dataset with both visible and IR frames of the same fire
- Report: does zero-shot detection rate drop from visible to IR?

**Experiment 3: Early Fusion vs Late Fusion**
- Train one model on RGB+IR concatenated smoke (4-channel input instead of 3)
- Test on RGB-only fire
- Does adding IR during training help or hurt RGB-only inference?

This section is optional but makes the paper stand out. Multispectral transfer is an open research problem.

### 10.5 Deployment Benchmark (Edge Readiness)

The business case (watchtower → early detection) implies edge deployment. Prove your lightest model works.

| Model | Framework | Device | Precision | FPS | Latency (ms) | Power (W) | Memory (MB) |
|-------|-----------|--------|-----------|-----|-------------|-----------|-------------|
| YOLO11n | PyTorch | T4 GPU | FP32 | 185 | 5.4 | 70W | 280 |
| YOLO11n | TensorRT | Jetson Orin Nano | FP16 | 42 | 23.8 | 7W | 145 |
| YOLO11n | TensorRT | Jetson Orin Nano | INT8 | 78 | 12.8 | 5W | 98 |
| RT-DETR | PyTorch | T4 GPU | FP32 | 62 | 16.1 | 70W | 620 |

Tools: `trtexec`, `onnxruntime`, `torch2trt`. NVIDIA provides free Jetson benchmarks.

### 10.6 Theoretical Contribution Framework

A-tier papers don't just report numbers — they offer a conceptual framework others can use. Your framework:

**The Visual Prototype Transfer Hypothesis:**

> "Object detectors trained on one visual phenomenon (smoke) can detect a semantically related but visually distinct phenomenon (fire) to the extent that both share low-level visual prototypes: turbulent fluid dynamics, high-frequency texture, semi-transparency against sky backgrounds, and upward motion trajectory."

**Testable predictions of this hypothesis:**
1. Transfer should be stronger for architectures that learn texture-rich features (CNN backbones) → YOLO/Faster R-CNN > DINO?
2. Transfer should be weaker for night fires (low contrast, different texture signature) → test with brightness-stratified evaluation
3. Transfer should be stronger for aerial views (both datasets are drone) than ground-level views → test with VisiFire (ground-level CCTV)
4. Augmentations that destroy texture (excessive blur, low resolution) should degrade transfer more than augmentations that preserve texture (HSV jitter)
5. Backbone layers that encode mid-level texture features (conv3-conv4) should show highest cosine similarity between smoke and fire; early layers (edges) and late layers (semantic) should diverge

**This turns your paper from** "we tested 4 models" **into** "we tested a falsifiable hypothesis about visual cognition in deep detectors."

### 10.7 Extended Related Work Taxonomy

A-tier papers have a Related Work section that positions the paper in a research landscape, not just a list of citations.

```
Related Work Organization:

1. Smoke Detection in Computer Vision
   ├── Classical methods (Gubbi et al. 2009, Toreyin et al. 2006 — color + motion)
   ├── CNN-based (Yuan et al. 2019 — DeepSmoke, Xu et al. 2021 — attention-guided)
   └── Transformer-based (Li et al. 2023 — SmokeViT)

2. Fire Detection and Localization
   ├── RGB-based (Sharma et al. 2023 — FireNet, de Venancio et al. 2022 — survey)
   ├── Multispectral (Sousa et al. 2022 — RGB+IR fusion)
   └── Edge deployment (Barmpoutis et al. 2020 — Jetson Nano, Perrolas et al. 2022 — UAV)

3. Zero-Shot and Cross-Domain Transfer in Object Detection
   ├── Language-guided (GLIP, Grounding DINO, OWL-ViT — text prompts for novel classes)
   ├── Domain adaptation (Chen et al. 2018 — DA-Faster R-CNN, Saito et al. 2019 — MME)
   └── Visual prototype transfer (OUR WORK — visual semantics without language)

4. Temporal Leakage in Sequential Data
   ├── Video understanding (Tran et al. 2019 — frame-level split risks)
   ├── Remote sensing (Zhong et al. 2020 — spatial autocorrelation leakage)
   └── OUR CONTRIBUTION: pHash + clip-level constraint optimization for drone footage

5. Data Augmentation for Small Object Detection
   ├── Mosaic, MixUp, Copy-Paste (Bochkovskiy et al. 2020, Ghiasi et al. 2021)
   └── Scale-aware augmentation (Kisantal et al. 2019 — oversampling small objects)
```

### 10.8 Paper Structure for A-Tier Conference (8 Pages, CVPR/ICCV Format)

```
Page 1:    Introduction
           - Wildfire problem: 4.5M km² burned globally per year
           - Early detection gap: humans see flames, not precursors
           - Our question: can models learn fire from smoke alone?
           - Contributions: (1) zero-shot smoke→fire transfer protocol,
             (2) 4-architecture benchmark, (3) temporal leakage mitigation,
             (4) visual prototype transfer framework

Page 2:    Related Work (condensed from 10.7)
           - 5 subsections, 25-30 citations

Page 3:    Methodology — Data Pipeline
           - Datasets B and A (Table 1: dataset statistics)
           - 6-task pipeline diagram (Figure 1)
           - Temporal leakage: pHash dedup + clip-level constraint optimization
           - Split verification table (Table 2)

Page 4:    Methodology — Models & Augmentations
           - 4 architectures (Table 3: model specs)
           - custom_hyp.yaml design rationale (mapped to Task 2 findings)
           - Anchor clustering for Faster R-CNN (Figure 2)
           - Domain-specific anchors vs COCO anchors

Page 5:    Results — Smoke Validation (Dataset B)
           - Table 4: per-model mAP, APsmall/medium/large, FPS, params
           - Figure 3: PR curves (4 models)
           - Figure 4: confidence sweep
           - Key finding: which architecture generalizes best on smoke?

Page 6:    Results — Zero-Shot Fire Transfer (Dataset A)
           - Table 5: Fire Detection Rate, FPR per model
           - Table 6: upper/lower bounds
           - Figure 5: detection rate vs confidence threshold
           - Figure 6: Grad-CAM attention maps on fire images

Page 7:    Discussion — Why Transfer Works (or Fails)
           - t-SNE feature space (Figure 7): smoke vs fire latent representations
           - Cosine similarity matrix (Figure 8)
           - Failure case gallery (Figure 9)
           - Ablation results (Table 7)
           - Which architectural features enable/disrupt transfer?

Page 8:    Conclusion & Limitations
           - Contributions restated
           - Limitations: single training dataset, daytime bias, no IR, no real-time validation
           - Future work: multispectral, video-level detection, on-device deployment
           - Broader impact: climate change adaptation, early warning systems
```

### 10.9 Target Venues — Detailed Breakdown

| Venue | Tier | Accept Rate | Paper Length | Emphasis | Why Us | Timeline |
|-------|------|-------------|-------------|----------|--------|----------|
| **WACV 2027** | A- | ~30% | 8 pages | Applications, real-world impact | Climate + edge deployment story | Submit ~Aug 2026, Decision ~Oct 2026 |
| **BMVC 2027** | A- | ~25% | 9 pages | British/European, strong on methodology | Rigorous pipeline + temporal leakage contribution | Submit ~May 2027 |
| **ACCV 2026** | A- | ~28% | 8 pages | Asian conference, broad scope | Cross-domain transfer + transformer analysis | Submit ~July 2026 (**earliest deadline**) |
| **ICPR 2026** | B+ | ~40% | 8 pages | Pattern recognition, broad | Zero-shot detection pattern recognition | Submit ~April 2026 (likely missed) |
| **CVPR EarthVision Workshop** | Workshop | ~40% | 4 pages | Earth observation, climate | Perfect thematic fit | Submit ~March 2027 |
| **NeurIPS CCAI Workshop** | Workshop | ~35% | 4 pages | Climate change AI | Strong social impact narrative | Submit ~May 2027 |
| **Remote Sensing (MDPI)** | Journal (IF 5.0) | ~60% | 15-20 pages | Remote sensing, drone, satellite | Drone footage, multispectral potential | Rolling submission |
| **Fire (MDPI)** | Journal | ~55% | 12-15 pages | Wildfire science | Directly on-topic | Rolling submission |

**Strategy:** Submit to ACCV 2026 (earliest deadline, good fit). If rejected, incorporate reviews and submit to WACV 2027. If rejected again, expand and submit to Remote Sensing (MDPI) with full experimental detail. Workshop papers can be submitted in parallel (different content — shorter, focused on climate impact narrative).

### 10.10 Example Papers to Model Your Structure After

| Paper | Why Model After It |
|-------|-------------------|
| **Kisantal et al. (2019)** — "Augmentation for small object detection" (CVPR Workshop) | Shows how to frame an augmentation study as a research contribution. Similar to your small plume augmentation story. |
| **Bochkovskiy et al. (2020)** — "YOLOv4: Optimal Speed and Accuracy" | The gold standard for how to present multi-model comparisons with ablation. Every design choice justified with experiment. |
| **Barmpoutis et al. (2020)** — "Early fire detection with autonomous UAV" (Remote Sensing) | Shows how to connect computer vision metrics to real-world fire detection requirements. Your deployment benchmark should follow this. |
| **Carion et al. (2020)** — "End-to-End Object Detection with Transformers (DETR)" (ECCV) | How to present a new paradigm (transformer detection) with clear comparison to CNN baselines. Your zero-shot transfer vs supervised is analogous. |

---

## Priority Order (What to Do Next)

```
NOW:         Phase 3 — COCO converter + training scripts
THIS WEEK:   Phase 4 — Train all 4 models
             Phase 5 — Evaluate on smoke val
             Phase 6 — Zero-shot evaluate on fire test probe
             Phase 7 — Feature analysis + Grad-CAM
             Phase 8 — 3-seed statistical runs
             Phase 9 — Write paper
SUMMER:      Phase 10.1 — 1-2 external validation datasets
             Phase 10.2 — Full ablation study
             Phase 10.3 — Upper/lower bound baselines
             Phase 10.5 — Deployment benchmark (TensorRT + Jetson)
             Phase 10.6 — Theoretical framework formalization
             Phase 10.7 — Extended related work
             SUBMIT to ACCV or WACV
```

---

## What Makes This "Big" (Summary)

| Layer | B-Tier | A-Tier (Our Target) |
|-------|--------|---------------------|
| Questions answered | "Which model works best?" | "Does smoke visual semantics transfer to fire, and WHICH architectural features enable/disrupt it?" |
| Evidence | mAP numbers | mAP + t-SNE + cosine similarity + Grad-CAM + ablation + upper/lower bounds |
| Datasets | 1 train, 1 test | 1 train, 2-3 test (cross-dataset generalization) |
| Reproducibility | Code in repo | `build_*.py` scripts + config files + seed reporting |
| Statistics | 1 run | 3-seed mean ± std + significance tests |
| Honesty | Claims it works | Shows where it fails + why + limitation analysis |
| Contribution | "Applied YOLO to fire" | "Proposed visual prototype transfer hypothesis and validated it across 4 architectures" |
| Impact | Another detection paper | Framework for thinking about cross-phenomenon visual transfer in safety-critical domains |

**The gap from current to A-tier:** Phases 7 + 8 + 10 = 2-3 months of additional work over the summer. The gap from current to strong B-tier: just finish Phases 4-6 and write it up honestly. You're already B-tier ready.
