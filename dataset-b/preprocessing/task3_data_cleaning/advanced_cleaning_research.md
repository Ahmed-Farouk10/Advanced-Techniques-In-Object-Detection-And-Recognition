# Advanced Data Cleaning Techniques for Object Detection

> **AIN7601 Cognitive Fire Defense — Task 3 Research**  
> **Context:** Dataset B (Boreal Watchtower), 4,954 images, 4,862 YOLO bboxes, single-class (smoke)  
> **Task 2 Findings Integrated:** All techniques below are cross-referenced against what we know about the data

---

## Table of Contents
1. [Standard Integrity Checks](#1-standard-integrity-checks)
2. [Label Noise Detection](#2-label-noise-detection)
3. [Bounding Box Quality Assessment](#3-bounding-box-quality-assessment)
4. [Annotation Consistency Validation](#4-annotation-consistency-validation)
5. [Outlier Detection in Feature/Latent Space](#5-outlier-detection-in-featurelatent-space)
6. [Duplicate & Near-Duplicate Detection](#6-duplicate--near-duplicate-detection)
7. [Image Quality Assessment](#7-image-quality-assessment)
8. [Cross-Frame Consistency (Sequence-Aware)](#8-cross-frame-consistency-sequence-aware)
9. [Background / Negative Image Validation](#9-background--negative-image-validation)
10. [Statistical Distribution Integrity](#10-statistical-distribution-integrity)
11. [Synthetic Validation & Meta-Cleaning](#11-synthetic-validation--meta-cleaning)
12. [Priority Matrix & Recommended Implementation](#12-priority-matrix--recommended-implementation)

---

## 1. Standard Integrity Checks

These are entry-level, non-negotiable, and must run before anything else.

| # | Check | Method | Our Data Concern |
|---|-------|--------|-----------------|
| 1.1 | Corrupt images | PIL `Image.open().verify()` on all .jpg files | 0 found in Task 2 sampling, but full 4,954 scan needed |
| 1.2 | Zero-byte files | `os.path.getsize() == 0` | Watchtower cameras might write partial frames |
| 1.3 | Image-label pairing | Cross-reference `X-Images/` vs `X-Labels/` by stem name | 4,693 images with labels + 256 empty = 4,949. Total is 4,954. 5 images unaccounted |
| 1.4 | YOLO format compliance | Validate 5 floats per line, class ID = 0 only | Trivial check, catches parsing errors |
| 1.5 | Coordinate bounds | Normalized [0,1]; denormalized within [0,W]x[0,H] | Essential for COCO conversion in Task 4 |
| 1.6 | Degenerate boxes | `bw <= 0 or bh <= 0` (zero-area) | Wastes loss computation, skews mAP |
| 1.7 | Empty images have empty labels | Check Empty-Labels/*.txt are either absent or zero-length | Violates our 5.17% negative ratio calculation |

**Business Impact:** A single corrupt image halts training on T4/Colab GPU. A single out-of-bounds box crashes COCO evaluation. These are P0 blockers, not optional.

**Implementation:** 7 cells in Task 3 notebook. ~30 seconds to run on full dataset.

---

## 2. Label Noise Detection

### 2.1 Confident Learning (Northcutt et al., 2021)
**What it is:** Train a quick baseline model, compute out-of-sample predicted probabilities for every sample, then flag samples where the model consistently disagrees with the annotation with high confidence.

**How it works:**
1. Split data into k folds
2. For fold i: train model on all other folds, predict on fold i
3. For each sample: compare predicted class/box with ground truth
4. Samples where model is "confidently wrong" (high predicted prob for different class) are likely mislabeled

**Applicability to us:** MEDIUM. Since we only have one class (smoke), class-level label noise is irrelevant. But box-level noise (wrong position, size, missing boxes) is detectable:
- Train a quick YOLO11n on 80% of data
- Run inference on the held-out 20%
- Flag images where:
  - Model predicts a box but no annotation exists (potential missed annotation)
  - Model predicts NO box but annotation exists (potential false annotation)
  - Model predicts a box with IoU < 0.3 from annotation (potential box error)

**Effort:** Medium. Requires training a lightweight model. ~20 min on T4.

**A+ Link:** A missed smoke annotation in training = model learns to ignore that plume shape. In production, that shape reappears and is missed = forest fire.

### 2.2 CleanLab for Object Detection
**What it is:** A Python library specifically for finding label errors in ML datasets. Provides `Datalab` API.

**How it works:**
```python
from cleanlab import Datalab
lab = Datalab(data={"images": image_paths}, label_name="bboxes")
lab.find_issues(features=image_features, pred_probs=model_predictions)
```

**Applicability:** LOW-MEDIUM. CleanLab's object detection support is less mature than classification. However, extracted features from a pretrained ResNet-50 backbone can still flag unusual samples.

**Effort:** Low (library handles most logic), but reliability for bbox regression is questionable.

### 2.3 Box Confidence Scoring with Pretrained Model
**What it is:** Use a model pretrained on COCO or similar to "verify" annotations. If a pretrained general object detector fails to detect anything in your bbox region, the annotation might be wrong.

**How it works:**
1. Crop each bbox region from the image
2. Run through a general object detector (e.g., YOLOv8 pretrained on COCO)
3. An empty crop where YOLO detects nothing = potentially bad annotation
4. A crop where YOLO detects "bird", "cloud" etc. = annotation might need review

**Applicability:** LOW. COCO has no "smoke" class. The pretrained model won't detect smoke. But it could detect if a box contains nothing but sky/forest (no object at all) — useful for catching completely empty boxes.

**Effort:** Low. `ultralytics` pip install, inference only. ~5 minutes.

---

## 3. Bounding Box Quality Assessment

### 3.1 Box Tightness Analysis (Edge-Based)
**What it is:** A well-annotated box should tightly enclose the object, with box edges landing on natural image gradients (edges). If a box is loose, it includes excess background that confuses the classifier head.

**How it works:**
1. Expand each bbox by 10% on all sides
2. Inside the expanded box, compute edge density (Canny edges)
3. Compute the ratio: (edges on box boundary) / (edges inside box)
4. Loose boxes: low boundary-to-interior edge ratio (object is in center, box goes far beyond)
5. Tight boxes: high ratio (box boundary aligns with object edges)

**Applicability to us:** MEDIUM-HIGH. Given that Task 2 found 95.7% of plumes are large (>10% area), many boxes may be loose (annotators drew large boxes around obvious smoke). Loose boxes hurt localization mAP specifically at high IoU thresholds (mAP@0.75).

**Effort:** Medium. OpenCV Canny + manual threshold tuning.

**A+ Link:** A loose box adds sky/forest background as "smoke features." The model learns that blue sky = smoke signal. This produces false positives on clear-sky empty images.

### 3.2 Box Region Texture Analysis
**What it is:** Compare texture statistics inside vs. outside the bounding box. If the texture is identical, the box encloses no distinct object.

**How it works:**
1. Compute GLCM (Gray-Level Co-occurrence Matrix) features for: (a) interior bbox region, (b) border around bbox
2. Compare contrast, homogeneity, energy, correlation
3. If interior and exterior are statistically indistinguishable — box likely empty or wrong

**Applicability:** MEDIUM. Smoke has distinct texture from forest/sky. Empty boxes would be caught quickly. However, thin diffuse smoke at horizon may have similar texture to clouds/haze.

**Effort:** Medium. Requires skimage or manual GLCM implementation.

### 3.3 GrabCut-Based Box Refinement
**What it is:** Use GrabCut interactive segmentation to refine loose bboxes into tight segmentation masks, then compute new tight bbox from the mask.

**How it works:**
1. Use annotated bbox as GrabCut rectangle seed
2. Run GrabCut to segment foreground (smoke) vs background
3. Compute new minimal bbox from the segmentation mask
4. If new box area is <50% of original — original box was very loose
5. Option: replace original box with tighter version

**Applicability:** LOW. GrabCut assumes clear foreground/background separation with color differences. Smoke is semi-transparent and blends into sky/forest. GrabCut will struggle. Plus, re-bounding all 4,862 boxes is slow (30 sec per image).

**Effort:** High. Computationally expensive, unreliable for transparent objects.

### 3.4 Otsu-Based Smoke Segmentation Check
**What it is:** Apply Otsu's thresholding within the bbox region to see if there's a bimodal intensity distribution (smoke vs. clear sky).

**How it works:**
1. Crop bbox region from grayscale image
2. Apply Otsu's thresholding
3. If Otsu fails to find two distinct intensity peaks (low bimodality) — box may be on uniform background (empty or wrong)
4. High bimodality = likely contains distinguishable object

**Applicability:** MEDIUM. Smoke against sky creates a bimodal intensity distribution (white/gray smoke vs blue/gray sky). Against forest background, smoke is harder to distinguish. Results vary by background type.

**Effort:** Low. Pure OpenCV.

---

## 4. Annotation Consistency Validation

### 4.1 Box Count Consistency Across Location Clips
**What it is:** Within a single video clip (`evoDJI_0001`), annotations should follow consistent patterns. If frame 1-50 have 1 box each, but frame 33 has 0 or 3, it's suspicious.

**How it works:**
1. Group frames by video clip prefix
2. For each clip, count boxes per frame
3. Flag frames where box count deviates from clip mode by >1

**Applicability:** HIGH. We already identified 30 video clips. 99.5% of images are single-box. A 2-box image in a sequence of 1-box frames is almost certainly an annotation error.

**Effort:** Low. Trivial with existing video_group logic from Task 2.

### 4.2 Consecutive Frame Box Similarity Threshold
**What it is:** In a video, consecutive frames should have nearly identical boxes. If box on frame N has IoU < 0.9 with box on frame N+1, something is wrong.

**How it works:**
1. For each clip, sort frames by number
2. Compute IoU between consecutive frame boxes
3. Flag frames where IoU < 0.5 (box "jumped")

**Applicability:** HIGH. We already computed frame displacement (mean 0.041 normalized). IoU-based check is the natural complement. Large IoU drops = annotation errors or major scene changes.

**Effort:** Low. Reuses existing frame grouping logic.

### 4.3 Inter-Frame Box Value Interpolation Check
**What it is:** For a box on frame N, linearly interpolate between frames N-1 and N+1 to predict expected position. Compare prediction to actual. Large deviation = annotation error.

**How it works:**
1. For each clip, iterate triplets: (frame N-1, frame N, frame N+1)
2. Interpolate box center and size linearly: `expected = avg(box_{N-1}, box_{N+1})`
3. Compute IoU between expected and actual box at frame N
4. Flag if IoU < 0.7

**Applicability:** HIGH. Far more sensitive than pairwise IoU. Catches single-frame annotation drift/deletion. Especially valuable because smoke moves slowly and predictably.

**Effort:** Low. Pure numpy math.

---

## 5. Outlier Detection in Feature/Latent Space

### 5.1 Pretrained Backbone Embedding + DBSCAN
**What it is:** Extract features from cropped bbox regions using a pretrained model (ResNet-50, ViT), then cluster in feature space. Outliers that don't belong to any cluster are suspicious.

**How it works:**
1. Load pretrained ResNet-50 (up to avgpool layer, 2048-dim output)
2. Crop and resize all bbox regions to 224x224
3. Run through ResNet, collect 2048-dim feature vectors
4. Apply DBSCAN (eps tuned by k-distance graph, min_samples=5)
5. Samples labeled as noise (cluster=-1) are outliers

**Applicability:** MEDIUM-HIGH. 4,862 boxes is manageable. Features capture visual similarity of smoke patterns. Outliers could be:
- Misannotated (box on cloud, bird, lens flare, etc.)
- Rare smoke types (thin, wispy, unusual conditions) — not errors, but worth reviewing
- Artifacts (dust on lens, sensor noise mistaken as smoke)

**Effort:** Medium. torchvision models, ~10 min on CPU, ~2 min on GPU.

### 5.2 Autoencoder Reconstruction Error
**What it is:** Train a small convolutional autoencoder on all bbox crops. Samples with abnormally high reconstruction error don't match the learned "smoke manifold."

**How it works:**
1. Crop and resize all bbox regions to 128x128
2. Train small autoencoder (Conv→Bottleneck→Deconv) for 50 epochs
3. Compute reconstruction MSE for every sample
4. Flag samples with MSE > μ + 3σ

**Applicability:** MEDIUM. More sensitive to texture anomalies than DBSCAN. Catches boxes that contain objects with fundamentally different texture from smoke (e.g., a bird inside a box accidentally labeled as smoke).

**Effort:** Medium-High. Requires training. ~30 min.

### 5.3 t-SNE/UMAP Manual Visualization
**What it is:** Project 2048-dim feature vectors to 2D for visual inspection. Human can spot clusters and outliers.

**How it works:**
1. Use same features as Section 5.1
2. Apply UMAP (faster than t-SNE for 6K samples)
3. Plot scatter, color by location (Evo, Heinola, etc.)
4. Manually inspect outliers

**Applicability:** LOW for automated cleaning, HIGH for research insight. Best used as a "sanity check" after automated methods, not as primary cleaning.

**Effort:** Medium. umap-learn, interactive plotting.

---

## 6. Duplicate & Near-Duplicate Detection

### 6.1 Exact MD5 Hash (Already Planned)
**What it is:** Byte-for-byte identical files.

**Applicability:** Trivial. Must-run.

### 6.2 Perceptual Hash (pHash)
**What it is:** A hash function where visually similar images produce similar hashes (unlike MD5 where one bit difference = completely different hash).

**How it works:**
1. Resize image to 32x32 grayscale
2. Compute DCT (Discrete Cosine Transform)
3. Keep top-left 8x8 of DCT (low frequencies)
4. Compare to median → binary hash
5. Hamming distance < 5 between two hashes = near-duplicate

**Applicability to us:** HIGH. With 30 video clips at multiple FPS, there will be near-identical consecutive frames (e.g., smoke barely moved, identical lighting). These near-duplicates cause:
- Over-representation of some smoke shapes
- "Soft" temporal leakage even with clip-level splits (if a clip has many near-frozen frames, the model overfits that specific scene)

**Effort:** Low. `imagehash` library. <30 seconds on 5K images.

### 6.3 Feature-Space Duplicate Detection (CNN Embedding Distance)
**What it is:** Two frames can have different hashes (different exact pixels) but be semantically identical. Use CNN embeddings + cosine similarity to find them.

**How it works:**
1. Extract ResNet-50 features for every full image (not just bbox)
2. Compute pairwise cosine similarity within each clip
3. Flag frame pairs with cosine sim > 0.99 (near-semantically-identical)

**Applicability:** HIGH. More robust than pHash for outdoor scenes (clouds move, but scene remains identical). Helps identify redundant frames that add no training value but contribute to overfitting.

**Effort:** Medium. Same as Section 5.1 pipeline.

---

## 7. Image Quality Assessment

### 7.1 Blur Detection (Laplacian Variance)
**What it is:** Compute variance of Laplacian filter response. High variance = sharp edges = in focus. Low variance = blurry.

**How it works:**
```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
# Threshold: < 50 = blurry (for 4K image)
```

**Applicability to us:** HIGH. Watchtower cameras suffer from:
- Heat haze (atmospheric distortion on hot days)
- Motion blur (wind shaking the camera mount)
- Out-of-focus (autofocus hunting on horizon)
- Blurry frames produce smeared smoke features → poor detector performance

Our Task 2 found: 4K resolution. Blur is even more critical at this resolution because blur removes the fine texture edges that distinguish distant smoke from clouds.

**Effort:** Low. Pure OpenCV, 1 line per image.

### 7.2 Contrast / Visibility Analysis
**What it is:** Smoke against sky requires contrast to be visible. Low-contrast smoke (thin, diffuse, same color as sky) is effectively invisible and unannotatable. If annotations exist on invisible smoke, they're unreliable.

**How it works:**
1. Compute local contrast within bbox: RMS contrast = `std(pixel_values)`
2. Compute Weber contrast: `(I_object - I_background) / I_background` where I_background = pixels just outside bbox
3. Flag boxes with contrast < threshold

**Applicability:** HIGH. Thin smoke at the horizon has near-zero contrast against pale sky. Annotators may still draw boxes, but the model can't learn from these — they're noise in training data.

**Effort:** Low. Numpy stats on cropped regions.

### 7.3 Smoke Visibility Score (Composite Metric)
**What it is:** Combine blur, contrast, and box size into a single "visibility score" that ranks each annotation by how learnable it is.

**How it works:**
```
visibility_score = α * (1 - blur_rank) + β * contrast_rank + γ * size_rank
```
Where α, β, γ are weights summing to 1. Low visibility scores = the model physically cannot learn from this box.

**Applicability:** MEDIUM. Useful for stratified analysis (e.g., "our model fails on low-visibility smoke" — is that a model problem or a data problem?). Good for the research paper.

**Effort:** Low (composite of already-computed metrics).

### 7.4 JPEG Compression Artifact Detection
**What it is:** JPEG compression creates block artifacts (8x8 blocks). Heavy compression can make small smoke plumes indistinguishable from compression noise.

**How it works:**
1. Compute blockiness metric: variance of DCT coefficients at block boundaries vs interior
2. High blockiness = heavy compression

**Applicability:** LOW. 4K images at reasonable JPEG quality won't have severe artifacts. Only relevant if the dataset was re-compressed at some point.

**Effort:** Medium. Custom DCT analysis.

### 7.5 Overexposure / Underexposure Detection
**What it is:** Clipped pixels (pure white = 255, pure black = 0) mean information is lost. Textures in over/under exposed regions cannot be recovered.

**How it works:**
1. Count pixels at intensity 0 (underexposed) and 255 (overexposed)
2. Flag images where >5% of pixels are clipped

**Applicability:** LOW. Task 2 found 0% overexposed, average brightness 110/255. Not a concern.

**Effort:** Trivial. 2 lines of numpy.

---

## 8. Cross-Frame Consistency (Sequence-Aware)

### 8.1 Optical Flow Box Validation
**What it is:** Use dense optical flow to predict where each pixel moves from frame N to frame N+1. Apply this to the box corners — the predicted box should match the annotation on frame N+1.

**How it works:**
1. Compute Farneback optical flow between frame N and frame N+1
2. Track the 4 corners of the bbox on frame N through the flow field
3. Arrive at a predicted bbox for frame N+1
4. Compute IoU between predicted box and actual annotation
5. IoU < 0.5 = annotation error on frame N+1

**Applicability to us:** HIGH. The single most sophisticated check for sequential data. Catches:
- Frame where annotation is missing (model predicts box, no annotation = missed smoke)
- Frame where annotation is wrong (flow predicts different location)
- Frame where new smoke appears (flow correctly shows no prediction, box exists = actually a new plume)

**Effort:** Medium. OpenCV farneback, but sensitive to parameter tuning.

### 8.2 Box Trajectory Smoothness (Kalman Filtering)
**What it is:** A Kalman filter models the expected smooth trajectory of a moving smoke plume. Actual box positions that deviate significantly from the Kalman prediction are suspicious.

**How it works:**
1. Initialize Kalman filter with motion model (constant velocity)
2. Feed box centers frame by frame through each clip
3. After each observation, check if actual position is within 3σ of Kalman prediction
4. Outliers = potential annotation errors

**Applicability:** HIGH. More principled than simple interpolation. Handles occlusion and new plume appearance naturally (large innovation = new object, not error).

**Effort:** Medium-High. Requires filterpy or manual Kalman implementation.

### 8.3 Scene Change Detection
**What it is:** Detect when the camera is moved, panned, or scene changes abruptly. Annotations across scene boundaries cannot be compared for consistency.

**How it works:**
1. Compute frame-to-frame structural similarity (SSIM) or MSE
2. Large SSIM drop = scene change
3. Box consistency checks should NOT span scene boundaries

**Applicability:** MEDIUM. Watchtower cameras might have scheduled pans or preset positions. If scene boundaries exist within a clip, our splitting logic and consistency checks need to respect them.

**Effort:** Low. skimage SSIM.

---

## 9. Background / Negative Image Validation

### 9.1 Empty Image Content Verification
**What it is:** Verify that "Empty-Images" actually contain no smoke. A single mislabeled smoke image in the empty set = False Negative by definition.

**How it works:**
1. Train a quick smoke classifier on known positive crops
2. Run sliding window detection on all empty images
3. Flag any empty image where classifier fires with high confidence

**Applicability:** MEDIUM. 256 images is small enough for manual review if needed. But automated check is good practice.

**Effort:** Medium. Requires training a binary classifier.

### 9.2 Motion Detection on Empty Images
**What it is:** Even without smoke, a watchtower camera captures motion (birds, clouds, swaying trees). Motion artifacts in "empty" images can cause false positives during inference.

**How it works:**
1. Compute frame difference between consecutive empty frames
2. If significant motion exists (bird, vehicle), the image is "not truly empty" — contains a distracter object
3. Either remove from empty set or add annotation for the distracter

**Applicability:** LOW. Empty images are from a curated "Empty-Images" folder, likely pre-verified. Not a major concern.

**Effort:** Low.

---

## 10. Statistical Distribution Integrity

### 10.1 Box Size Distribution per Location
**What it is:** Check if any location has anomalous box size distribution. If Heinola has systematically smaller boxes than Evo, that's a feature of the location (different camera distance). If one clip within a location has dramatically different box sizes, it might be annotation error.

**How it works:**
1. Group box sizes by location and by clip
2. Apply Kruskal-Wallis test (non-parametric ANOVA) across clips within each location
3. Flag clips that are statistically anomalous (p < 0.01)

**Applicability:** MEDIUM. Useful for understanding annotation quality variation across locations. If one location was annotated by a different person/team, quality may differ.

**Effort:** Low. scipy stats.

### 10.2 Box Position Distribution per Clip
**What it is:** Similar to 10.1 but for box center position. If a clip has boxes in the bottom 20% (violating the horizon bias we discovered), it might be misannotated.

**How it works:**
1. Group box centers by clip
2. Compute mean Y-center per clip
3. Flag any clip where mean Y-center > 0.6 (boxes in bottom of image)
4. These violate the physical constraint that smoke rises and watchtowers look horizontal/upward

**Applicability:** MEDIUM-LOW. Our Task 2 spatial heatmap showed smoke in top 40%. Boxes in bottom of image would be suspicious.

**Effort:** Trivial.

---

## 11. Synthetic Validation & Meta-Cleaning

### 11.1 Train-and-Flag (Feedback Loop)
**What it is:** The most pragmatic cleaning method. Train a quick model, use it to flag problematic samples, clean them, retrain, repeat.

**How it works:**
1. Train YOLO11n on full dataset (10 epochs, quick)
2. Compute per-image training loss (yes, YOLO can do this)
3. Sort images by loss descending
4. Top 1-2% highest-loss images are likely mislabeled
5. Manually review top-50 images
6. Fix or remove, retrain, check improvement

**Applicability to us:** HIGH. This is the "brute force" method that catches everything — format errors, box errors, missing annotations, corrupt images without needing to write specific checks for each.

**Effort:** Low-Medium. 10-epoch YOLO11n on T4 = ~5 minutes.

**A+ Link:** This method is model-aware. It doesn't flag "unusual smoke" — it flags "smoke the model cannot learn from." If the model can't learn it, adding it to training data is just noise.

### 11.2 Active Learning Cleaning Loop
**What it is:** Variant of 11.1. Use uncertainty sampling to prioritize which samples to manually review.

**How it works:**
1. Train ensemble of 3 small models (different seeds)
2. For each image, compute ensemble disagreement (variance in predicted boxes)
3. High-variance samples = model uncertain = likely annotation issue or edge case
4. Review top-N by uncertainty

**Applicability:** LOW-MEDIUM. Overengineered for 5K images. More relevant for 100K+ datasets.

**Effort:** Medium-High.

---

## 12. Priority Matrix & Recommended Implementation

### Legend
- **P0** (Must-have, non-negotiable baseline)
- **P1** (High value, implement in Task 3)
- **P2** (Valuable, implement if time permits)
- **P3** (Nice-to-have, academic interest only)

| Section | Technique | Priority | Rationale | Implementation Effort | Runtime |
|---------|-----------|----------|-----------|----------------------|---------|
| 1.1-1.7 | Standard integrity checks | **P0** | Catches crashes, training halts | Low (7 cells) | <1 min |
| 6.1 | MD5 exact duplicates | **P0** | Basic dedup | Trivial | <1 min |
| 6.2 | Perceptual hash (pHash) | **P1** | Near-identical frames inflate representation | Low | <1 min |
| 7.1 | Blur detection | **P1** | Blurry 4K frames produce useless features | Low | <1 min |
| 4.1 | Box count consistency per clip | **P1** | Multi-box anomaly detection | Low | <1 min |
| 4.2 | Consecutive frame IoU | **P1** | Catches annotation "jumps" in video | Low | <1 min |
| 7.2 | Contrast analysis | **P1** | Invisible smoke = unlearnable | Low | <1 min |
| 11.1 | Train-and-flag (YOLO quick-scan) | **P1** | Catches everything else in one pass | Medium | ~5 min |
| 4.3 | Frame interpolation check | **P1** | More sensitive than pairwise IoU | Low | <1 min |
| 8.1 | Optical flow box validation | **P2** | Sophisticated video-based check | Medium | ~10 min |
| 5.1 | ResNet + DBSCAN outlier detection | **P2** | Catches mislabeled non-smoke objects | Medium | ~10 min |
| 6.3 | Feature-space duplicate detection | **P2** | Semantic dedup beyond pHash | Medium | ~10 min |
| 7.3 | Smoke visibility composite score | **P2** | Good for research paper analysis | Low | <1 min |
| 8.2 | Kalman filter trajectory | **P2** | Principled motion validation | Medium-High | ~10 min |
| 3.1 | Box tightness (edge-based) | **P3** | Loose box analysis | Medium | ~5 min |
| 3.2 | Box region texture (GLCM) | **P3** | Academic interest | Medium | ~5 min |
| 8.3 | Scene change detection | **P3** | SSIM-based | Low | <1 min |
| 10.1 | Statistical distribution per location | **P3** | Academic interest | Low | <1 min |
| 2.1 | Confident learning | **P3** | Overkill for single-class | Medium | ~20 min |
| 2.3 | COCO pretrained box verification | **P3** | COCO has no smoke class | Low | ~5 min |
| 3.3 | GrabCut box refinement | **P3** | Fails on transparent smoke | High | ~2 hours |
| 5.2 | Autoencoder reconstruction | **P3** | Overkill for 5K images | Medium-High | ~30 min |
| 9.1 | Empty image smoke check | **P3** | 256 images, manual review possible | Medium | ~5 min |
| 11.2 | Active learning ensemble | **P3** | Overengineered for dataset size | Medium-High | ~20 min |

---

## Summary: Recommended Task 3 Execution Plan

### Phase A — P0 Baseline (Must run first, ~30 seconds)
1. Image integrity: corrupt, zero-byte, unreadable
2. Label pairing: orphans, missing labels, empty-image label violations
3. YOLO format compliance: field count, class ID, float parsing
4. Coordinate bounds: normalized + denormalized
5. Degenerate boxes: zero-area, tiny boxes
6. MD5 duplicates

### Phase B — P1 High-Value (Run after Phase A passes, ~7 minutes)
7. pHash near-duplicate detection (captures redundant video frames)
8. Blur detection (finds useless frames)
9. Box count consistency per video clip
10. Consecutive frame IoU consistency
11. Local contrast within bboxes
12. Frame interpolation validation (3-frame triplets)
13. Train-and-flag with YOLO11n quick-scan (10 epochs, 5 min)

### Phase C — P2 Optional Enhancements (If issues found in B, or for paper quality)
14. Optical flow box validation
15. ResNet feature embedding + DBSCAN
16. Smoke visibility composite score (for paper Figure)
17. Feature-space semantic dedup

### Outputs
- `cleaning_report.md` — summary of all issues found, counts, actions taken
- `cleaned_manifest.json` — list of kept/removed/flagged images with reasons
- Updated `preprocess.ipynb` with all cells, outputs, and business insights
- Statistics: images_removed, images_flagged, percentage_clean

### Files Affected
- Removal: Images moved to `dataset-b/preprocessing/task3_data_cleaning/rejected/` (never deleted — audit trail)
- Flagging: JSON manifest for manual review in `dataset-b/preprocessing/task3_data_cleaning/flagged.json`

---

## Research References

1. Northcutt, C., Jiang, L., & Chuang, I. (2021). "Confident Learning: Estimating Uncertainty in Dataset Labels." *Journal of Artificial Intelligence Research*, 70, 1373-1411.
2. Tkachenko, M., Malyuk, M., Holmanyuk, A., & Liubimov, N. (2022). "Label Studio: Data labeling software." *GitHub repository*.
3. Müller, N. M., & Markert, K. (2019). "Identifying Mislabeled Instances in Classification Datasets." *IJCNN 2019*.
4. Rolnick, D., Veit, A., Belongie, S., & Shavit, N. (2017). "Deep Learning is Robust to Massive Label Noise." *arXiv:1705.10694*.
5. Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). "Understanding deep learning requires rethinking generalization." *ICLR 2017*. — The seminal paper on label noise and memorization.
6. Kolesnikov, A., Beyer, L., Zhai, X., Puigcerver, J., Yung, J., Gelly, S., & Houlsby, N. (2020). "Big Transfer (BiT): General Visual Representation Learning." *ECCV 2020*. — Pretrained features for outlier detection.
7. Chen, G., Song, Y., Wang, F., Zhang, Z., Wang, Y., Hu, Z., & Sun, L. (2021). "Semi-supervised Learning for Object Detection with Perturbation Consistent Teacher." *arXiv:2108.09162*.
8. Zaheer, M. Z., Lee, J. H., Astrid, M., & Lee, S. I. (2022). "Cleaning Label Noise with Clusters for Minimally Supervised Anomaly Detection." *CVPR Workshop 2022*. — Relevant to finding anomalous annotations in single-class settings.
9. Li, J., Socher, R., & Hoi, S. C. H. (2020). "DivideMix: Learning with Noisy Labels as Semi-supervised Learning." *ICLR 2020*. — State-of-the-art label noise handling.
