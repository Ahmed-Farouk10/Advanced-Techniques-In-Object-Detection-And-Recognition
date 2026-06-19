# Project Premortem: Phase 0 & Revised Plan Execution

**Project:** AIN7601 Cognitive Fire Defense
**Focus:** Smoke→Fire Cross-Domain Transfer Learning

This premortem documents critical risks and anticipated failure modes BEFORE we execute the data processing and training phases.

## 1. The Temporal Leakage Risk (CRITICAL)

> **The Flaw:** The Boreal Watchtower dataset (Dataset B) consists of sequential video frames (e.g., `evoDJI_0001_frame0`, `evoDJI_0001_frame1`, etc.). If we perform a naive randomized 70/15/15 train/val/test split across all images, we will inevitably place adjacent frames of the exact same physical scene into both the training and validation sets.
> 
> **The Consequence:** The model will not learn to detect "smoke"; it will learn to recognize "the trees in evoDJI_0001" and map that to a high confidence score. Validation metrics (mAP, Precision, Recall) will be artificially inflated (near 99%), but the model will fail entirely on new data. This is called "temporal leakage" or "data snooping".
> 
> **The Mitigation:** We MUST split the data at the **video/clip level**. All frames belonging to `evoDJI_0001` must remain strictly together in either train, val, or test.

## 2. Test Cases & Verification Plan

Before calling any phase complete, the following test cases must pass:

### Dataset B Split Verification
- [ ] **Test:** Ensure zero cross-over of video prefix IDs (e.g., `evoDJI_0001`) between train, val, and test splits.
- [ ] **Test:** Check that the 70/15/15 ratio is roughly maintained across the *number of images* despite being split by *clip ID*.
- [ ] **Test:** Validate that the empty (no-smoke) images are distributed proportionally.
- [ ] **Test:** After pHash dedup in Task 3, verify that no near-duplicate frame pairs (Hamming < 5) exist across train/val boundaries from different clips of the same location.

### Label Integrity (Task 3)
- [ ] **Test:** Ensure all converted COCO JSON bounding boxes fall strictly within [0, image_width] and [0, image_height].
- [ ] **Test:** Verify YOLO TXT coordinates are normalized strictly between [0.0, 1.0].
- [ ] **Test:** Ensure Empty-Images mapped to training contain zero bounding boxes in their YOLO TXT and COCO JSON representations.
- [ ] **Test:** After Task 3 cleaning, verify the cleaning log sums correctly: Retained + Removed = 4,954.
- [ ] **Test:** Final validation: zero remaining corrupt images (PIL.verify passes), zero invalid YOLO format lines, zero degenerate boxes (w≤0 or h≤0).
- [ ] **Test:** Verify the 5 unaccounted images (4,954 - 4,693 annotated - 256 empty) are resolved — either paired with labels, confirmed as intentional negatives, or removed.

### Model Evaluation on Dataset A
- [ ] **Test:** Verify that predicting ANY bounding box > confidence threshold maps correctly to the `fire` classification class.
- [ ] **Test:** Run inference on `nofire` images and ensure false alarm rate is calculated properly (predicting a box on a `nofire` image = False Positive).
- [ ] **Test:** Report APsmall, APmedium, APlarge separately. The primary risk is APsmall = 0% while mAP looks acceptable. Per-size metrics expose this.
- [ ] **Test:** Evaluate at multiple brightness levels on Dataset A (day/dusk). The illumination bias risk requires testing whether the model degrades in dark conditions.
- [ ] **Test:** Evaluate YOLO11n anchor alignment: after training on custom anchors from Task 2 clustering, compare mAP against default COCO anchors.

## 3. Risks Discovered During Task 2 (EDA)

### 3.1 Large Plume Bias — Scale Generalization Failure
> **The Flaw:** 95.7% of smoke plumes occupy >10% of the image. Only 1.3% are small (<1% area). The model will learn that "smoke = large foreground object."
>
> **The Consequence:** Early, distant smoke at 5km+ occupies <1% of the frame. The model will fail to detect the very cases the watchtower is designed to catch. High validation mAP will hide this — a model detecting 100% of large plumes and 0% of small plumes still scores well on average.
>
> **The Mitigation:** Aggressive scale/crop augmentations in Task 5 (RandomResizedCrop, zoom-out paste-in), plus report APsmall separately from APlarge.

### 3.2 4K Resolution Bottleneck — Downsampling Destroys Small Objects
> **The Flaw:** All images are 4096×2160. Standard YOLO input is 640×640. A 21× downsampling factor means a plume occupying 1% of a 4K frame (40×21 pixels) compresses to ~4×2 pixels — below the detection floor of any backbone.
>
> **The Consequence:** Simple resizing eliminates small smoke entirely. Training on resized images produces a model that has never seen small smoke at all.
>
> **The Mitigation:** Random cropping from 4K frames (extract 640×640 tiles) instead of whole-image resize. This preserves full-resolution smoke textures.

### 3.3 Daytime Illumination Bias — Night/Dusk Blindness
> **The Flaw:** 96% of images are bright (mean 112/255). Only 4% are dark. The model learns smoke features under daytime overcast conditions exclusively.
>
> **The Consequence:** At dawn, dusk, or night, the same smoke plume has completely different pixel statistics. The model will produce false negatives after sunset — exactly when watchtower monitoring is most critical (human observers are off-duty or visibility is poor).
>
> **The Mitigation:** Brightness/contrast/hue jitter in Task 5, but augmentation is synthetic and cannot fully compensate for real low-light smoke textures. Consider testing on Dataset A fire images at multiple brightness levels.

### 3.4 Horizon Bias — Vertical Structure Violation
> **The Flaw:** Smoke appears in the top 40% of the image (mean Y-center = 0.395). The bottom 50% is foreground forest, never annotated as smoke.
>
> **The Consequence:** If vertical flip augmentation is accidentally enabled, the model learns that smoke can originate from the ground — violating the physical prior that smoke rises. This would produce nonsensical bounding boxes on inverted test images.
>
> **The Mitigation:** Vertical flip must be explicitly disabled in the augmentation pipeline. (Already documented, but listed here for completeness as a confirmed risk.)

### 3.5 Single-Box Dominance — NMS Behavior is Untested
> **The Flaw:** 99.5% of images contain exactly one bounding box. Only 23 images have 2+ boxes (max 12). The NMS module will be exercised almost exclusively on synthetic multi-object scenes from Mosaic/MixUp augmentations.
>
> **The Consequence:** If Mosaic is enabled during training but NMS parameters are tuned on single-box validation data, the model may under-suppress or over-suppress boxes when tested on real multi-plume scenes (e.g., Dataset A fire with multiple clusters).
>
> **The Mitigation:** Explicitly test NMS behavior on artificially constructed multi-box images. Document NMS threshold assumptions.

### 3.6 Near-Duplicate Frame Risk ("Soft" Temporal Leakage)
> **The Flaw:** Within a single video clip, consecutive frames may be near-identical (smoke displacement ~4% per frame). Perceptual hash (pHash) Hamming distance < 5 between adjacent frames makes them functionally duplicates. Even with clip-level splits, if a clip has 200 near-identical frames in the training set, that specific scene gets 200× the representation weight.
>
> **The Consequence:** Over-representation of specific forest backgrounds, weather conditions, and smoke shapes. The model overfits to the visual features of those specific scenes, reducing generalization to new watchtower locations.
>
> **The Mitigation:** pHash-based deduplication in Task 3. Optionally: frame sampling (keep every Nth frame) in Task 4 for clips with low inter-frame displacement.

### 3.7 Low-Contrast Smoke — Unlearnable Annotations
> **The Flaw:** Thin, diffuse smoke at the horizon has near-zero RMS contrast against pale sky. These annotations exist in the dataset but the pixel-level signal is vanishingly weak.
>
> **The Consequence:** Training on near-invisible smoke forces the model to fit noise. The loss oscillates without converging because the model cannot distinguish annotated smoke pixels from unannotated sky pixels. These samples add training time but no signal.
>
> **The Mitigation:** Flag low-contrast boxes in Task 3 cleaning log. Consider removing boxes below visibility threshold or weighting them lower in loss.

### 3.8 Resolution Gap Between Datasets A and B
> **The Flaw:** Dataset B (training) is 4096×2160. Dataset A (zero-shot test) is ~250×250. The feature scales learned from 4K watchtower footage may not transfer to low-resolution drone imagery.
>
> **The Consequence:** Even if the smoke→fire semantic transfer succeeds, the feature-space mismatch between 4K and 250×250 input distributions may cause false negatives. The model's backbone was trained to detect smoke textures at ~20 pixel/mm; drone fire textures are at ~2 pixel/mm — an order of magnitude coarser.
>
> **The Mitigation:** During Task 5, augment training images with aggressive downsampling + upsampling cycles to expose the model to low-resolution inputs. Alternatively: evaluate on Dataset A at multiple resolutions.

## 4. Other Anticipated Risks

- **OOM on DINO:** DINO with 800x800 images will likely run out of memory on a T4 GPU. *Mitigation: Reduce batch size to 2 and use gradient accumulation of 4.*
- **OOM on RT-DETR at high resolution:** RT-DETR with 640×640 inputs from 4K random crops will still have large activation maps in early layers. *Mitigation: Reduce input size to 512 if memory is tight; RT-DETR is less resolution-dependent than pure CNN architectures.*
- **Zero-Shot Failure:** Smoke models may completely fail to trigger on fire images (0% detection rate). *Mitigation: This is acceptable academically; we will sweep confidence thresholds down to 0.1 to study the sensitivity curve.*
- **Negative Ratio Mismatch:** Training set has 5.17% negatives. Dataset A test has 50% negatives (760/760 balanced). The model will be conditioned to rarely output "no detection." On Dataset A, it will produce more false positives than expected. *Mitigation: Account for this in FP rate analysis. The "50% false alarm rate" on Dataset A does not mean the model is broken — it means the model's prior matches training.*
- **Train/Val Split Artifacts from Clip-Level Grouping:** 30 clips of varying sizes (931 to 1,765 images) make it impossible to hit exact 70/15/15. Some clips will be disproportionately large. *Mitigation: Accept approximation. Report actual split ratios in the paper.*"
