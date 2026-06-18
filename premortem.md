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

### Label Integrity
- [ ] **Test:** Ensure all converted COCO JSON bounding boxes fall strictly within [0, image_width] and [0, image_height].
- [ ] **Test:** Verify YOLO TXT coordinates are normalized strictly between [0.0, 1.0].
- [ ] **Test:** Ensure Empty-Images mapped to training contain zero bounding boxes in their YOLO TXT and COCO JSON representations.

### Model Evaluation on Dataset A
- [ ] **Test:** Verify that predicting ANY bounding box > confidence threshold maps correctly to the `fire` classification class.
- [ ] **Test:** Run inference on `nofire` images and ensure false alarm rate is calculated properly (predicting a box on a `nofire` image = False Positive).

## 3. Other Anticipated Risks
- **OOM on DINO:** DINO with 800x800 images will likely run out of memory on a T4 GPU. *Mitigation: Reduce batch size to 2 and use gradient accumulation of 4.*
- **Zero-Shot Failure:** Smoke models may completely fail to trigger on fire images (0% detection rate). *Mitigation: This is acceptable academically; we will sweep confidence thresholds down to 0.1 to study the sensitivity curve.*
