# Task 4: Data Transformation & Standardization

> **Objective:** Standardize test probe images to $640\times640$ resolution and generate multi-framework annotation assets.

---

## 1. Transformations Applied

1. **Resolution Standardization ($640 \times 640$):**
   * Resizes all 637 probe images via bilinear interpolation (`cv2.INTER_LINEAR`).
   * Output directory: `dataset-a/processed/images/`
2. **YOLO TXT Standardization:**
   * Validated bounding box center $(x, y)$ and extent $(w, h)$ normalized to $[0.0, 1.0]$.
   * Output directory: `dataset-a/processed/labels/`
3. **COCO JSON Generation:**
   * Converts YOLO annotations into standard COCO format for two-stage and transformer detectors (Faster R-CNN, Deformable DETR).
   * Bounding box format: `[x_min, y_min, width, height]` in absolute pixel coordinates ($640\times640$).
   * Output file: `dataset-a/processed/annotations/test_coco.json` (1,891 total ground-truth annotations).
