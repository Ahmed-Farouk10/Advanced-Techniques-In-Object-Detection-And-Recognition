# Task 4 — Data Transformation (Format Conversion)

> **Phase 2: Data Preparation | Dataset B**  
> **Note:** This directory corresponds to the pipeline spec's "Data Transformation" step. The actual train/val/test SPLITTING logic lives in `../task4_data_splitting/`. This task covers FORMAT CONVERSION after splitting.

## Objective

Convert the split data from YOLO TXT format to COCO JSON format to serve all 4 training frameworks.

## Why Two Formats?

| Format | Used By | Contents |
|--------|---------|----------|
| YOLO TXT | YOLO11n, RT-DETR (Ultralytics) | `yolo_format/images/` + `yolo_format/labels/` |
| COCO JSON | Faster R-CNN (torchvision), DINO (HuggingFace) | `coco_format/images/` + `coco_format/annotations/` |

## Conversion Details

- **Script:** `../../shared/yolo_to_coco.py`
- **Process:** Reads YOLO TXT normalized coordinates → converts to absolute COCO `[x_min, y_min, width, height]`
- **Image handling:** Hard-linked (no duplication) to `coco_format/images/`
- **Validation:** Debug overlay PNGs generated for `train`, `val`, `test` to verify coordinate translation

## Output COCO Statistics

| Split | Images | Annotations |
|-------|--------|-------------|
| Train | 3,066 | 3,022 |
| Val | 926 | 898 |
| Test | 823 | 803 |

## Outputs

- `../../coco_format/annotations/{train,val,test}.json` — COCO annotation files
- `../../coco_format/images/{train,val,test}/` — Hard-linked images
- `debug_train.png`, `debug_val.png`, `debug_test.png` — Bbox overlay validation images

## For First-Time Students

This task is about FORMAT, not content. The data is the same — just represented differently so that PyTorch and HuggingFace can read it. YOLO format uses normalized `[class, cx, cy, w, h]` in `.txt` files. COCO format uses absolute pixel `[x, y, width, height]` in `.json` files with image metadata.
