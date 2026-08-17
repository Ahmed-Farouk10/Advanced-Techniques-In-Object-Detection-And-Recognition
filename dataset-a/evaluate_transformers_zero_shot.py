"""Unified Transformer Zero-Shot Evaluation on Preprocessed Dataset A.
Evaluates both RT-DETR and Deformable DETR on standardized 640x640 Dataset A test images
under both Strict Matching (IoU >= 0.50) and Relaxed Alerting (IoU >= 0.10).
"""

import json
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from ultralytics import RTDETR

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_A_IMG = SCRIPT_DIR / "processed" / "images"
DATASET_A_COCO = SCRIPT_DIR / "processed" / "annotations" / "test_coco.json"
RTDETR_WEIGHTS = PROJECT_ROOT / "dataset-b" / "model_training" / "rtdetr" / "runs" / "detect" / "Cognitive_Fire_Defense" / "RTDETR_Custom_Aug" / "weights" / "best.pt"
DETR_CHECKPOINT = PROJECT_ROOT / "dataset-b" / "model_training" / "dino" / "Cognitive_Fire_Defense" / "DETR" / "checkpoint-18183"

if not DETR_CHECKPOINT.exists():
    DETR_CHECKPOINT = PROJECT_ROOT / "dataset-b" / "model_training" / "dino" / "Cognitive_Fire_Defense" / "DETR" / "checkpoint-8613"

CONF_THRESHOLDS = [0.50, 0.30, 0.20, 0.10, 0.05]
IOU_THRESHOLDS = [0.50, 0.10]

def compute_box_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def load_ground_truth(coco_json_path):
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
        
    img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    file_to_fire_boxes = {img["file_name"]: [] for img in coco["images"]}
    
    total_fire_boxes = 0
    total_fire_images = 0
    
    for ann in coco["annotations"]:
        if ann["category_id"] == 0:  # 0 is FIRE
            file_name = img_id_to_file[ann["image_id"]]
            x, y, w, h = ann["bbox"]
            file_to_fire_boxes[file_name].append([x, y, x + w, y + h])
            total_fire_boxes += 1
            
    total_fire_images = sum(1 for boxes in file_to_fire_boxes.values() if len(boxes) > 0)
    print(f"Loaded {len(coco['images'])} test images from COCO.")
    print(f"Found {total_fire_images} fire-positive images with {total_fire_boxes} ground-truth fire boxes.")
    return file_to_fire_boxes, len(coco["images"]), total_fire_images, total_fire_boxes

def evaluate_predictions(predictions, file_to_gt_boxes, total_images, total_fire_images, total_fire_boxes, conf_thresh, iou_thresh):
    """
    Evaluates detector predictions for a given confidence and IoU threshold.
    predictions: dict of {file_name: list of (box, score)}
    """
    total_preds = 0
    tp = 0
    fp = 0
    detected_fire_images = 0
    
    for file_name, gt_boxes in file_to_gt_boxes.items():
        preds = [p for p in predictions.get(file_name, []) if p[1] >= conf_thresh]
        total_preds += len(preds)
        
        gt_matched = [False] * len(gt_boxes)
        img_has_fire_detection = False
        
        for pred_box, pred_score in preds:
            best_iou = 0.0
            best_gt_idx = -1
            
            for g_idx, gt_box in enumerate(gt_boxes):
                iou = compute_box_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
                    
            if best_iou >= iou_thresh:
                if not gt_matched[best_gt_idx]:
                    tp += 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp += 1  # Duplicate detection
                img_has_fire_detection = True
            else:
                fp += 1  # False alarm / missed localization
                
        if img_has_fire_detection and len(gt_boxes) > 0:
            detected_fire_images += 1
            
    fn = total_fire_boxes - tp
    precision = (tp / total_preds) * 100 if total_preds > 0 else 0.0
    recall = (tp / total_fire_boxes) * 100 if total_fire_boxes > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    img_det_rate = (detected_fire_images / total_fire_images) * 100 if total_fire_images > 0 else 0.0
    
    return {
        "Conf": f"{conf_thresh:.2f}",
        "IoU": f"{iou_thresh:.2f}",
        "Preds": total_preds,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": f"{precision:.2f}%",
        "Recall": f"{recall:.2f}%",
        "F1": f"{f1:.2f}%",
        "Images_Det": f"{detected_fire_images}/{total_fire_images}",
        "Image_Det_Rate": f"{img_det_rate:.2f}%",
        "Box_Det_Rate": f"{recall:.2f}%"
    }

def run_rtdetr_eval(file_to_gt_boxes, total_images, total_fire_images, total_fire_boxes):
    print("\n=======================================================")
    print("🚀 Evaluating RT-DETR Zero-Shot Transfer on Dataset A...")
    print(f"Loading weights from: {RTDETR_WEIGHTS}")
    print("=======================================================")
    
    if not RTDETR_WEIGHTS.exists():
        print(f"Error: RT-DETR weights not found at {RTDETR_WEIGHTS}")
        return None
        
    model = RTDETR(str(RTDETR_WEIGHTS))
    predictions = {file_name: [] for file_name in file_to_gt_boxes.keys()}
    
    for file_name in tqdm(file_to_gt_boxes.keys(), desc="RT-DETR Inference"):
        img_path = DATASET_A_IMG / file_name
        if not img_path.exists():
            continue
            
        # Inference with lowest conf threshold to catch all predictions
        results = model.predict(str(img_path), conf=0.01, imgsz=640, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        
        for box, score in zip(boxes, scores):
            predictions[file_name].append((box.tolist(), float(score)))
            
    # Evaluate across thresholds
    rows = []
    for iou in IOU_THRESHOLDS:
        for conf in CONF_THRESHOLDS:
            res = evaluate_predictions(predictions, file_to_gt_boxes, total_images, total_fire_images, total_fire_boxes, conf, iou)
            res["Model"] = "RT-DETR"
            rows.append(res)
            
    return pd.DataFrame(rows)

def run_deformable_detr_eval(file_to_gt_boxes, total_images, total_fire_images, total_fire_boxes):
    print("\n=======================================================")
    print("🦅 Evaluating Deformable DETR Zero-Shot Transfer on Dataset A...")
    print(f"Loading checkpoint from: {DETR_CHECKPOINT}")
    print("=======================================================")
    
    if not DETR_CHECKPOINT.exists():
        print(f"Error: Deformable DETR checkpoint not found at {DETR_CHECKPOINT}")
        return None
        
    processor = AutoImageProcessor.from_pretrained(str(DETR_CHECKPOINT))
    model = AutoModelForObjectDetection.from_pretrained(str(DETR_CHECKPOINT))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    predictions = {file_name: [] for file_name in file_to_gt_boxes.keys()}
    
    for file_name in tqdm(file_to_gt_boxes.keys(), desc="Deformable DETR Inference"):
        img_path = DATASET_A_IMG / file_name
        if not img_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        inputs = processor(images=img_rgb, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        target_sizes = torch.tensor([img_rgb.shape[:2]]).to(device)
        post_processed = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.01)[0]
        
        boxes = post_processed["boxes"].cpu().numpy()
        scores = post_processed["scores"].cpu().numpy()
        
        for box, score in zip(boxes, scores):
            predictions[file_name].append((box.tolist(), float(score)))
            
    rows = []
    for iou in IOU_THRESHOLDS:
        for conf in CONF_THRESHOLDS:
            res = evaluate_predictions(predictions, file_to_gt_boxes, total_images, total_fire_images, total_fire_boxes, conf, iou)
            res["Model"] = "Deformable DETR"
            rows.append(res)
            
    return pd.DataFrame(rows)

def main():
    file_to_gt_boxes, total_images, total_fire_images, total_fire_boxes = load_ground_truth(DATASET_A_COCO)
    
    df_rtdetr = run_rtdetr_eval(file_to_gt_boxes, total_images, total_fire_images, total_fire_boxes)
    df_detr = run_deformable_detr_eval(file_to_gt_boxes, total_images, total_fire_images, total_fire_boxes)
    
    dfs = []
    if df_rtdetr is not None:
        dfs.append(df_rtdetr)
    if df_detr is not None:
        dfs.append(df_detr)
        
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        out_csv = SCRIPT_DIR / "zero_shot_transformer_eval_results.csv"
        df_all.to_csv(out_csv, index=False)
        print(f"\nSaved combined results to: {out_csv}")
        
        print("\n" + "="*80)
        print("SUMMARY: RELAXED LOCALIZATION (IoU >= 0.10) - EARLY WARNING RESULTS")
        print("="*80)
        relaxed_df = df_all[df_all["IoU"] == "0.10"][["Model", "Conf", "Preds", "TP", "FP", "FN", "Precision", "Recall", "F1", "Image_Det_Rate"]]
        print(relaxed_df.to_string(index=False))
        
        print("\n" + "="*80)
        print("SUMMARY: STRICT LOCALIZATION (IoU >= 0.50) - ZERO-SHOT MATCHING RESULTS")
        print("="*80)
        strict_df = df_all[df_all["IoU"] == "0.50"][["Model", "Conf", "Preds", "TP", "FP", "FN", "Precision", "Recall", "F1", "Image_Det_Rate"]]
        print(strict_df.to_string(index=False))

if __name__ == "__main__":
    main()
