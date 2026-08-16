import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import cv2

def main():
    script_dir = Path(__file__).resolve().parent
    checkpoint_dir = script_dir / "Cognitive_Fire_Defense" / "DETR" / "checkpoint-8613"
    coco_val_json = script_dir.parent.parent / "coco_format" / "annotations" / "val.json"
    coco_val_images = script_dir.parent.parent / "coco_format" / "images" / "val"

    print(f"Using checkpoint: {checkpoint_dir}")

    print("Loading processor and model...")
    processor = AutoImageProcessor.from_pretrained(checkpoint_dir)
    model = AutoModelForObjectDetection.from_pretrained(checkpoint_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("Loading validation dataset...")
    with open(coco_val_json, "r") as f:
        coco_data = json.load(f)
    
    images = coco_data["images"]
    annotations = coco_data["annotations"]
    
    # Build a fast lookup for annotations
    img_to_anns = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    print("Running evaluation...")
    for img_info in tqdm(images):
        img_id = img_info["id"]
        img_path = coco_val_images / img_info["file_name"]
        
        if not img_path.exists():
            continue

        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ground truth
        anns = img_to_anns.get(img_id, [])
        gt_boxes = []
        gt_labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(ann["category_id"])
            
        target = [
            dict(
                boxes=torch.tensor(gt_boxes, dtype=torch.float32).to(device) if len(gt_boxes) > 0 else torch.empty((0, 4), dtype=torch.float32).to(device),
                labels=torch.tensor(gt_labels, dtype=torch.int64).to(device)
            )
        ]
        
        # Prediction
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        target_sizes = torch.tensor([image.shape[:2]]).to(device)
        # We use a low threshold for evaluation so mAP calculation has access to low-confidence boxes for the PR curve
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.001)[0]
        
        pred = [
            dict(
                boxes=results["boxes"],
                scores=results["scores"],
                labels=results["labels"]
            )
        ]
        
        metric.update(pred, target)

    print("\nComputing metrics (this may take a minute)...")
    results = metric.compute()
    
    print("\n" + "="*50)
    print("Deformable DETR - COCO mAP Metrics (Validation Set)")
    print("="*50)
    print(f"mAP (0.50:0.95): {results['map'].item():.4f}")
    print(f"mAP@0.50       : {results['map_50'].item():.4f}")
    print(f"mAP@0.75       : {results['map_75'].item():.4f}")
    print(f"mAR (100 dets) : {results['mar_100'].item():.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
