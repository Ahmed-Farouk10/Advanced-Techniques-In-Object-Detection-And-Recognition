import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def main():
    script_dir = Path(__file__).resolve().parent
    output_root = script_dir / "Cognitive_Fire_Defense" / "DETR"
    checkpoints = sorted([p for p in output_root.glob("checkpoint-*") if p.is_dir()],
                         key=lambda p: int(p.name.split("-")[1]))
    if not checkpoints:
        print(f"Error: No checkpoints found under {output_root}")
        return
    checkpoint_dir = checkpoints[-1]
    coco_val_json = script_dir.parent.parent / "coco_format" / "annotations" / "val.json"
    coco_val_images = script_dir.parent.parent / "coco_format" / "images" / "val"

    print(f"Using checkpoint: {checkpoint_dir}")

    print("Loading processor and model...")
    processor = AutoImageProcessor.from_pretrained(checkpoint_dir)
    model = AutoModelForObjectDetection.from_pretrained(checkpoint_dir)
    model.eval()

    print("Loading validation dataset...")
    with open(coco_val_json, "r") as f:
        coco_data = json.load(f)
    
    images = coco_data["images"]
    
    # Pick 5 random images
    random.seed(42)
    sample_images = random.sample(images, 5)

    for img_info in sample_images:
        img_path = coco_val_images / img_info["file_name"]
        if not img_path.exists():
            continue
            
        print(f"\nProcessing {img_info['file_name']}...")
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare inputs
        inputs = processor(images=image, return_tensors="pt")
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Post-process
        target_sizes = torch.tensor([image.shape[:2]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)
        ax.set_title(f"Predictions: {img_info['file_name']}")
        
        scores = results["scores"].tolist()
        labels = results["labels"].tolist()
        boxes = results["boxes"].tolist()
        
        print(f"Found {len(boxes)} predictions (threshold >= 0.3)")
        
        for score, label, box in zip(scores, labels, boxes):
            xmin, ymin, xmax, ymax = box
            w, h = xmax - xmin, ymax - ymin
            
            # Draw red box for prediction
            rect = plt.Rectangle((xmin, ymin), w, h, fill=False, color="red", linewidth=2)
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f"{model.config.id2label[label]}: {score:.2f}", color="red", weight="bold", backgroundcolor="white")
            
        plt.axis("off")
        
        # Save output
        output_name = f"pred_{img_info['file_name']}"
        plt.savefig(script_dir / output_name, bbox_inches="tight")
        plt.close()
        print(f"Saved visualization to {output_name}")


if __name__ == "__main__":
    main()
