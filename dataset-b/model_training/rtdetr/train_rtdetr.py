import os
from ultralytics import YOLO

def train_rtdetr():
    # Load RT-DETR-L (Anchor-Free Query-Based)
    model = YOLO("rtdetr-l.pt")
    
    print("Initiating RT-DETR Training Pipeline...")
    print("Using Custom Augmentations (Mosaic tuned, HSV, Copy-Paste, Scale)")
    
    # Define paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_data = os.path.join(script_dir, "..", "yolo11n", "smoke_data.yaml")
    yaml_hyp = os.path.join(script_dir, "..", "yolo11n", "custom_hyp.yaml")
    
    results = model.train(
        data=yaml_data,
        cfg=yaml_hyp,
        epochs=100,
        imgsz=640,
        batch=4,          # Reduced from 8: RT-DETR-L needs ~3GB/sample on 8GB VRAM
        device=0,
        workers=4,
        optimizer="AdamW",   # Transformer decoder requires AdamW, not SGD
        lr0=0.0001,          # RT-DETR diverges at YOLO default 0.01 (transformer LR)
        lrf=0.01,
        project="Cognitive_Fire_Defense",
        name="RTDETR_Custom_Aug",
        exist_ok=True
    )
    
if __name__ == "__main__":
    train_rtdetr()
