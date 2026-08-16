import os
import random
from pathlib import Path
from ultralytics import YOLO
import sys

def create_subset_yaml(dataset_dir, percent=0.2):
    dataset_dir = Path(dataset_dir)
    images_train = list((dataset_dir / "images" / "train").glob("*.jpg"))
    images_val = list((dataset_dir / "images" / "val").glob("*.jpg"))
    
    # 20% subset
    train_subset = random.sample(images_train, max(1, int(len(images_train) * percent)))
    val_subset = random.sample(images_val, max(1, int(len(images_val) * percent)))
    
    train_txt = dataset_dir / "train_subset.txt"
    val_txt = dataset_dir / "val_subset.txt"
    
    with open(train_txt, "w") as f:
        for img in train_subset:
            f.write(f"{img.absolute()}\n")
            
    with open(val_txt, "w") as f:
        for img in val_subset:
            f.write(f"{img.absolute()}\n")
            
    yaml_content = f"""
path: {dataset_dir.absolute()}
train: {train_txt.absolute()}
val: {val_txt.absolute()}
nc: 2
names: ['background', 'smoke']
    """
    
    yaml_path = dataset_dir / "subset_smoke_data.yaml"
    yaml_path.write_text(yaml_content)
    return yaml_path

def run_temporal_leakage_ablation(yaml_path):
    print("--- Running YOLO11n on Clip-Level Split (5% subset) for 1 epoch ---")
    model = YOLO('yolo11n.pt')
    results_clip = model.train(data=str(yaml_path), epochs=1, imgsz=640, batch=4, project="runs", name="ablation_clip", device="cpu")
    
    # Create random split (Temporal leakage)
    print("--- Creating Random Split (Temporal Leakage) ---")
    dataset_dir = Path(yaml_path).parent
    all_imgs = []
    with open(dataset_dir / "train_subset.txt") as f:
        all_imgs.extend(f.read().splitlines())
    with open(dataset_dir / "val_subset.txt") as f:
        all_imgs.extend(f.read().splitlines())
        
    random.shuffle(all_imgs)
    split_idx = int(len(all_imgs) * 0.8)
    rand_train = all_imgs[:split_idx]
    rand_val = all_imgs[split_idx:]
    
    rand_train_txt = dataset_dir / "rand_train.txt"
    rand_val_txt = dataset_dir / "rand_val.txt"
    with open(rand_train_txt, "w") as f: f.write("\n".join(rand_train))
    with open(rand_val_txt, "w") as f: f.write("\n".join(rand_val))
    
    rand_yaml_content = f"""
path: {dataset_dir.absolute()}
train: {rand_train_txt.absolute()}
val: {rand_val_txt.absolute()}
nc: 2
names: ['background', 'smoke']
    """
    rand_yaml_path = dataset_dir / "rand_smoke_data.yaml"
    rand_yaml_path.write_text(rand_yaml_content)
    
    print("--- Running YOLO11n on Random Split (5% subset) for 1 epoch ---")
    model_rand = YOLO('yolo11n.pt')
    results_rand = model_rand.train(data=str(rand_yaml_path), epochs=1, imgsz=640, batch=4, project="runs", name="ablation_random", device="cpu")
    
    return results_clip, results_rand

if __name__ == "__main__":
    yolo_dir = "dataset-b/yolo_format"
    if not Path(yolo_dir).exists():
        print(f"Directory {yolo_dir} not found.")
        sys.exit(1)
        
    yaml_path = create_subset_yaml(yolo_dir, 0.05)
    run_temporal_leakage_ablation(yaml_path)
    print("Temporal Leakage Ablation finished.")
