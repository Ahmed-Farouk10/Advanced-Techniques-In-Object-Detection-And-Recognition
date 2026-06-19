import os
import json
import shutil
from pathlib import Path
import cv2

def convert_yolo_to_coco(yolo_dir, coco_dir):
    yolo_dir = Path(yolo_dir)
    coco_dir = Path(coco_dir)
    
    # Create coco directories
    images_out_dir = coco_dir / 'images'
    ann_out_dir = coco_dir / 'annotations'
    ann_out_dir.mkdir(parents=True, exist_ok=True)
    
    categories = [{"id": 0, "name": "smoke", "supercategory": "none"}]
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"Processing split: {split}")
        split_images_out = images_out_dir / split
        split_images_out.mkdir(parents=True, exist_ok=True)
        
        yolo_img_dir = yolo_dir / 'images' / split
        yolo_lbl_dir = yolo_dir / 'labels' / split
        
        if not yolo_img_dir.exists():
            continue
            
        coco_data = {
            "info": {"description": f"Cognitive Fire Defense - {split}"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories
        }
        
        ann_id = 0
        img_id = 0
        first_img_processed = False
        
        for img_file in yolo_img_dir.glob('*.jpg'):
            # Copy or hardlink
            dest_img = split_images_out / img_file.name
            if not dest_img.exists():
                try:
                    os.link(img_file, dest_img)
                except OSError:
                    shutil.copy2(img_file, dest_img)
                    
            # Get dimensions
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            h, w, _ = img.shape
            
            coco_data['images'].append({
                "id": img_id,
                "file_name": img_file.name,
                "width": w,
                "height": h
            })
            
            lbl_file = yolo_lbl_dir / (img_file.stem + '.txt')
            has_boxes = False
            if lbl_file.exists():
                with open(lbl_file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # YOLO format: cx, cy, bw, bh (normalized)
                        cx, cy, bw, bh = map(float, parts[1:5])
                        
                        # Convert to COCO format: xmin, ymin, width, height (absolute)
                        abs_bw = bw * w
                        abs_bh = bh * h
                        xmin = (cx * w) - (abs_bw / 2)
                        ymin = (cy * h) - (abs_bh / 2)
                        
                        # Fix negative coords
                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        
                        coco_data['annotations'].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": class_id,
                            "bbox": [xmin, ymin, abs_bw, abs_bh],
                            "area": abs_bw * abs_bh,
                            "iscrowd": 0
                        })
                        ann_id += 1
                        has_boxes = True
            
            # Save a debug image for the very first image that has boxes
            if not first_img_processed and has_boxes:
                debug_img = img.copy()
                for ann in [a for a in coco_data['annotations'] if a['image_id'] == img_id]:
                    bx, by, bvw, bvh = [int(v) for v in ann['bbox']]
                    cv2.rectangle(debug_img, (bx, by), (bx+bvw, by+bvh), (0, 0, 255), 2)
                debug_path = coco_dir / f"debug_{split}.png"
                cv2.imwrite(str(debug_path), debug_img)
                print(f"Saved debug overlay to {debug_path}")
                first_img_processed = True
                
            img_id += 1
            
        out_json = ann_out_dir / f"{split}.json"
        with open(out_json, 'w') as f:
            json.dump(coco_data, f)
        print(f"Saved {out_json} with {img_id} images and {ann_id} annotations.")

if __name__ == "__main__":
    base_dir = Path("dataset-b")
    yolo_dir = base_dir / "yolo_format"
    coco_dir = base_dir / "coco_format"
    convert_yolo_to_coco(yolo_dir, coco_dir)
