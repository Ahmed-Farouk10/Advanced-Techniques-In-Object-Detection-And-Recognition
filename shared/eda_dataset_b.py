import os
import glob
import json
import matplotlib.pyplot as plt

def run_eda(dataset_path):
    print("Starting EDA on Dataset B (Boreal Forest Watchtower)...\n")
    
    locations = ["Evo", "Heinola", "Karkkila", "Ruokolahti"]
    
    total_images = 0
    total_boxes = 0
    empty_images = 0
    
    bbox_areas = []
    bbox_aspect_ratios = []
    images_per_location = {}
    boxes_per_location = {}
    
    for loc in locations:
        images_dir = os.path.join(dataset_path, f"{loc}-Images")
        labels_dir = os.path.join(dataset_path, f"{loc}-Labels")
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Skipping {loc} - directories not found.")
            continue
            
        images = glob.glob(os.path.join(images_dir, "*.jpg"))
        labels = glob.glob(os.path.join(labels_dir, "*.txt"))
        
        images_per_location[loc] = len(images)
        total_images += len(images)
        
        loc_boxes = 0
        
        for label_file in labels:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    loc_boxes += len(lines)
                    total_boxes += len(lines)
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            c, cx, cy, w, h = parts
                            w, h = float(w), float(h)
                            area = w * h
                            if h > 0:
                                bbox_aspect_ratios.append(w / h)
                            bbox_areas.append(area)
            except Exception as e:
                print(f"Error reading {label_file}: {e}")
                
        boxes_per_location[loc] = loc_boxes
    
    # Process Empty Images
    empty_dir = os.path.join(dataset_path, "Empty-Images")
    if os.path.exists(empty_dir):
        empty_imgs = glob.glob(os.path.join(empty_dir, "*.jpg"))
        empty_images = len(empty_imgs)
        total_images += empty_images
        
    print("=== DATASET ANATOMY ===")
    print(f"Total Images: {total_images}")
    print(f"Total Bounding Boxes: {total_boxes}")
    print(f"Total Empty (No-Smoke) Images: {empty_images}")
    print(f"Ratio of Empty Images: {empty_images / total_images * 100:.2f}%\n")
    
    print("=== PER LOCATION DISTRIBUTION ===")
    for loc in locations:
        imgs = images_per_location.get(loc, 0)
        boxes = boxes_per_location.get(loc, 0)
        print(f"{loc}: {imgs} images, {boxes} bounding boxes ({boxes/imgs:.2f} boxes/image)")
        
    print("\n=== BOUNDING BOX ANALYSIS (Smoke Plume Size) ===")
    if bbox_areas:
        avg_area = sum(bbox_areas) / len(bbox_areas)
        small_boxes = sum(1 for a in bbox_areas if a < 0.01) # Area < 1% of image
        medium_boxes = sum(1 for a in bbox_areas if 0.01 <= a < 0.1)
        large_boxes = sum(1 for a in bbox_areas if a >= 0.1)
        
        print(f"Average BBox Area (normalized): {avg_area:.4f} ({(avg_area*100):.2f}% of image)")
        print(f"Small Plumes (< 1% of image): {small_boxes} ({(small_boxes/total_boxes)*100:.1f}%)")
        print(f"Medium Plumes (1-10% of image): {medium_boxes} ({(medium_boxes/total_boxes)*100:.1f}%)")
        print(f"Large Plumes (> 10% of image): {large_boxes} ({(large_boxes/total_boxes)*100:.1f}%)")
        
        # Calculate aspect ratio
        avg_ar = sum(bbox_aspect_ratios) / len(bbox_aspect_ratios)
        print(f"Average Aspect Ratio (W/H): {avg_ar:.2f}")

if __name__ == "__main__":
    # Point to Dataset B raw folder
    base_path = r"d:\Masters Academy AI\Advanced Techniques in Object Detection and Recognition\Research Paper\dataset-b\raw\Boreal-Forest-Fire-Subset-A"
    run_eda(base_path)
