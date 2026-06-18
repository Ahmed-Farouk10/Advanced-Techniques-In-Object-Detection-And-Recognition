import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

def run_advanced_eda(dataset_path):
    print("Starting Advanced EDA (12GB Deep Dive)...\n")
    locations = ["Evo", "Heinola", "Karkkila", "Ruokolahti"]
    
    # 1. Spatial Heatmap Data
    all_cx = []
    all_cy = []
    
    # 2. Resolution Data
    resolutions = set()
    
    # 3. Illumination Data (Sample of 500 images for speed)
    all_images = []

    print("Gathering data...")
    for loc in locations:
        img_dir = os.path.join(dataset_path, f"{loc}-Images")
        lbl_dir = os.path.join(dataset_path, f"{loc}-Labels")
        
        if not os.path.exists(img_dir): continue
            
        imgs = glob.glob(os.path.join(img_dir, "*.jpg"))
        all_images.extend(imgs)
        
        # Spatial data
        if os.path.exists(lbl_dir):
            for txt in glob.glob(os.path.join(lbl_dir, "*.txt")):
                with open(txt, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            _, cx, cy, _, _ = map(float, parts)
                            all_cx.append(cx)
                            all_cy.append(cy)

    # Resolution Check (Check first 100 images to find unique resolutions)
    print("Checking resolutions...")
    for img_path in all_images[:200]:
        try:
            with Image.open(img_path) as img:
                resolutions.add(img.size)
        except:
            pass

    # Illumination Analysis
    print("Analyzing illumination on 500 random samples...")
    sample_imgs = random.sample(all_images, min(500, len(all_images)))
    brightness_values = []
    
    for img_path in sample_imgs:
        try:
            with Image.open(img_path) as img:
                # Convert to grayscale and get mean pixel value (0-255)
                gray = img.convert('L')
                stat = np.mean(np.array(gray))
                brightness_values.append(stat)
        except Exception as e:
            pass

    print("\n=== ADVANCED EDA RESULTS ===")
    
    # Report Resolutions
    print(f"1. Image Resolutions found: {resolutions}")
    if len(resolutions) > 1:
        print("   -> WARNING: Mixed resolutions detected. Squashing to 640x640 will cause varied aspect ratio distortion.")
    else:
        print("   -> GOOD: Uniform resolution. Standard resizing will not cause varied distortion.")

    # Report Illumination
    mean_bright = np.mean(brightness_values)
    dark_imgs = sum(1 for b in brightness_values if b < 85) # arbitrary threshold for "dark/night"
    bright_imgs = sum(1 for b in brightness_values if b > 170)
    print(f"\n2. Illumination (Brightness 0-255):")
    print(f"   -> Average Brightness: {mean_bright:.1f}")
    print(f"   -> Dark/Night images (<85): {dark_imgs} / {len(sample_imgs)} ({(dark_imgs/len(sample_imgs))*100:.1f}%)")
    print(f"   -> Overexposed/Bright (>170): {bright_imgs} / {len(sample_imgs)} ({(bright_imgs/len(sample_imgs))*100:.1f}%)")
    
    # Save Heatmap Plot
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(brightness_values, bins=30, color='gold', edgecolor='black')
    plt.title("Illumination (Brightness) Distribution")
    plt.xlabel("Mean Pixel Intensity (0=Black, 255=White)")
    plt.ylabel("Image Count (Sample)")
    
    plt.subplot(1, 2, 2)
    # y is inverted in images (0 is top)
    plt.hist2d(all_cx, all_cy, bins=20, cmap='hot')
    plt.gca().invert_yaxis()
    plt.title("Spatial Heatmap of Smoke Plumes")
    plt.xlabel("X Coordinate (Normalized)")
    plt.ylabel("Y Coordinate (Normalized)")
    plt.colorbar(label="Frequency")
    
    out_path = "dataset-b/preprocessing/task2_data_understanding/advanced_eda_plots.png"
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"\n3. Spatial Heatmap:")
    print(f"   -> Mean X center: {np.mean(all_cx):.3f}")
    print(f"   -> Mean Y center: {np.mean(all_cy):.3f} (Lower = closer to sky/horizon)")
    print(f"   -> Plots saved to {out_path}")

if __name__ == "__main__":
    base_path = r"d:\Masters Academy AI\Advanced Techniques in Object Detection and Recognition\Research Paper\dataset-b\raw\Boreal-Forest-Fire-Subset-A"
    run_advanced_eda(base_path)
