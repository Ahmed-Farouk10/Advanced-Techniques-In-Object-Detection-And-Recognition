# Task 2: Data Understanding (EDA) - Dataset A

> **Phase:** Data Understanding (EDA)  
> **A+ Philosophy:** Explain WHY. Link to Business. Show Critical Thinking.

## 1. Initial Dataset Exploration
We executed an exploratory data analysis script (`shared/eda_dataset_a.py`) against the downloaded `alik05/forest-fire-dataset`.

**Raw Statistical Output:**
- **Class `fire`**: 760 images (Resolution: 250x250)
- **Class `nofire`**: 760 images (Resolutions: ~250x250)
- **Missing Values / Imbalance:** The dataset is perfectly balanced (50/50). No missing image files were detected.

## 2. The Fatal Flaw: ML Translation Mismatch (Critical Thinking)

While a standard analysis might stop at checking image resolutions and class balance, our **Business Constraint** requires us to deploy an *Object Detection* model (YOLO11n / Faster R-CNN) to a drone. The drone must report the exact coordinates of the fire within its field of view to ground control.

**The Discovery:**
The dataset contains **ZERO bounding box annotations** (`.txt`, `.xml`, or `.json`). The images are merely sorted into `fire` and `nofire` folders.

**Business Impact (WHY this matters):**
- **Image Classification (Current Dataset State):** Tells the drone "There is a fire *somewhere* in this 250x250 image."
- **Object Detection (Our Requirement):** Tells the drone "There is a fire at `x:120, y:80` with a width of `30px`."

If we blindly proceed to train YOLO11n on this dataset, the code will fail because YOLO requires target coordinates, not just folder names. We cannot build our real-world solution with this data as-is.

## 3. Options for Adaptation (Moving to Task 3)

To act as a Business Consultant solving problems rather than just a coder running scripts, we have identified three paths forward:

1. **Manual Annotation (High Cost):** We open all 760 fire images in a tool like Roboflow or CVAT and draw bounding boxes manually. This costs significant time.
2. **Pseudo-Labeling (Technical Risk):** We use a pre-trained zero-shot model (like Grounding DINO) to automatically generate bounding boxes around anything that looks like "fire". We accept some noise in the labels.
3. **Change the Dataset (Strategic Pivot):** We discard Dataset A and find a different drone dataset that already contains bounding boxes.

### Conclusion
Dataset A is currently incompatible with our ML translation pipeline. We must decide which of the 3 adaptations above to execute in **Task 3: Data Cleaning / Transformation** to recover the project timeline.
