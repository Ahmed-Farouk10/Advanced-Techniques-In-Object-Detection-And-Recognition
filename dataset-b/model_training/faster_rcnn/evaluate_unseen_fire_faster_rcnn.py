"""
Faster R-CNN + MobileNetV3
UNSEEN FIRE GENERALIZATION TEST

Research Question:
Can a smoke-only Faster R-CNN detector detect fire
without being trained on fire images?

Training:
    Smoke only

Testing:
    Fire + Smoke dataset

Important:
    The Faster R-CNN model has only one foreground class:
        smoke

    Therefore, ANY prediction that overlaps a FIRE ground-truth
    box with IoU >= 0.5 is counted as an unseen-fire detection,
    regardless of the predicted class.

Evaluation:
    Confidence thresholds:
        0.50, 0.30, 0.20, 0.10, 0.05

    IoU threshold:
        0.50
"""

from pathlib import Path
import sys
import csv
import time

import torch
from PIL import Image
from torchvision.transforms import functional as F


# ============================================================
# PATHS
# ============================================================

# ------------------------------------------------------------
# Find project root automatically
# ------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

# dataset-b/model_training/faster_rcnn
#        ↑
# project root is 4 levels above this file
PROJECT_ROOT = SCRIPT_DIR.parents[2]


# ------------------------------------------------------------
# Faster R-CNN model
# ------------------------------------------------------------

MODEL_PATH = (
    PROJECT_ROOT
    / "dataset-b"
    / "model_training"
    / "faster_rcnn"
    / "runs"
    / "faster_rcnn_loss_check"
    / "best_model.pth"
)


# ------------------------------------------------------------
# Kaggle test dataset
# ------------------------------------------------------------

TEST_IMAGES = (
    PROJECT_ROOT
    / "dataset-a"
    / "test_dataset"
    / "Fire-and-Smoke-Detection-Dataset"
    / "Fire-and-Smoke-Detection-Dataset"
    / "dataset"
    / "test"
    / "images"
)

TEST_LABELS = (
    PROJECT_ROOT
    / "dataset-a"
    / "test_dataset"
    / "Fire-and-Smoke-Detection-Dataset"
    / "Fire-and-Smoke-Detection-Dataset"
    / "dataset"
    / "test"
    / "labels"
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset-b"
    / "model_training"
    / "faster_rcnn"
    / "runs"
    / "unseen_fire_evaluation"
)

RESULTS_CSV = OUTPUT_DIR / "unseen_fire_faster_rcnn_results.csv"


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INFERENCE_CONFIDENCE = 0.01

CONFIDENCE_THRESHOLDS = [
    0.50,
    0.30,
    0.20,
    0.10,
    0.05,
]

IOU_THRESHOLD = 0.50

IMAGE_SIZE = 640


# ============================================================
# IMPORT MODEL CREATION FROM train.py
# ============================================================

sys.path.insert(0, str(SCRIPT_DIR))

from train import create_model


# ============================================================
# YOLO LABEL → XYXY
# ============================================================

def load_fire_boxes(label_path, image_width, image_height):
    """
    Read YOLO-format labels.

    YOLO format:
        class_id x_center y_center width height

    Fire class:
        0

    Smoke class:
        1

    We only keep FIRE boxes.
    """

    fire_boxes = []

    if not label_path.exists():
        return fire_boxes

    with open(label_path, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(float(parts[0]))

            # ------------------------------------------------
            # Fire only
            # ------------------------------------------------

            if class_id != 0:
                continue

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            x_center *= image_width
            y_center *= image_height
            width *= image_width
            height *= image_height

            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            # Clamp to image boundaries

            x1 = max(0, min(x1, image_width))
            y1 = max(0, min(y1, image_height))
            x2 = max(0, min(x2, image_width))
            y2 = max(0, min(y2, image_height))

            if x2 > x1 and y2 > y1:

                fire_boxes.append(
                    [
                        x1,
                        y1,
                        x2,
                        y2,
                    ]
                )

    return torch.tensor(
        fire_boxes,
        dtype=torch.float32,
    )


# ============================================================
# IoU
# ============================================================

def calculate_iou(box_a, box_b):
    """
    Calculate IoU between two XYXY boxes.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)

    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)

    intersection = inter_width * inter_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# MATCH PREDICTIONS TO FIRE BOXES
# ============================================================

def match_predictions(pred_boxes, pred_scores, fire_boxes, confidence):

    """
    Match predictions against FIRE ground-truth boxes.

    A prediction is considered an unseen-fire detection when:

        prediction score >= confidence

    AND

        IoU(prediction, fire_box) >= 0.5

    Greedy one-to-one matching is used.
    """

    selected_indices = [
        i
        for i, score in enumerate(pred_scores)
        if float(score) >= confidence
    ]

    predictions = [
        pred_boxes[i]
        for i in selected_indices
    ]

    prediction_scores = [
        float(pred_scores[i])
        for i in selected_indices
    ]

    if len(fire_boxes) == 0:

        return (
            0,                          # TP
            len(predictions),           # FP
            0,                          # FN
        )

    matched_gt = set()

    true_positives = 0

    # --------------------------------------------------------
    # Sort predictions by confidence
    # --------------------------------------------------------

    order = sorted(
        range(len(predictions)),
        key=lambda i: prediction_scores[i],
        reverse=True,
    )

    for pred_idx in order:

        prediction = predictions[pred_idx]

        best_iou = 0.0
        best_gt = None

        for gt_idx in range(len(fire_boxes)):

            if gt_idx in matched_gt:
                continue

            gt_box = fire_boxes[gt_idx]

            iou = calculate_iou(
                prediction,
                gt_box,
            )

            if iou > best_iou:

                best_iou = iou
                best_gt = gt_idx

        if (
            best_gt is not None
            and best_iou >= IOU_THRESHOLD
        ):

            true_positives += 1
            matched_gt.add(best_gt)

    false_positives = len(predictions) - true_positives

    false_negatives = len(fire_boxes) - true_positives

    return (
        true_positives,
        false_positives,
        false_negatives,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FASTER R-CNN UNSEEN FIRE GENERALIZATION TEST")
    print("=" * 70)

    print()
    print("Research Question:")
    print(
        "Can a smoke-only Faster R-CNN detector detect fire "
        "without being trained on fire images?"
    )

    print()
    print("Model:")
    print(MODEL_PATH)

    print()
    print("Test Dataset:")
    print(TEST_IMAGES)

    print()
    print("Inference confidence:", INFERENCE_CONFIDENCE)

    print(
        "Evaluation thresholds:",
        CONFIDENCE_THRESHOLDS,
    )

    print("IoU threshold:", IOU_THRESHOLD)

    # ========================================================
    # CHECK PATHS
    # ========================================================

    print()
    print("=" * 30)
    print("CHECKING PATHS")
    print("=" * 30)

    if not PROJECT_ROOT.exists():

        print("ERROR: Project root not found:")
        print(PROJECT_ROOT)
        return

    print("✓ Project root exists")

    if not MODEL_PATH.exists():

        print()
        print("ERROR: Model not found:")
        print(MODEL_PATH)
        print()
        print("Try:")
        print(
            "find dataset-b -type f -name "
            "'best_model.pth' -print"
        )

        return

    print("✓ Model exists")

    if not TEST_IMAGES.exists():

        print()
        print("ERROR: Test images directory not found:")
        print(TEST_IMAGES)

        return

    print("✓ Test images directory exists")

    if not TEST_LABELS.exists():

        print()
        print("ERROR: Test labels directory not found:")
        print(TEST_LABELS)

        return

    print("✓ Test labels directory exists")

    # ========================================================
    # GET TEST IMAGES
    # ========================================================

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".JPG",
        ".JPEG",
        ".PNG",
    }

    image_paths = sorted(
        [
            p
            for p in TEST_IMAGES.iterdir()
            if p.suffix in image_extensions
        ]
    )

    print()
    print("Total test images:", len(image_paths))

    if len(image_paths) == 0:

        print("ERROR: No test images found.")
        return

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print("=" * 30)
    print("LOADING FASTER R-CNN")
    print("=" * 30)

    print("Device:", DEVICE)

    model = create_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Handle checkpoint formats
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(DEVICE)

    model.eval()

    print("✓ Model loaded")

    # ========================================================
    # STORAGE
    # ========================================================

    all_results = {
        threshold: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "fire_images": 0,
            "fire_images_detected": 0,
            "ground_truth_boxes": 0,
        }
        for threshold in CONFIDENCE_THRESHOLDS
    }

    # ========================================================
    # INFERENCE
    # ========================================================

    print()
    print("=" * 30)
    print("RUNNING INFERENCE")
    print("=" * 30)

    start_time = time.time()

    for image_index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        image_tensor = F.to_tensor(image)

        image_tensor = image_tensor.to(DEVICE)

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        label_path = (
            TEST_LABELS
            / f"{image_path.stem}.txt"
        )

        fire_boxes = load_fire_boxes(
            label_path,
            width,
            height,
        )

        has_fire = len(fire_boxes) > 0

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        with torch.no_grad():

            output = model(
                [image_tensor]
            )[0]

        pred_boxes = (
            output["boxes"]
            .detach()
            .cpu()
        )

        pred_scores = (
            output["scores"]
            .detach()
            .cpu()
        )

        # ----------------------------------------------------
        # Keep predictions above very low inference threshold
        # ----------------------------------------------------

        keep = pred_scores >= INFERENCE_CONFIDENCE

        pred_boxes = pred_boxes[keep]

        pred_scores = pred_scores[keep]

        # ----------------------------------------------------
        # Evaluate every confidence threshold
        # ----------------------------------------------------

        for threshold in CONFIDENCE_THRESHOLDS:

            tp, fp, fn = match_predictions(
                pred_boxes,
                pred_scores,
                fire_boxes,
                threshold,
            )

            result = all_results[threshold]

            result["tp"] += tp
            result["fp"] += fp
            result["fn"] += fn

            result["ground_truth_boxes"] += len(
                fire_boxes
            )

            if has_fire:

                result["fire_images"] += 1

                if tp > 0:

                    result["fire_images_detected"] += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            image_index % 50 == 0
            or image_index == len(image_paths)
        ):

            print(
                f"Processed "
                f"[{image_index}/{len(image_paths)}]"
            )

    elapsed = time.time() - start_time

    # ========================================================
    # FIRE SUMMARY
    # ========================================================

    total_fire_images = 0
    total_fire_boxes = 0

    for image_path in image_paths:

        label_path = (
            TEST_LABELS
            / f"{image_path.stem}.txt"
        )

        image = Image.open(
            image_path
        )

        width, height = image.size

        fire_boxes = load_fire_boxes(
            label_path,
            width,
            height,
        )

        if len(fire_boxes) > 0:

            total_fire_images += 1
            total_fire_boxes += len(fire_boxes)

    print()
    print("=" * 70)
    print("UNSEEN FIRE EVALUATION RESULTS")
    print("=" * 70)

    print()
    print(
        f"Fire Images             : "
        f"{total_fire_images}"
    )

    print(
        f"Ground Truth Fire Boxes : "
        f"{total_fire_boxes}"
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # CSV
    # ========================================================

    with open(
        RESULTS_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "confidence",
                "predictions",
                "true_positives",
                "false_positives",
                "false_negatives",
                "precision",
                "recall",
                "f1",
                "fire_images",
                "fire_images_detected",
                "image_detection_rate",
                "ground_truth_fire_boxes",
                "fire_boxes_detected",
                "box_detection_rate",
            ]
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        for threshold in CONFIDENCE_THRESHOLDS:

            r = all_results[threshold]

            tp = r["tp"]
            fp = r["fp"]
            fn = r["fn"]

            predictions = tp + fp

            precision = (
                tp / predictions
                if predictions > 0
                else 0.0
            )

            recall = (
                tp / (tp + fn)
                if (tp + fn) > 0
                else 0.0
            )

            f1 = (
                2 * precision * recall
                / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            fire_image_detection_rate = (
                r["fire_images_detected"]
                / r["fire_images"]
                if r["fire_images"] > 0
                else 0.0
            )

            fire_box_detection_rate = (
                tp / r["ground_truth_boxes"]
                if r["ground_truth_boxes"] > 0
                else 0.0
            )

            print()
            print(
                f"Confidence Threshold: "
                f"{threshold:.2f}"
            )

            print(
                f"Predictions          : "
                f"{predictions}"
            )

            print(
                f"True Positives       : "
                f"{tp}"
            )

            print(
                f"False Positives      : "
                f"{fp}"
            )

            print(
                f"False Negatives      : "
                f"{fn}"
            )

            print(
                f"Precision            : "
                f"{precision:.4f}"
            )

            print(
                f"Recall               : "
                f"{recall:.4f}"
            )

            print(
                f"F1 Score             : "
                f"{f1:.4f}"
            )

            print(
                f"Fire Images Detected : "
                f"{r['fire_images_detected']}/"
                f"{r['fire_images']}"
            )

            print(
                f"Image Detection Rate: "
                f"{fire_image_detection_rate:.4f}"
            )

            print(
                f"Fire Boxes Detected  : "
                f"{tp}/"
                f"{r['ground_truth_boxes']}"
            )

            print(
                f"Box Detection Rate   : "
                f"{fire_box_detection_rate:.4f}"
            )

            writer.writerow(
                [
                    threshold,
                    predictions,
                    tp,
                    fp,
                    fn,
                    f"{precision:.6f}",
                    f"{recall:.6f}",
                    f"{f1:.6f}",
                    r["fire_images"],
                    r["fire_images_detected"],
                    f"{fire_image_detection_rate:.6f}",
                    r["ground_truth_boxes"],
                    tp,
                    f"{fire_box_detection_rate:.6f}",
                ]
            )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Inference time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print()
    print("Results saved to:")
    print(RESULTS_CSV)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The Faster R-CNN model was trained only on SMOKE."
    )

    print(
        "A prediction overlapping a FIRE ground-truth "
        "box with IoU >= 0.5 is treated as an "
        "UNSEEN FIRE DETECTION, regardless of the "
        "predicted class."
    )

    print(
        "This directly tests whether visual features "
        "learned from smoke generalize to unseen fire."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()