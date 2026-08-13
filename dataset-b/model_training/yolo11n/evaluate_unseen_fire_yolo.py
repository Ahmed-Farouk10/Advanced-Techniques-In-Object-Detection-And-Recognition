from pathlib import Path
import csv
import sys

import torch
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

# Script location:
# dataset-b/model_training/yolo11n/evaluate_unseen_fire_yolo.py
#
# parents[0] = yolo11n
# parents[1] = model_training
# parents[2] = dataset-b
# parents[3] = project root

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_A = PROJECT_ROOT / "dataset-a"
DATASET_B = PROJECT_ROOT / "dataset-b"


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = (
    DATASET_B
    / "model_training"
    / "yolo11n"
    / "runs"
    / "detect"
    / "runs"
    / "yolo11n_custom_aug"
    / "weights"
    / "best.pt"
)


# ============================================================
# TEST DATASET
# ============================================================

TEST_ROOT = (
    DATASET_A
    / "test_dataset"
    / "Fire-and-Smoke-Detection-Dataset"
    / "Fire-and-Smoke-Detection-Dataset"
    / "dataset"
    / "test"
)

TEST_IMAGES = TEST_ROOT / "images"
TEST_LABELS = TEST_ROOT / "labels"


# ============================================================
# EVALUATION SETTINGS
# ============================================================

INFERENCE_CONFIDENCE = 0.01

EVAL_THRESHOLDS = [
    0.50,
    0.30,
    0.20,
    0.10,
    0.05,
]

IOU_THRESHOLD = 0.50


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "runs"
    / "unseen_fire_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FUNCTIONS
# ============================================================

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes.

    Boxes format:
        [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)

    intersection = (
        intersection_width *
        intersection_height
    )

    area1 = (
        max(0.0, box1[2] - box1[0]) *
        max(0.0, box1[3] - box1[1])
    )

    area2 = (
        max(0.0, box2[2] - box2[0]) *
        max(0.0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def load_fire_boxes(label_path, image_width, image_height):
    """
    Load only FIRE annotations.

    Dataset classes:
        0 = fire
        1 = smoke

    YOLO format:
        class_id cx cy width height
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

            class_id = int(parts[0])

            # IMPORTANT:
            # 0 = fire
            # 1 = smoke
            if class_id != 0:
                continue

            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            x1 = (cx - w / 2) * image_width
            y1 = (cy - h / 2) * image_height

            x2 = (cx + w / 2) * image_width
            y2 = (cy + h / 2) * image_height

            fire_boxes.append(
                [x1, y1, x2, y2]
            )

    return fire_boxes


def match_predictions(
    predictions,
    ground_truths,
    iou_threshold
):
    """
    Match predictions to fire ground-truth boxes.

    Because the trained model only knows SMOKE,
    we do NOT require predicted class == fire.

    A prediction is considered a fire detection
    if it overlaps a fire ground-truth box.
    """

    if len(ground_truths) == 0:

        return 0, len(predictions), 0

    matched_gt = set()

    true_positives = 0
    false_positives = 0

    # Highest confidence first
    predictions = sorted(
        predictions,
        key=lambda x: x["confidence"],
        reverse=True
    )

    for prediction in predictions:

        best_iou = 0.0
        best_gt_index = None

        for gt_index, gt_box in enumerate(ground_truths):

            if gt_index in matched_gt:
                continue

            iou = calculate_iou(
                prediction["box"],
                gt_box
            )

            if iou > best_iou:

                best_iou = iou
                best_gt_index = gt_index

        if (
            best_gt_index is not None
            and best_iou >= iou_threshold
        ):

            true_positives += 1
            matched_gt.add(best_gt_index)

        else:

            false_positives += 1

    false_negatives = (
        len(ground_truths)
        - len(matched_gt)
    )

    return (
        true_positives,
        false_positives,
        false_negatives
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("YOLO11n UNSEEN FIRE GENERALIZATION TEST")
    print("=" * 70)

    print()
    print("Research Question:")
    print(
        "Can a smoke-only YOLO11n detector detect fire "
        "without being trained on fire images?"
    )

    print()
    print("Project Root:")
    print(PROJECT_ROOT)

    print()
    print("Model:")
    print(MODEL_PATH)

    print()
    print("Test Dataset:")
    print(TEST_ROOT)

    print()
    print("Test Images:")
    print(TEST_IMAGES)

    print()
    print("Test Labels:")
    print(TEST_LABELS)

    print()
    print("Inference confidence:", INFERENCE_CONFIDENCE)
    print("Evaluation thresholds:", EVAL_THRESHOLDS)
    print("IoU threshold:", IOU_THRESHOLD)

    # ========================================================
    # CHECK PATHS
    # ========================================================

    print()
    print("=" * 70)
    print("CHECKING PATHS")
    print("=" * 70)

    if not PROJECT_ROOT.exists():

        print("ERROR: Project root not found:")
        print(PROJECT_ROOT)

        sys.exit(1)

    print("✓ Project root exists")

    if not MODEL_PATH.exists():

        print()
        print("ERROR: Model not found:")
        print(MODEL_PATH)

        print()
        print("Searching for best.pt...")

        for path in PROJECT_ROOT.rglob("best.pt"):
            print(path)

        sys.exit(1)

    print("✓ Model exists")

    if not TEST_IMAGES.exists():

        print()
        print("ERROR: Test images directory not found:")
        print(TEST_IMAGES)

        print()
        print("Searching for test/images directories...")

        for path in DATASET_A.rglob("images"):

            if path.is_dir():
                print(path)

        sys.exit(1)

    print("✓ Test images directory exists")

    if not TEST_LABELS.exists():

        print()
        print("ERROR: Test labels directory not found:")
        print(TEST_LABELS)

        print()
        print("Searching for labels directories...")

        for path in DATASET_A.rglob("labels"):

            if path.is_dir():
                print(path)

        sys.exit(1)

    print("✓ Test labels directory exists")

    # ========================================================
    # FIND IMAGES
    # ========================================================

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    image_paths = sorted(
        [
            p
            for p in TEST_IMAGES.iterdir()
            if p.is_file()
            and p.suffix.lower() in image_extensions
        ]
    )

    if len(image_paths) == 0:

        print()
        print("ERROR: No test images found.")

        sys.exit(1)

    print()
    print("Total test images:", len(image_paths))

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING YOLO11n")
    print("=" * 70)

    model = YOLO(
        str(MODEL_PATH)
    )

    print("✓ Model loaded")

    # ========================================================
    # RUN INFERENCE
    # ========================================================

    print()
    print("=" * 70)
    print("RUNNING INFERENCE")
    print("=" * 70)

    results_data = []

    total_fire_images = 0
    total_fire_boxes = 0

    # --------------------------------------------------------
    # First pass:
    # Run inference once with very low confidence.
    # --------------------------------------------------------

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        # Read image dimensions through PIL
        from PIL import Image

        with Image.open(image_path) as img:

            image_width, image_height = img.size

        label_path = (
            TEST_LABELS
            / f"{image_path.stem}.txt"
        )

        fire_boxes = load_fire_boxes(
            label_path,
            image_width,
            image_height
        )

        total_fire_boxes += len(fire_boxes)

        if len(fire_boxes) > 0:
            total_fire_images += 1

        # YOLO inference
        predictions = []

        result = model.predict(
            source=str(image_path),
            conf=INFERENCE_CONFIDENCE,
            verbose=False,
            device="cpu"
        )[0]

        if result.boxes is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, confidence in zip(
                boxes,
                confidences
            ):

                predictions.append(
                    {
                        "box": box.tolist(),
                        "confidence": float(confidence)
                    }
                )

        results_data.append(
            {
                "image_path": image_path,
                "fire_boxes": fire_boxes,
                "predictions": predictions,
            }
        )

        if (
            index % 50 == 0
            or index == len(image_paths)
        ):

            print(
                f"Processed "
                f"[{index}/{len(image_paths)}]"
            )

    # ========================================================
    # EVALUATION AT MULTIPLE CONFIDENCE THRESHOLDS
    # ========================================================

    print()
    print("=" * 70)
    print("UNSEEN FIRE EVALUATION")
    print("=" * 70)

    print()
    print(
        f"Fire images             : "
        f"{total_fire_images}"
    )

    print(
        f"Ground Truth Fire Boxes : "
        f"{total_fire_boxes}"
    )

    print()

    all_results = []

    for confidence_threshold in EVAL_THRESHOLDS:

        total_predictions = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0

        fire_images_detected = 0

        for item in results_data:

            fire_boxes = item["fire_boxes"]

            predictions = [
                prediction
                for prediction in item["predictions"]
                if prediction["confidence"]
                >= confidence_threshold
            ]

            total_predictions += len(predictions)

            tp, fp, fn = match_predictions(
                predictions,
                fire_boxes,
                IOU_THRESHOLD
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn

            # Image-level detection:
            # At least one prediction overlaps
            # at least one fire GT box.
            if (
                len(fire_boxes) > 0
                and tp > 0
            ):

                fire_images_detected += 1

        if total_tp + total_fp > 0:

            precision = (
                total_tp
                / (total_tp + total_fp)
            )

        else:

            precision = 0.0

        if total_tp + total_fn > 0:

            recall = (
                total_tp
                / (total_tp + total_fn)
            )

        else:

            recall = 0.0

        if precision + recall > 0:

            f1 = (
                2
                * precision
                * recall
                / (precision + recall)
            )

        else:

            f1 = 0.0

        if total_fire_images > 0:

            image_detection_rate = (
                fire_images_detected
                / total_fire_images
            )

        else:

            image_detection_rate = 0.0

        if total_fire_boxes > 0:

            box_detection_rate = (
                total_tp
                / total_fire_boxes
            )

        else:

            box_detection_rate = 0.0

        result_row = {
            "confidence": confidence_threshold,
            "predictions": total_predictions,
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fire_images_detected": fire_images_detected,
            "fire_images_total": total_fire_images,
            "image_detection_rate": image_detection_rate,
            "fire_boxes_detected": total_tp,
            "fire_boxes_total": total_fire_boxes,
            "box_detection_rate": box_detection_rate,
        }

        all_results.append(result_row)

        print("-" * 70)
        print(
            f"Confidence Threshold: "
            f"{confidence_threshold:.2f}"
        )

        print(
            f"Predictions          : "
            f"{total_predictions}"
        )

        print(
            f"True Positives       : "
            f"{total_tp}"
        )

        print(
            f"False Positives      : "
            f"{total_fp}"
        )

        print(
            f"False Negatives      : "
            f"{total_fn}"
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
            f"{fire_images_detected}/"
            f"{total_fire_images}"
        )

        print(
            f"Image Detection Rate: "
            f"{image_detection_rate:.4f}"
        )

        print(
            f"Fire Boxes Detected  : "
            f"{total_tp}/"
            f"{total_fire_boxes}"
        )

        print(
            f"Box Detection Rate   : "
            f"{box_detection_rate:.4f}"
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        / "unseen_fire_results.csv"
    )

    fieldnames = list(
        all_results[0].keys()
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"Fire images             : "
        f"{total_fire_images}"
    )

    print(
        f"Ground Truth Fire Boxes : "
        f"{total_fire_boxes}"
    )

    print()

    for row in all_results:

        print(
            f"Conf={row['confidence']:.2f} | "
            f"Precision={row['precision']:.4f} | "
            f"Recall={row['recall']:.4f} | "
            f"F1={row['f1']:.4f} | "
            f"Image Detection="
            f"{row['image_detection_rate']:.4f} | "
            f"Box Detection="
            f"{row['box_detection_rate']:.4f}"
        )

    print()
    print("Results saved to:")
    print(csv_path)

    print()
    print("=" * 70)
    print("IMPORTANT INTERPRETATION")
    print("=" * 70)

    print(
        "The YOLO11n model was trained only on SMOKE."
    )

    print(
        "Therefore, a prediction overlapping a FIRE "
        "ground-truth box is treated as an "
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