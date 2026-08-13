import csv
from pathlib import Path

import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    "/home/esraa/Public/AIN7015/advancedTechniques/"
    "Paper_Presentation/project/"
    "Advanced-Techniques-In-Object-Detection-And-Recognition"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "dataset-b/model_training/faster_rcnn/"
    / "runs/faster_rcnn_loss_check/best_model.pth"
)

TEST_IMAGES = (
    PROJECT_ROOT
    / "dataset-a/test_dataset/"
    / "Fire-and-Smoke-Detection-Dataset/"
    "Fire-and-Smoke-Detection-Dataset/dataset/test/images"
)

TEST_LABELS = (
    PROJECT_ROOT
    / "dataset-a/test_dataset/"
    / "Fire-and-Smoke-Detection-Dataset/"
    "Fire-and-Smoke-Detection-Dataset/dataset/test/labels"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset-b/model_training/faster_rcnn/"
    / "runs/unseen_fire_evaluation/qualitative"
)

CONFIDENCE_THRESHOLD = 0.05
IOU_THRESHOLD = 0.50

NUM_CLASSES = 2
# 0 = background
# 1 = smoke


# ============================================================
# MODEL
# ============================================================

def create_model():

    print("Creating Faster R-CNN + MobileNetV3...")

    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights="DEFAULT"
    )

    in_features = (
        model
        .roi_heads
        .box_predictor
        .cls_score
        .in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )

    return model


# ============================================================
# IoU
# ============================================================

def calculate_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)

    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)

    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# YOLO LABEL → XYXY
# ============================================================

def load_fire_boxes(label_path, image_width, image_height):

    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            # Kaggle:
            # 0 = fire
            # 1 = smoke
            if class_id != 0:
                continue

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            x1 = (x_center - width / 2) * image_width
            y1 = (y_center - height / 2) * image_height

            x2 = (x_center + width / 2) * image_width
            y2 = (y_center + height / 2) * image_height

            x1 = max(0, min(x1, image_width))
            y1 = max(0, min(y1, image_height))
            x2 = max(0, min(x2, image_width))
            y2 = max(0, min(y2, image_height))

            boxes.append(
                [x1, y1, x2, y2]
            )

    return boxes


# ============================================================
# FONT
# ============================================================

def get_font(size=20):

    possible_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]

    for font_path in possible_fonts:

        if Path(font_path).exists():

            return ImageFont.truetype(
                font_path,
                size
            )

    return ImageFont.load_default()


# ============================================================
# DRAW BOX
# ============================================================

def draw_box(
    draw,
    box,
    text,
    color,
    width=4
):

    x1, y1, x2, y2 = box

    draw.rectangle(
        [x1, y1, x2, y2],
        outline=color,
        width=width
    )

    font = get_font(20)

    bbox = draw.textbbox(
        (x1, y1),
        text,
        font=font
    )

    text_height = bbox[3] - bbox[1]

    background = [
        x1,
        max(0, y1 - text_height - 6),
        bbox[2] + 6,
        y1
    ]

    draw.rectangle(
        background,
        fill=color
    )

    draw.text(
        (x1 + 3, max(0, y1 - text_height - 4)),
        text,
        fill="white",
        font=font
    )


# ============================================================
# SAVE IMAGE
# ============================================================

def save_visualization(
    image,
    gt_boxes,
    predictions,
    matched_predictions,
    status,
    output_path
):

    image = image.copy()

    draw = ImageDraw.Draw(image)

    # --------------------------------------------------------
    # Ground Truth Fire = GREEN
    # --------------------------------------------------------

    for index, gt_box in enumerate(gt_boxes):

        draw_box(
            draw,
            gt_box,
            f"GT FIRE #{index + 1}",
            "green",
            width=5
        )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    for pred_index, pred in enumerate(predictions):

        box = pred["box"]
        confidence = pred["confidence"]
        iou = pred.get("iou", 0.0)

        if pred_index in matched_predictions:

            color = "blue"

            text = (
                f"UNSEEN FIRE "
                f"{confidence:.3f} "
                f"IoU={iou:.2f}"
            )

        else:

            color = "red"

            text = (
                f"SMOKE PRED "
                f"{confidence:.3f}"
            )

        draw_box(
            draw,
            box,
            text,
            color,
            width=4
        )

    # --------------------------------------------------------
    # Status banner
    # --------------------------------------------------------

    font = get_font(24)

    banner = (
        f"{status} | "
        f"GT={len(gt_boxes)} | "
        f"Pred={len(predictions)}"
    )

    draw.rectangle(
        [0, 0, image.width, 40],
        fill="black"
    )

    draw.text(
        (10, 8),
        banner,
        fill="white",
        font=font
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    image.save(output_path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FASTER R-CNN QUALITATIVE UNSEEN FIRE ANALYSIS")
    print("=" * 70)

    print()
    print("Model:")
    print(MODEL_PATH)

    print()
    print("Test images:")
    print(TEST_IMAGES)

    print()
    print("Test labels:")
    print(TEST_LABELS)

    print()
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"IoU threshold: {IOU_THRESHOLD}")

    # --------------------------------------------------------
    # Path validation
    # --------------------------------------------------------

    print()
    print("=" * 30)
    print("CHECKING PATHS")
    print("=" * 30)

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not TEST_IMAGES.exists():

        raise FileNotFoundError(
            f"Test images directory not found:\n"
            f"{TEST_IMAGES}"
        )

    if not TEST_LABELS.exists():

        raise FileNotFoundError(
            f"Test labels directory not found:\n"
            f"{TEST_LABELS}"
        )

    print("✓ Model exists")
    print("✓ Test images directory exists")
    print("✓ Test labels directory exists")

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    TP_DIR = OUTPUT_DIR / "true_positives"
    FP_DIR = OUTPUT_DIR / "false_positives"
    FN_DIR = OUTPUT_DIR / "false_negatives"

    TP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("=" * 30)
    print("LOADING MODEL")
    print("=" * 30)

    model = create_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    # Support checkpoints containing model_state_dict
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(device)

    model.eval()

    print("✓ Model loaded")

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    image_files = sorted(
        list(TEST_IMAGES.glob("*.jpg"))
        + list(TEST_IMAGES.glob("*.jpeg"))
        + list(TEST_IMAGES.glob("*.png"))
    )

    print()
    print(
        f"Total test images: {len(image_files)}"
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / "qualitative_analysis.csv"
    )

    csv_file = open(
        csv_path,
        "w",
        newline=""
    )

    writer = csv.writer(csv_file)

    writer.writerow([
        "image",
        "status",
        "gt_fire_boxes",
        "predictions",
        "matched_fire_predictions",
        "best_iou",
        "max_confidence",
    ])

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    tp_images = 0
    fp_images = 0
    fn_images = 0

    tp_boxes = 0

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print()
    print("=" * 30)
    print("RUNNING QUALITATIVE ANALYSIS")
    print("=" * 30)

    with torch.no_grad():

        for index, image_path in enumerate(image_files):

            image = Image.open(
                image_path
            ).convert("RGB")

            width, height = image.size

            # ------------------------------------------------
            # Ground Truth
            # ------------------------------------------------

            label_path = (
                TEST_LABELS
                / f"{image_path.stem}.txt"
            )

            gt_boxes = load_fire_boxes(
                label_path,
                width,
                height
            )

            # ------------------------------------------------
            # Inference
            # ------------------------------------------------

            image_tensor = F.to_tensor(
                image
            ).to(device)

            output = model(
                [image_tensor]
            )[0]

            predictions = []

            for box, score, label in zip(
                output["boxes"],
                output["scores"],
                output["labels"]
            ):

                confidence = float(
                    score.cpu()
                )

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                box = box.cpu().tolist()

                predictions.append({
                    "box": box,
                    "confidence": confidence,
                    "label": int(
                        label.cpu()
                    ),
                    "iou": 0.0,
                })

            # ------------------------------------------------
            # Match predictions to FIRE GT
            # ------------------------------------------------

            matched_predictions = set()

            best_iou_overall = 0.0

            for pred_index, pred in enumerate(
                predictions
            ):

                best_iou = 0.0

                for gt_box in gt_boxes:

                    iou = calculate_iou(
                        pred["box"],
                        gt_box
                    )

                    best_iou = max(
                        best_iou,
                        iou
                    )

                pred["iou"] = best_iou

                best_iou_overall = max(
                    best_iou_overall,
                    best_iou
                )

                if (
                    best_iou
                    >= IOU_THRESHOLD
                ):

                    matched_predictions.add(
                        pred_index
                    )

            # ------------------------------------------------
            # Determine image status
            # ------------------------------------------------

            if len(gt_boxes) > 0:

                if len(matched_predictions) > 0:

                    status = "TRUE_POSITIVE"

                    tp_images += 1

                    tp_boxes += len(
                        matched_predictions
                    )

                    output_dir = TP_DIR

                else:

                    status = "FALSE_NEGATIVE"

                    fn_images += 1

                    output_dir = FN_DIR

            else:

                if len(predictions) > 0:

                    status = "FALSE_POSITIVE"

                    fp_images += 1

                    output_dir = FP_DIR

                else:

                    continue

            # ------------------------------------------------
            # Save visualization
            # ------------------------------------------------

            output_path = (
                output_dir
                / image_path.name
            )

            save_visualization(
                image=image,
                gt_boxes=gt_boxes,
                predictions=predictions,
                matched_predictions=matched_predictions,
                status=status,
                output_path=output_path
            )

            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            max_confidence = 0.0

            if predictions:

                max_confidence = max(
                    p["confidence"]
                    for p in predictions
                )

            writer.writerow([
                image_path.name,
                status,
                len(gt_boxes),
                len(predictions),
                len(matched_predictions),
                f"{best_iou_overall:.4f}",
                f"{max_confidence:.4f}",
            ])

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                (index + 1) % 50 == 0
                or index + 1 == len(image_files)
            ):

                print(
                    f"Processed "
                    f"[{index + 1}/{len(image_files)}]"
                )

    csv_file.close()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("QUALITATIVE ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        f"True Positive images  : {tp_images}"
    )

    print(
        f"False Positive images : {fp_images}"
    )

    print(
        f"False Negative images : {fn_images}"
    )

    print()
    print(
        f"Unseen Fire TP boxes  : {tp_boxes}"
    )

    print()
    print("Results saved to:")

    print(OUTPUT_DIR)

    print()
    print("CSV:")
    print(csv_path)

    print()
    print("Folders:")
    print(f"  TP: {TP_DIR}")
    print(f"  FP: {FP_DIR}")
    print(f"  FN: {FN_DIR}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()