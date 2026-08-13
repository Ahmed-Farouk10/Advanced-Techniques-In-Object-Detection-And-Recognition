from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ============================================================
# Configuration
# ============================================================

ROOT = Path("../../yolo_format")

VAL_IMAGES = ROOT / "images" / "val"
VAL_LABELS = ROOT / "labels" / "val"

MODEL_PATH = Path(
    "runs/faster_rcnn_loss_check/best_model.pth"
)

NUM_CLASSES = 2

# 0 = background
# 1 = smoke

IMAGE_SIZE = 640
BATCH_SIZE = 1

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Detection settings
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5


# ============================================================
# Dataset
# ============================================================

class SmokeDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir, label_dir):

        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        self.images = sorted(
            self.image_dir.glob("*.jpg")
        )

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        image_path = self.images[idx]

        image = (
            torchvision.io.read_image(
                str(image_path)
            )
            .float()
            / 255.0
        )

        # Original image dimensions
        _, original_height, original_width = image.shape

        # Resize image to 640x640
        image = torchvision.transforms.functional.resize(
            image,
            [IMAGE_SIZE, IMAGE_SIZE]
        )

        label_path = (
            self.label_dir
            / f"{image_path.stem}.txt"
        )

        boxes = []

        if label_path.exists():

            with open(label_path, "r") as f:

                for line in f:

                    values = line.strip().split()

                    if len(values) != 5:
                        continue

                    class_id, xc, yc, w, h = map(
                        float,
                        values
                    )

                    # YOLO normalized coordinates
                    x_center = xc * original_width
                    y_center = yc * original_height

                    box_width = w * original_width
                    box_height = h * original_height

                    xmin = (
                        x_center
                        - box_width / 2
                    )

                    ymin = (
                        y_center
                        - box_height / 2
                    )

                    xmax = (
                        x_center
                        + box_width / 2
                    )

                    ymax = (
                        y_center
                        + box_height / 2
                    )

                    # Scale to 640x640
                    scale_x = (
                        IMAGE_SIZE
                        / original_width
                    )

                    scale_y = (
                        IMAGE_SIZE
                        / original_height
                    )

                    xmin *= scale_x
                    xmax *= scale_x
                    ymin *= scale_y
                    ymax *= scale_y

                    # Clamp
                    xmin = max(
                        0,
                        min(IMAGE_SIZE, xmin)
                    )

                    ymin = max(
                        0,
                        min(IMAGE_SIZE, ymin)
                    )

                    xmax = max(
                        0,
                        min(IMAGE_SIZE, xmax)
                    )

                    ymax = max(
                        0,
                        min(IMAGE_SIZE, ymax)
                    )

                    if xmax > xmin and ymax > ymin:

                        boxes.append(
                            [
                                xmin,
                                ymin,
                                xmax,
                                ymax
                            ]
                        )

        boxes = torch.as_tensor(
            boxes,
            dtype=torch.float32
        ).reshape(-1, 4)

        labels = torch.ones(
            len(boxes),
            dtype=torch.int64
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        return image, target


# ============================================================
# Collate Function
# ============================================================

def collate_fn(batch):

    return tuple(zip(*batch))


# ============================================================
# Model
# ============================================================

def create_model():

    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=None
    )

    in_features = (
        model.roi_heads
        .box_predictor
        .cls_score
        .in_features
    )

    model.roi_heads.box_predictor = (
        FastRCNNPredictor(
            in_features,
            NUM_CLASSES
        )
    )

    return model


# ============================================================
# IoU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0])
        * max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0])
        * max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# Evaluate Detection Metrics
# ============================================================

def evaluate_model(
    model,
    data_loader,
    device
):

    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_gt_boxes = 0
    total_predictions = 0

    with torch.no_grad():

        for batch_idx, (images, targets) in enumerate(
            data_loader
        ):

            images = [
                image.to(device)
                for image in images
            ]

            outputs = model(images)

            for output, target in zip(
                outputs,
                targets
            ):

                gt_boxes = (
                    target["boxes"]
                    .cpu()
                    .tolist()
                )

                pred_boxes = (
                    output["boxes"]
                    .cpu()
                    .tolist()
                )

                pred_scores = (
                    output["scores"]
                    .cpu()
                    .tolist()
                )

                # Apply confidence threshold
                filtered_predictions = []

                for box, score in zip(
                    pred_boxes,
                    pred_scores
                ):

                    if score >= CONFIDENCE_THRESHOLD:

                        filtered_predictions.append(
                            (
                                box,
                                score
                            )
                        )

                # Sort predictions by confidence
                filtered_predictions.sort(
                    key=lambda x: x[1],
                    reverse=True
                )

                matched_gt = set()

                tp = 0
                fp = 0

                for pred_box, score in (
                    filtered_predictions
                ):

                    best_iou = 0.0
                    best_gt_idx = None

                    for gt_idx, gt_box in enumerate(
                        gt_boxes
                    ):

                        if gt_idx in matched_gt:
                            continue

                        iou = calculate_iou(
                            pred_box,
                            gt_box
                        )

                        if iou > best_iou:

                            best_iou = iou
                            best_gt_idx = gt_idx

                    if (
                        best_iou >= IOU_THRESHOLD
                        and best_gt_idx is not None
                    ):

                        tp += 1

                        matched_gt.add(
                            best_gt_idx
                        )

                    else:

                        fp += 1

                fn = (
                    len(gt_boxes)
                    - len(matched_gt)
                )

                total_tp += tp
                total_fp += fp
                total_fn += fn

                total_gt_boxes += len(
                    gt_boxes
                )

                total_predictions += len(
                    filtered_predictions
                )

            if (batch_idx + 1) % 100 == 0:

                print(
                    f"Processed "
                    f"{batch_idx + 1}/"
                    f"{len(data_loader)} images"
                )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    return {
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "GT_boxes": total_gt_boxes,
        "Predictions": total_predictions,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Faster R-CNN + MobileNetV3 Evaluation")
    print("=" * 60)

    print("Device:", DEVICE)

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Confidence threshold:",
        CONFIDENCE_THRESHOLD
    )

    print(
        "IoU threshold:",
        IOU_THRESHOLD
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = SmokeDataset(
        VAL_IMAGES,
        VAL_LABELS
    )

    print(
        "Validation images:",
        len(dataset)
    )

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\nLoading model..."
    )

    model = create_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Handle checkpoints saved either
    # as state_dict or as a dictionary
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif "state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            model.load_state_dict(
                checkpoint
            )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.to(DEVICE)

    print(
        "✓ Model loaded successfully"
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print(
        "\nStarting evaluation...\n"
    )

    results = evaluate_model(
        model,
        data_loader,
        DEVICE
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"Ground Truth Boxes : "
        f"{results['GT_boxes']}"
    )

    print(
        f"Predictions        : "
        f"{results['Predictions']}"
    )

    print(
        f"True Positives     : "
        f"{results['TP']}"
    )

    print(
        f"False Positives    : "
        f"{results['FP']}"
    )

    print(
        f"False Negatives    : "
        f"{results['FN']}"
    )

    print(
        f"\nPrecision          : "
        f"{results['Precision']:.4f}"
    )

    print(
        f"Recall             : "
        f"{results['Recall']:.4f}"
    )

    print(
        f"F1 Score           : "
        f"{results['F1']:.4f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()