from pathlib import Path

import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from train import (
    SmokeDataset,
    VAL_IMAGES,
    VAL_LABELS,
    collate_fn,
    NUM_CLASSES,
    IMAGE_SIZE,
)


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path(
    "runs/faster_rcnn_loss_check/best_model.pth"
)

BATCH_SIZE = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CONFIDENCE_THRESHOLD = 0.5

IOU_THRESHOLDS = torch.arange(
    0.50,
    0.96,
    0.05
)


# ============================================================
# Model
# ============================================================

def create_model():

    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=None
    )

    in_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )

    return model


# ============================================================
# Load Model
# ============================================================

def load_model():

    print("Creating Faster R-CNN + MobileNetV3...")

    model = create_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Support both:
    # 1. state_dict directly
    # 2. checkpoint dictionary

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

    model.to(DEVICE)

    model.eval()

    print(
        f"Model loaded from:\n{MODEL_PATH}"
    )

    return model


# ============================================================
# IoU Matching
# ============================================================

def match_predictions(
    gt_boxes,
    pred_boxes,
    pred_scores,
    iou_threshold,
):
    """
    Match predictions to ground-truth boxes.

    Each ground-truth box can be matched only once.

    Returns:
        TP
        FP
        FN
    """

    if len(pred_boxes) == 0:

        return 0, 0, len(gt_boxes)

    if len(gt_boxes) == 0:

        return 0, len(pred_boxes), 0

    ious = box_iou(
        pred_boxes,
        gt_boxes
    )

    # Sort predictions by confidence
    order = torch.argsort(
        pred_scores,
        descending=True
    )

    matched_gt = set()

    tp = 0
    fp = 0

    for pred_idx in order:

        pred_idx = pred_idx.item()

        best_iou = 0.0
        best_gt = -1

        for gt_idx in range(
            len(gt_boxes)
        ):

            if gt_idx in matched_gt:
                continue

            iou = ious[
                pred_idx,
                gt_idx
            ].item()

            if iou > best_iou:

                best_iou = iou
                best_gt = gt_idx

        if (
            best_iou >= iou_threshold
            and best_gt >= 0
        ):

            tp += 1

            matched_gt.add(
                best_gt
            )

        else:

            fp += 1

    fn = (
        len(gt_boxes)
        - len(matched_gt)
    )

    return tp, fp, fn


# ============================================================
# AP Calculation
# ============================================================

def calculate_ap(
    all_predictions,
    all_ground_truths,
    iou_threshold,
):
    """
    Calculate Average Precision at a given IoU threshold.

    Uses confidence-ranked predictions and
    precision-recall integration.
    """

    total_gt = sum(
        len(gt)
        for gt in all_ground_truths
    )

    if total_gt == 0:

        return 0.0

    # --------------------------------------------------------
    # Collect all predictions
    # --------------------------------------------------------

    predictions = []

    for image_idx, prediction in enumerate(
        all_predictions
    ):

        boxes = prediction["boxes"]
        scores = prediction["scores"]

        for box, score in zip(
            boxes,
            scores
        ):

            predictions.append(
                {
                    "image_idx": image_idx,
                    "box": box,
                    "score": score.item(),
                }
            )

    # --------------------------------------------------------
    # Sort by confidence
    # --------------------------------------------------------

    predictions.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    matched = [
        set()
        for _ in all_ground_truths
    ]

    tp_list = []
    fp_list = []

    # --------------------------------------------------------
    # Match predictions
    # --------------------------------------------------------

    for prediction in predictions:

        image_idx = prediction[
            "image_idx"
        ]

        pred_box = prediction[
            "box"
        ].unsqueeze(0)

        gt_boxes = all_ground_truths[
            image_idx
        ]

        if len(gt_boxes) == 0:

            tp_list.append(0)
            fp_list.append(1)

            continue

        ious = box_iou(
            pred_box,
            gt_boxes
        )[0]

        best_iou = 0.0
        best_gt = -1

        for gt_idx in range(
            len(gt_boxes)
        ):

            if gt_idx in matched[
                image_idx
            ]:

                continue

            iou = ious[
                gt_idx
            ].item()

            if iou > best_iou:

                best_iou = iou
                best_gt = gt_idx

        if (
            best_iou >= iou_threshold
            and best_gt >= 0
        ):

            tp_list.append(1)
            fp_list.append(0)

            matched[
                image_idx
            ].add(best_gt)

        else:

            tp_list.append(0)
            fp_list.append(1)

    if len(tp_list) == 0:

        return 0.0

    tp_tensor = torch.tensor(
        tp_list,
        dtype=torch.float32
    )

    fp_tensor = torch.tensor(
        fp_list,
        dtype=torch.float32
    )

    cumulative_tp = torch.cumsum(
        tp_tensor,
        dim=0
    )

    cumulative_fp = torch.cumsum(
        fp_tensor,
        dim=0
    )

    recalls = (
        cumulative_tp
        / total_gt
    )

    precisions = (
        cumulative_tp
        / (
            cumulative_tp
            + cumulative_fp
            + 1e-8
        )
    )

    # --------------------------------------------------------
    # Precision envelope
    # --------------------------------------------------------

    mrec = torch.cat(
        [
            torch.tensor([0.0]),
            recalls,
            torch.tensor([1.0]),
        ]
    )

    mpre = torch.cat(
        [
            torch.tensor([0.0]),
            precisions,
            torch.tensor([0.0]),
        ]
    )

    for i in range(
        len(mpre) - 2,
        -1,
        -1
    ):

        mpre[i] = torch.maximum(
            mpre[i],
            mpre[i + 1]
        )

    indices = torch.where(
        mrec[1:] != mrec[:-1]
    )[0]

    ap = torch.sum(
        (
            mrec[indices + 1]
            - mrec[indices]
        )
        * mpre[indices + 1]
    )

    return ap.item()


# ============================================================
# Main Evaluation
# ============================================================

def main():

    print("=" * 60)
    print("Faster R-CNN + MobileNetV3 Evaluation")
    print("=" * 60)

    print(
        "Device:",
        DEVICE
    )

    print(
        "Image size:",
        IMAGE_SIZE
    )

    print(
        "Confidence threshold:",
        CONFIDENCE_THRESHOLD
    )

    print(
        "Model:",
        MODEL_PATH
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

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Collect predictions
    # --------------------------------------------------------

    all_predictions = []
    all_ground_truths = []

    print(
        "\nRunning inference..."
    )

    with torch.no_grad():

        for batch_idx, (
            images,
            targets
        ) in enumerate(loader):

            images = [
                image.to(DEVICE)
                for image in images
            ]

            outputs = model(
                images
            )

            for target, output in zip(
                targets,
                outputs
            ):

                gt_boxes = (
                    target["boxes"]
                    .cpu()
                )

                pred_boxes = (
                    output["boxes"]
                    .cpu()
                )

                pred_scores = (
                    output["scores"]
                    .cpu()
                )

                # Confidence filtering
                keep = (
                    pred_scores
                    >= CONFIDENCE_THRESHOLD
                )

                pred_boxes = (
                    pred_boxes[keep]
                )

                pred_scores = (
                    pred_scores[keep]
                )

                all_ground_truths.append(
                    gt_boxes
                )

                all_predictions.append(
                    {
                        "boxes": pred_boxes,
                        "scores": pred_scores,
                    }
                )

            if (
                (batch_idx + 1) % 50 == 0
                or batch_idx == len(loader) - 1
            ):

                print(
                    f"Processed "
                    f"[{batch_idx + 1}/"
                    f"{len(loader)}]"
                )

    # ========================================================
    # Precision / Recall / F1 @ IoU 0.50
    # ========================================================

    tp, fp, fn = 0, 0, 0

    for predictions, gt_boxes in zip(
        all_predictions,
        all_ground_truths
    ):

        image_tp, image_fp, image_fn = (
            match_predictions(
                gt_boxes,
                predictions["boxes"],
                predictions["scores"],
                iou_threshold=0.50,
            )
        )

        tp += image_tp
        fp += image_fp
        fn += image_fn

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
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

    # ========================================================
    # mAP
    # ========================================================

    print(
        "\nCalculating mAP..."
    )

    ap_values = []

    for iou_threshold in IOU_THRESHOLDS:

        threshold = (
            iou_threshold.item()
        )

        ap = calculate_ap(
            all_predictions,
            all_ground_truths,
            threshold,
        )

        ap_values.append(ap)

        print(
            f"AP@{threshold:.2f}: "
            f"{ap:.4f}"
        )

    map50 = ap_values[0]

    map5095 = sum(
        ap_values
    ) / len(ap_values)

    # ========================================================
    # Final Results
    # ========================================================

    print("\n")
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"Ground Truth Boxes : "
        f"{sum(len(x) for x in all_ground_truths)}"
    )

    print(
        f"Predictions        : "
        f"{sum(len(x['boxes']) for x in all_predictions)}"
    )

    print(
        f"True Positives     : {tp}"
    )

    print(
        f"False Positives    : {fp}"
    )

    print(
        f"False Negatives    : {fn}"
    )

    print()

    print(
        f"Precision          : "
        f"{precision:.4f}"
    )

    print(
        f"Recall             : "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score           : "
        f"{f1:.4f}"
    )

    print(
        f"mAP@50             : "
        f"{map50:.4f}"
    )

    print(
        f"mAP@50:95          : "
        f"{map5095:.4f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()