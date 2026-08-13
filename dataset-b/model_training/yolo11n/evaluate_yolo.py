from pathlib import Path
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

DATA_YAML = "smoke_data.yaml"

# Validation split
SPLIT = "val"

IMAGE_SIZE = 640
BATCH_SIZE = 16

DEVICE = "cpu"

# Models
MODELS = {
    "YOLO11n Baseline": (
        "runs/detect/runs/yolo11n_baseline/weights/best.pt"
    ),
    "YOLO11n Custom Augmentation": (
        "runs/detect/runs/yolo11n_custom_aug/weights/best.pt"
    ),
}


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model_name, model_path):

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    model_path = Path(model_path)

    if not model_path.exists():

        print(
            f"ERROR: Model not found:\n"
            f"{model_path}"
        )

        return None

    print("Model:", model_path)

    model = YOLO(str(model_path))

    print("\nRunning validation...")

    results = model.val(
        data=DATA_YAML,
        split=SPLIT,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        plots=True,
        verbose=True
    )

    metrics = results.box

    precision = metrics.mp
    recall = metrics.mr
    map50 = metrics.map50
    map5095 = metrics.map

    # F1 from Precision and Recall
    if precision + recall > 0:

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)

    print(
        f"Precision      : {precision:.4f}"
    )

    print(
        f"Recall         : {recall:.4f}"
    )

    print(
        f"F1 Score       : {f1:.4f}"
    )

    print(
        f"mAP@50         : {map50:.4f}"
    )

    print(
        f"mAP@50:95      : {map5095:.4f}"
    )

    print("-" * 60)

    return {
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map5095,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("YOLO11n Validation Evaluation")
    print("=" * 60)

    print("Dataset:", DATA_YAML)
    print("Split:", SPLIT)
    print("Image size:", IMAGE_SIZE)
    print("Device:", DEVICE)

    all_results = []

    for model_name, model_path in MODELS.items():

        result = evaluate_model(
            model_name,
            model_path
        )

        if result is not None:

            all_results.append(result)

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print("\n\n" + "=" * 80)
    print("YOLO11n COMPARISON")
    print("=" * 80)

    print(
        f"{'Model':35}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'mAP50':>12}"
        f"{'mAP50-95':>12}"
    )

    print("-" * 80)

    for result in all_results:

        print(
            f"{result['model'][:35]:35}"
            f"{result['precision']:>12.4f}"
            f"{result['recall']:>12.4f}"
            f"{result['f1']:>12.4f}"
            f"{result['map50']:>12.4f}"
            f"{result['map50_95']:>12.4f}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()