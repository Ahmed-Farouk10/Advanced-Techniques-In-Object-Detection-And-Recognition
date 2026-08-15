from ultralytics import YOLO


def train_baseline():

    print("=" * 60)
    print("YOLO11n BASELINE")
    print("=" * 60)

    # Load pretrained YOLO11n
    model = YOLO("yolo11n.pt")

    print("Model: YOLO11n")
    print("Dataset: Boreal Forest Fire — Subset A")
    print("Training split: train")
    print("Validation split: val")
    print("Configuration: Ultralytics default")
    print("Custom hyperparameters: DISABLED")

    results = model.train(
        data="smoke_data.yaml",

        # Keep these consistent with later experiments
        epochs=100,
        imgsz=640,
        batch=16,

        # use this if you haven't NVIDIA CUDA GPU
        device="cpu",

        # Separate output from smoke test/custom experiments
        project="runs",
        name="yolo11n_baseline",
        exist_ok=True
    )

    print("\nBaseline training completed.")
    print("Results saved to:")
    print("runs/yolo11n_baseline")


if __name__ == "__main__":
    train_baseline()