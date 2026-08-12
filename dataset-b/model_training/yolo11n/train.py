from ultralytics import YOLO


def train_model():
    # Load YOLO11n (Anchor-Free)
    model = YOLO("yolo11n.pt")

    print("Initiating YOLO11n Custom Augmentation Training...")

    results = model.train(
        data="smoke_data.yaml",
        cfg="custom_hyp.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device="cpu", # Edit by Esraa Nasr
        project="runs",
        name="yolo11n_custom_aug",
        exist_ok=True
    )

    return results


if __name__ == "__main__":
    train_model()