from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision

from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ============================================================
# Configuration
# ============================================================

ROOT = Path("../../yolo_format")

TRAIN_IMAGES = ROOT / "images" / "train"
TRAIN_LABELS = ROOT / "labels" / "train"

IMAGE_SIZE = 640
NUM_CLASSES = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Dataset
# ============================================================

class SmokeDataset(Dataset):

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

        image = Image.open(image_path).convert("RGB")

        original_width, original_height = image.size

        # Resize image to 640x640
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        label_path = (
            self.label_dir /
            f"{image_path.stem}.txt"
        )

        boxes = []
        labels = []

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

                    xmin = x_center - box_width / 2
                    ymin = y_center - box_height / 2
                    xmax = x_center + box_width / 2
                    ymax = y_center + box_height / 2

                    # Scale to 640x640
                    scale_x = IMAGE_SIZE / original_width
                    scale_y = IMAGE_SIZE / original_height

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

                        boxes.append([
                            xmin,
                            ymin,
                            xmax,
                            ymax
                        ])

                        # 0 = background
                        # 1 = smoke
                        labels.append(1)

        boxes = torch.as_tensor(
            boxes,
            dtype=torch.float32
        ).reshape(-1, 4)

        labels = torch.as_tensor(
            labels,
            dtype=torch.int64
        )

        image_tensor = torchvision.transforms.functional.to_tensor(
            image
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        return image_tensor, target


# ============================================================
# Model
# ============================================================

def create_model():

    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights="DEFAULT"
    )

    in_features = (
        model.roi_heads.box_predictor
        .cls_score.in_features
    )

    model.roi_heads.box_predictor = (
        FastRCNNPredictor(
            in_features,
            NUM_CLASSES
        )
    )

    return model


# ============================================================
# Smoke Test
# ============================================================

def main():

    print("=" * 60)
    print("Faster R-CNN Smoke Test")
    print("=" * 60)

    print("Device:", DEVICE)

    dataset = SmokeDataset(
        TRAIN_IMAGES,
        TRAIN_LABELS
    )

    print(
        "Dataset size:",
        len(dataset)
    )

    # --------------------------------------------------------
    # Load one sample
    # --------------------------------------------------------

    image, target = dataset[0]

    print(
        "Image shape:",
        image.shape
    )

    print(
        "Boxes shape:",
        target["boxes"].shape
    )

    print(
        "Labels:",
        target["labels"]
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model()

    model.to(DEVICE)

    model.train()

    image = image.to(DEVICE)

    target = {
        key: value.to(DEVICE)
        for key, value in target.items()
    }

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    loss_dict = model(
        [image],
        [target]
    )

    print("\nLoss components:")

    total_loss = 0.0

    for name, loss in loss_dict.items():

        print(
            f"{name}: {loss.item():.4f}"
        )

        total_loss += loss

    print(
        f"\nTotal loss: {total_loss.item():.4f}"
    )

    # --------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------

    total_loss.backward()

    print("\n✓ Forward pass successful")
    print("✓ Backward pass successful")
    print("✓ Faster R-CNN smoke test PASSED")


if __name__ == "__main__":
    main()