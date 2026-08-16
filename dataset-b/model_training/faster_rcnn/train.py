from pathlib import Path
import csv

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torchvision
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
)


# ============================================================
# Configuration
# ============================================================

ROOT = Path("../../yolo_format")

TRAIN_IMAGES = ROOT / "images" / "train"
TRAIN_LABELS = ROOT / "labels" / "train"

VAL_IMAGES = ROOT / "images" / "val"
VAL_LABELS = ROOT / "labels" / "val"

NUM_CLASSES = 2
# 0 = background
# 1 = smoke

IMAGE_SIZE = 640
BATCH_SIZE = 2

# Diagnostic run
NUM_EPOCHS = 5

RESUME_CHECKPOINT = None

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR = Path("runs/faster_rcnn_loss_check")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
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

        image = Image.open(
            image_path
        ).convert("RGB")

        # Original dimensions
        original_width, original_height = image.size

        # Resize image
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

                    xmin = (
                        x_center -
                        box_width / 2
                    )

                    ymin = (
                        y_center -
                        box_height / 2
                    )

                    xmax = (
                        x_center +
                        box_width / 2
                    )

                    ymax = (
                        y_center +
                        box_height / 2
                    )

                    # Scale to 640x640
                    scale_x = (
                        IMAGE_SIZE /
                        original_width
                    )

                    scale_y = (
                        IMAGE_SIZE /
                        original_height
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

                    # Ignore invalid boxes
                    if (
                        xmax > xmin
                        and ymax > ymin
                    ):

                        boxes.append([
                            xmin,
                            ymin,
                            xmax,
                            ymax
                        ])

                        # Faster R-CNN:
                        # 0 = background
                        # 1 = smoke
                        labels.append(1)

        # IMPORTANT:
        # Empty images must have shape [0, 4]
        boxes = torch.as_tensor(
            boxes,
            dtype=torch.float32
        ).reshape(-1, 4)

        labels = torch.as_tensor(
            labels,
            dtype=torch.int64
        )

        image_tensor = (
            torchvision.transforms.functional
            .to_tensor(image)
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(
                [idx],
                dtype=torch.int64
            ),
        }

        return image_tensor, target


# ============================================================
# Collate Function
# ============================================================

def collate_fn(batch):

    return tuple(zip(*batch))


# ============================================================
# Model
# ============================================================

def create_model():

    print(
        "Creating Faster R-CNN + MobileNetV3..."
    )

    model = (
        fasterrcnn_mobilenet_v3_large_fpn(
            weights="DEFAULT"
        )
    )

    in_features = (
        model
        .roi_heads
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
# Training One Epoch
# ============================================================

def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch
):

    model.train()

    total_batches = len(data_loader)

    loss_sums = {
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
        "total_loss": 0.0,
    }

    for batch_idx, (images, targets) in enumerate(
        data_loader
    ):
    # ----------------------------------------------------
    # Check model parameters for NaN / Inf
    # ----------------------------------------------------

        for name, parameter in model.named_parameters():

            if not torch.isfinite(parameter).all():

                print(
                    "\n"
                    "======================================================================"
                )

                print(
                    "NON-FINITE MODEL PARAMETER DETECTED"
                )

                print(
                    f"Epoch: {epoch}"
                )

                print(
                    f"Batch: {batch_idx + 1}/{total_batches}"
                )

                print(
                    f"Parameter: {name}"
                )

                print(
                    "======================================================================"
                )

                return None
        images = [
            image.to(device)
            for image in images
        ]

        targets = [
            {
                key: value.to(device)
                for key, value in target.items()
            }
            for target in targets
        ]

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------
        # Debug target information
        for i, target in enumerate(targets):

            boxes = target["boxes"]

            if boxes.numel() > 0:

                if not torch.isfinite(boxes).all():
                    print(
                        "\nERROR: Non-finite target boxes"
                    )
                    print("Batch:", batch_idx + 1)
                    print("Image index:", i)
                    print("Boxes:", boxes)
                    return None

                if (boxes[:, 2] <= boxes[:, 0]).any():
                    print(
                        "\nERROR: Invalid X coordinates"
                    )
                    print("Batch:", batch_idx + 1)
                    print("Boxes:", boxes)
                    return None

                if (boxes[:, 3] <= boxes[:, 1]).any():
                    print(
                        "\nERROR: Invalid Y coordinates"
                    )
                    print("Batch:", batch_idx + 1)
                    print("Boxes:", boxes)
                    return None

                if (boxes < 0).any():
                    print(
                        "\nERROR: Negative box coordinate"
                    )
                    print("Batch:", batch_idx + 1)
                    print("Boxes:", boxes)
                    return None

                if (boxes > IMAGE_SIZE).any():
                    print(
                        "\nERROR: Box exceeds image size"
                    )
                    print("Batch:", batch_idx + 1)
                    print("Boxes:", boxes)
                    return None  
        loss_dict = model(
            images,
            targets
        )

        losses = sum(
            loss
            for loss in loss_dict.values()
        )

        # ----------------------------------------------------
        # Check for NaN / Inf
        # ----------------------------------------------------

        if not torch.isfinite(losses).item():

            print(
                "\n"
                "================================================"
            )

            print(
                "ERROR: Non-finite loss detected!"
            )

            print(
                f"Epoch: {epoch}"
            )

            print(
                f"Batch: {batch_idx + 1}"
                f"/{total_batches}"
            )

            print(
                "\nLoss components:"
            )

            for name, value in loss_dict.items():

                print(
                    f"{name}: "
                    f"{value.item()}"
                )

            print(
                "================================================"
            )

            return None

        # ----------------------------------------------------
        # Accumulate losses
        # ----------------------------------------------------

        for name in [
            "loss_classifier",
            "loss_box_reg",
            "loss_objectness",
            "loss_rpn_box_reg",
        ]:

            loss_sums[name] += (
                loss_dict[name].item()
            )

        loss_sums["total_loss"] += (
            losses.item()
        )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        losses.backward()

        # Check gradients before optimizer step
        for name, parameter in model.named_parameters():

            if parameter.grad is not None:

                if not torch.isfinite(
                    parameter.grad
                ).all():

                    print(
                        "\nNON-FINITE GRADIENT DETECTED"
                    )

                    print(
                        f"Parameter: {name}"
                    )

                    return None
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )
        optimizer.step()

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (batch_idx + 1) % 50 == 0
            or
            (batch_idx + 1) == total_batches
        ):

            print(
                f"Batch "
                f"[{batch_idx + 1}/{total_batches}] "
                f"Loss: {losses.item():.4f}"
            )

    # --------------------------------------------------------
    # Average losses
    # --------------------------------------------------------

    num_batches = float(total_batches)

    averages = {
        key: value / num_batches
        for key, value in loss_sums.items()
    }

    return averages


# ============================================================
# Save CSV
# ============================================================

def save_history(history):

    csv_path = (
        OUTPUT_DIR /
        "training_history.csv"
    )

    fieldnames = [
        "epoch",
        "total_loss",
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
    ]

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(history)

    return csv_path


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)

    print(
        "Faster R-CNN + MobileNetV3 "
        "Loss Diagnostic Training"
    )

    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Image size: "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Epochs: {NUM_EPOCHS}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = SmokeDataset(
        TRAIN_IMAGES,
        TRAIN_LABELS
    )

    val_dataset = SmokeDataset(
        VAL_IMAGES,
        VAL_LABELS
    )

    print(
        f"Training images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(val_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Created for future validation
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model()

    model.to(DEVICE)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.SGD(
        params,
        lr=0.0005,
        momentum=0.9,
        weight_decay=0.0005
    )
    # ============================================================
    # Resume Training
    # ============================================================

    start_epoch = 1

    if RESUME_CHECKPOINT.exists():

        print(
            "\n"
            "============================================================"
        )

        print(
            f"Loading checkpoint: {RESUME_CHECKPOINT}"
        )

        checkpoint = torch.load(
            RESUME_CHECKPOINT,
            map_location=DEVICE
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        # --------------------------------------------------------
        # Override learning rate for stability
        # --------------------------------------------------------

        for param_group in optimizer.param_groups:
            param_group["lr"] = 0.0005

        start_epoch = checkpoint["epoch"] + 1

        print(
            f"✓ Resumed from epoch "
            f"{checkpoint['epoch']}"
        )

        print(
            f"✓ Starting epoch: {start_epoch}"
        )

        print(
            f"✓ Learning rate: "
            f"{optimizer.param_groups[0]['lr']}"
        )

        print(
            "============================================================"
        )
    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print(
        "\nStarting diagnostic training...\n"
    )

    history = []

    best_loss = float("inf")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):

        print(
            f"\nEpoch [{epoch}/{NUM_EPOCHS}]"
        )

        losses = train_one_epoch(
            model,
            optimizer,
            train_loader,
            DEVICE,
            epoch
        )

        # Stop if NaN / Inf occurred
        if losses is None:

            print(
                "\nTraining stopped because "
                "non-finite loss was detected."
            )

            return

        # ----------------------------------------------------
        # Print epoch summary
        # ----------------------------------------------------

        print(
            "\n"
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            "Average Losses:"
        )

        print(
            f"  Total Loss       : "
            f"{losses['total_loss']:.6f}"
        )

        print(
            f"  Classifier Loss  : "
            f"{losses['loss_classifier']:.6f}"
        )

        print(
            f"  Box Regression   : "
            f"{losses['loss_box_reg']:.6f}"
        )

        print(
            f"  Objectness Loss  : "
            f"{losses['loss_objectness']:.6f}"
        )

        print(
            f"  RPN Box Reg      : "
            f"{losses['loss_rpn_box_reg']:.6f}"
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history.append({
            "epoch": epoch,
            "total_loss": losses["total_loss"],
            "loss_classifier": losses[
                "loss_classifier"
            ],
            "loss_box_reg": losses[
                "loss_box_reg"
            ],
            "loss_objectness": losses[
                "loss_objectness"
            ],
            "loss_rpn_box_reg": losses[
                "loss_rpn_box_reg"
            ],
        })

        csv_path = save_history(
            history
        )

        # ----------------------------------------------------
        # Save checkpoint every epoch
        # ----------------------------------------------------

        checkpoint_path = (
            OUTPUT_DIR /
            f"checkpoint_epoch_{epoch}.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "loss": losses[
                    "total_loss"
                ],
            },
            checkpoint_path
        )

        print(
            f"Checkpoint saved: "
            f"{checkpoint_path}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if losses["total_loss"] < best_loss:

            best_loss = losses[
                "total_loss"
            ]

            best_model_path = (
                OUTPUT_DIR /
                "best_model.pth"
            )

            torch.save(
                model.state_dict(),
                best_model_path
            )

            print(
                f"✓ New best model saved "
                f"(loss={best_loss:.6f})"
            )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    final_model_path = (
        OUTPUT_DIR /
        "faster_rcnn_mobilenetv3_"
        "diagnostic.pth"
    )

    torch.save(
        model.state_dict(),
        final_model_path
    )

    print(
        "\n"
        "================================================"
    )

    print(
        "Diagnostic training complete."
    )

    print(
        f"Training history: {csv_path}"
    )

    print(
        f"Final model: {final_model_path}"
    )

    print(
        "================================================"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()