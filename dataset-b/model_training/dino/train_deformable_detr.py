"""DETR training for Cognitive Fire Defense (Dataset B, 80/20 clip-level split).

Uses SenseTime/deformable-detr (Deformable DETR).
Fixes the "dead head" Focal Loss collapse by manually overriding focal_alpha to 0.95 
to balance the ~299:1 background-to-object gradient pressure.

Data: dataset-b/coco_format (regenerated from yolo_format 80/20 split via yolo_to_coco.py)
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


class SmokeCOCODataset(Dataset):
    """COCO dataset with pre-decoded, pre-resized image cache.

    Images are decoded + resized to IMG_SIZE once at init (uint8 RAM cache,
    ~1.2 MB/img, ~4.7 GB total) so the training loop has zero disk IO.
    Labels are built by the HF processor each call (cheap on cached arrays),
    which guarantees normalization/box-format correctness.

    Windows multiprocess DataLoader workers are unreliable (hang), so we
    cache in-process instead of parallelizing loading.
    """

    IMG_SIZE = 640

    def __init__(self, coco_dir, split, processor):
        self.coco_dir = Path(coco_dir)
        self.images_dir = self.coco_dir / "images" / split
        self.ann_file = self.coco_dir / "annotations" / f"{split}.json"
        self.processor = processor

        with open(self.ann_file, "r", encoding="utf-8") as f:
            self.coco_data = json.load(f)

        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.annotations = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.image_ids = list(self.images.keys())

        import cv2

        self.pixels = []
        for img_id in self.image_ids:
            img_info = self.images[img_id]
            fname = str(self.images_dir / img_info["file_name"])
            img = cv2.imread(fname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE),
                             interpolation=cv2.INTER_LINEAR)
            self.pixels.append(img)  # uint8 HxWx3

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        image = self.pixels[idx]
        img_info = self.images[img_id]
        
        orig_w = img_info["width"]
        orig_h = img_info["height"]
        scale_x = self.IMG_SIZE / orig_w
        scale_y = self.IMG_SIZE / orig_h

        anns = self.annotations.get(img_id, [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            # Scale coordinates to match the 640x640 resized image
            x, y = x * scale_x, y * scale_y
            w, h = w * scale_x, h * scale_y
            
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        target = {
            "image_id": img_id,
            "annotations": [
                {
                    "bbox": boxes[i],
                    "category_id": labels[i],
                    "area": anns[i]["area"] * scale_x * scale_y,
                    "iscrowd": anns[i].get("iscrowd", 0),
                }
                for i in range(len(boxes))
            ],
        }

        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"].squeeze(),
            "labels": encoding["labels"][0],
        }


def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    return {"pixel_values": torch.stack(pixel_values), "labels": labels}


def train():
    print("Initiating DETR Training Pipeline...")

    script_dir = Path(__file__).resolve().parent
    coco_dir = script_dir.parent.parent / "coco_format"

    model_name = "SenseTime/deformable-detr"
    processor = AutoImageProcessor.from_pretrained(model_name)
    processor.size = {"height": 640, "width": 640}
    train_dataset = SmokeCOCODataset(coco_dir, "train", processor)
    val_dataset = SmokeCOCODataset(coco_dir, "val", processor)
    print(f"train={len(train_dataset)} val={len(val_dataset)}")

    id2label = {0: "smoke"}
    label2id = {"smoke": 0}
    
    config = AutoConfig.from_pretrained(model_name)
    config.id2label = id2label
    config.label2id = label2id
    # CRITICAL FIX: The default focal_alpha=0.25 collapses on single sparse classes
    # due to 299:1 negative gradient pressure. We set it to 0.95 to heavily upweight
    # the positive class and mathematically balance the gradients.
    config.focal_alpha = 0.95
    
    model = AutoModelForObjectDetection.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )

    # RTX 3070 8GB: batch 4 at 640x640, in-memory cached dataset (no disk IO in loop).
    # fp32: Deformable DETR required bf16 (fp16 NaN), but DETR trains cleanly in fp32
    # at 3.27 steps/s (probe-verified) - no precision tricks needed.
    args = TrainingArguments(
        output_dir=str(script_dir / "Cognitive_Fire_Defense" / "DETR"),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=50,
        learning_rate=5e-5,
        weight_decay=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to=[],
        remove_unused_columns=False,
        # 0 workers: Windows multiprocess spawn in HF Trainer hangs; in-memory
        # dataset makes workers unnecessary
        dataloader_num_workers=0,
        seed=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    trainer.train()

    best = trainer.state.best_model_checkpoint
    print(f"Training complete. Best checkpoint: {best}")


if __name__ == "__main__":
    train()