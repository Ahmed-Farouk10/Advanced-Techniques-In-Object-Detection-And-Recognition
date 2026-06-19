import os
import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForObjectDetection, TrainingArguments, Trainer

class SmokeCOCODataset(Dataset):
    def __init__(self, coco_dir, split, processor):
        self.coco_dir = Path(coco_dir)
        self.images_dir = self.coco_dir / 'images' / split
        self.ann_file = self.coco_dir / 'annotations' / f'{split}.json'
        self.processor = processor
        
        with open(self.ann_file, 'r') as f:
            self.coco_data = json.load(f)
            
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
            
        self.image_ids = list(self.images.keys())
        
    def __len__(self):
        return len(self.image_ids)
        
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = self.images_dir / img_info['file_name']
        
        image = Image.open(img_path).convert("RGB")
        
        anns = self.annotations.get(img_id, [])
        boxes = []
        labels = []
        for ann in anns:
            # COCO bbox: [x_min, y_min, width, height]
            x, y, w, h = ann['bbox']
            # Convert to [x_min, y_min, x_max, y_max] for HuggingFace Processor
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            
        target = {
            "image_id": img_id,
            "annotations": [
                {
                    "bbox": boxes[i],
                    "category_id": labels[i]
                }
                for i in range(len(boxes))
            ]
        }
        
        # AutoImageProcessor handles resizing to 800x800 dynamically
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        
        return {"pixel_values": pixel_values, "labels": target}

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    batch_tensors = {}
    batch_tensors["pixel_values"] = torch.stack(pixel_values)
    batch_tensors["labels"] = labels
    return batch_tensors

def train_dino():
    print("Initiating DINO Training Pipeline...")
    
    script_dir = Path(__file__).resolve().parent
    coco_dir = script_dir.parent.parent.parent / 'coco_format'
    
    # Note: User specified facebook/dino-resnet-50
    model_name = "facebook/dino-resnet-50"
    
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"Warning: Failed to load processor for {model_name}. Fallback to generic DetrImageProcessor if needed.")
        raise e
        
    # Load dataset
    train_dataset = SmokeCOCODataset(coco_dir, 'train', processor)
    val_dataset = SmokeCOCODataset(coco_dir, 'val', processor)
    
    # Load Model
    id2label = {0: "smoke"}
    label2id = {"smoke": 0}
    model = AutoModelForObjectDetection.from_pretrained(
        model_name,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    # If OOMs on T4, we will drop batch size to 1 as per constraints.
    # Initial attempt with batch=2, lr=1e-4, epochs=50
    training_args = TrainingArguments(
        output_dir="Cognitive_Fire_Defense/DINO",
        per_device_train_batch_size=2,
        num_train_epochs=50,
        learning_rate=1e-4,
        save_steps=500,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
        data_collator=collate_fn
    )
    
    trainer.train()

if __name__ == "__main__":
    train_dino()
