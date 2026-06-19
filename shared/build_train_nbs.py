import nbformat as nbf
import os

base = r'dataset-b\model_training'

notebooks = {
    'yolo11n': {
        'model_name': 'YOLO11n',
        'script': 'train.py',
        'desc': 'Anchor-free, one-stage CNN detector. 2.6M parameters. Fastest in comparison. Task-Aligned Assigner instead of explicit anchors. Ideal for edge deployment on drones/Jetson.',
        'config': 'imgsz=640, batch=16, epochs=100, device=0',
        'format': 'YOLO TXT (labels in yolo_format/)',
        'augs': 'custom_hyp.yaml: mosaic=0.4, scale=0.9, hsv jitter, copy_paste=0.15, flipud=0.0'
    },
    'rtdetr': {
        'model_name': 'RT-DETR',
        'script': 'train_rtdetr.py',
        'desc': 'Real-Time Detection Transformer. Query-based, no NMS. Replaces hand-crafted NMS with learned decoder queries. Balances speed and accuracy better than original DETR.',
        'config': 'rtdetr-l.pt, imgsz=640, batch=8, epochs=100, device=0',
        'format': 'YOLO TXT (same as YOLO11n, reuses ../yolo11n/smoke_data.yaml)',
        'augs': 'Same custom_hyp.yaml as YOLO11n (shared via relative path)'
    },
    'faster_rcnn': {
        'model_name': 'Faster R-CNN',
        'script': 'train_faster_rcnn.py',
        'desc': 'Classic two-stage detector. RPN proposes regions, classifier scores them. Only anchor-dependent model: custom k=5 smoke clusters injected into RPN AnchorGenerator. Directly tests whether domain-specific anchors beat COCO defaults.',
        'config': 'mobilenet_v3_large_320_fpn backbone, imgsz=800x1333, batch=4, epochs=100',
        'format': 'COCO JSON (annotations in coco_format/)',
        'augs': 'Torchvision transforms: RandomHorizontalFlip, ColorJitter. Custom anchors: [[0.15,0.20],[0.36,0.54],[0.62,0.46],[0.60,0.71],[0.78,0.94]]'
    },
    'dino': {
        'model_name': 'DINO',
        'script': 'train_dino.py',
        'desc': 'DETR with Improved DeNoising anchOr boxes. Deformable attention for irregular shapes (smoke plumes are irregular). Heaviest model (47M params). Highest OOM risk on T4.',
        'config': 'facebook/dino-resnet-50, imgsz=800x800, batch=2, lr=1e-4, epochs=50',
        'format': 'COCO JSON (annotations in coco_format/)',
        'augs': 'HuggingFace AutoImageProcessor defaults + custom target bounding. If OOM: drop to batch=1.'
    }
}

for dirname, info in notebooks.items():
    dirpath = os.path.join(base, dirname)
    os.makedirs(dirpath, exist_ok=True)
    
    nb = nbf.v4.new_notebook()
    nb.cells = []
    
    nb.cells.append(nbf.v4.new_markdown_cell(f"""# {info['model_name']} - Smoke Detection Training

**Project:** AIN7601 Cognitive Fire Defense Pipeline  
**Training Script:** `{info['script']}`

## Model Architecture

{info['desc']}

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | {info['config']} |
| Data Format | {info['format']} |
| Augmentations | {info['augs']} |

### Why This Model?

This model represents a distinct architectural paradigm in our 4-model comparison:
- **YOLO11n** - One-stage CNN (anchor-free)
- **RT-DETR** - Real-time Transformer (query-based)  
- **Faster R-CNN** - Two-stage RPN (anchor-based, custom smoke anchors)
- **DINO** - Deformable Attention Transformer (irregular shapes)
"""))
    
    nb.cells.append(nbf.v4.new_markdown_cell("""## How Training Works

1. The DataLoader reads images from `../../yolo_format/` or `../../coco_format/`
2. Augmentations are applied **on-the-fly** in GPU memory - no images permanently modified
3. The model learns to predict bounding boxes around smoke
4. Checkpoints saved to `../../{model_name}/weights/`
5. Results (mAP, loss curves) saved to `../../{model_name}/results/`

## Pre-Training Checklist

- [ ] GPU available (`nvidia-smi` shows T4 or better)
- [ ] Data exists in `yolo_format/` and/or `coco_format/`
- [ ] Dependencies installed (ultralytics, torch, torchvision, transformers)
- [ ] custom_hyp.yaml and smoke_data.yaml present (for YOLO/RT-DETR)
"""))
    
    nb.cells.append(nbf.v4.new_markdown_cell("""## Run Training

Execute the cell below to start training. Duration varies by model (30-120 min).
Monitor GPU memory with `nvidia-smi` in a separate terminal.

**Press Ctrl+C to stop early** - the latest checkpoint will be saved.
"""))
    
    nb.cells.append(nbf.v4.new_code_cell(f"""import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.')

print("Starting {info['model_name']} training...")
print("Configuration: {info['config']}")

result = subprocess.run([sys.executable, "{info['script']}"], capture_output=False)
print(f"Training complete with exit code: {{result.returncode}}")
"""))
    
    nbpath = os.path.join(dirpath, 'train.ipynb')
    nbf.write(nb, nbpath)
    print(f'Wrote {nbpath}')

print('All 4 training notebooks generated.')
