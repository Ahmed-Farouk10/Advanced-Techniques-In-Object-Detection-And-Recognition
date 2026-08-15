# YOLO11n Baseline

## Objective

Establish a baseline performance for YOLO11n on the
Boreal Forest Fire Dataset — Subset A before applying
custom augmentation and hyperparameter tuning.

## Model

- Model: YOLO11n
- Architecture: Anchor-Free YOLO detector
- Pretrained weights: `yolo11n.pt`
- Number of classes: 1
- Class: `smoke`

## Dataset

- Training images: 3,792
- Validation images: 984
- Test set: Not used during model development
- Image size: 640 × 640

## Training Configuration

- Epochs: 100
- Batch size: 16
- Device: CPU
- Training configuration: Ultralytics default
- Custom hyperparameters: Disabled
- Custom augmentation configuration: Disabled

## Best Validation Performance

The best validation performance was obtained at epoch 40,
based on the highest mAP50-95 score.

| Metric | Value |
|---|---:|
| Precision | 0.8951 |
| Recall | 0.8686 |
| mAP@50 | 0.9401 |
| mAP@50-95 | 0.6201 |

## Best Epoch

- Epoch: 40
- Precision: 89.51%
- Recall: 86.86%
- mAP@50: 94.01%
- mAP@50-95: 62.01%

## Training Time

The recorded training time at the best epoch was
approximately 42,505.6 seconds (~11.8 hours).

## Interpretation

The baseline achieved high mAP@50 while the stricter
mAP@50-95 metric was substantially lower. This provides
a reference point for evaluating whether subsequent
augmentation and hyperparameter changes improve
localization quality and generalization.

Recall is particularly important for the project because
the objective is early smoke detection, where missed smoke
detections are more critical than false positives.

## Next Phase

The next experiment will evaluate YOLO11n with the proposed
custom augmentation and hyperparameter configuration.

The baseline results will be retained as the reference
for comparison.