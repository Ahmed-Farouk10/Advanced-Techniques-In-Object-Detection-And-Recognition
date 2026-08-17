# Task 5: Feature Engineering & Test-Time Transforms

> **Objective:** Define evaluation transforms and probe invariance to environmental distortions.

---

## 1. Test-Time Evaluation Protocol

Unlike training datasets where heavy stochastic augmentations (Mosaic, MixUp, RandomAffine) are used to regularize weights, testing probes must maintain strict spatial fidelity.

### Standard Test Transform:
* **Image Resizing:** $640 \times 640$ Bilinear
* **Pixel Normalization:** Scale $[0, 255] \to [0.0, 1.0]$ with ImageNet mean/std $(\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225])$ where required by torchvision / HuggingFace backbones.

---

## 2. Invariance & Robustness Probes

To test whether zero-shot smoke models generalize under degraded atmospheric conditions, the evaluation framework supports testing under Albumentations synthetic distortions:
* **Atmospheric Haze:** `A.RandomFog(fog_coef_range=(0.1, 0.3))`
* **Sensor Noise:** `A.GaussNoise(var_limit=(10, 50))`
* **Solar Glare / Illumination:** `A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2)`
