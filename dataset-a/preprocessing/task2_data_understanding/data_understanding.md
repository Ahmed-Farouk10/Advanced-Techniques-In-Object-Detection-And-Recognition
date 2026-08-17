# Task 2 — Exploratory Data Analysis & Understanding (Dataset A)

> **Dataset:** Fire and Smoke Detection Dataset (Dataset A — Test Probe)

---

## 1. Class & Object Distribution

The testing probe comprises **637 total images**:
* **Fire-positive images:** 459 images (72.05%)
* **Ground-truth fire bounding boxes:** 995 boxes
* **Ground-truth smoke bounding boxes:** 896 boxes
* **Hard negative background images:** 178 images (27.95%)

---

## 2. Bounding Box Geometry: Fire vs. Smoke

| Feature | Dataset A (Fire Probe) | Dataset B (Smoke Training) |
| :--- | :--- | :--- |
| **Median Box Area** | **0.048 (4.8% of image)** | **0.342 (34.2% of image)** |
| **Area $> 10\%$ Image** | **18.3%** | **95.7%** |
| **Aspect Ratio Distribution** | Vertical / Dynamic ($0.4 \le \text{AR} \le 1.8$) | Highly wide plumes ($\text{AR} > 2.0$) |

### Key Insight: The Scale Inversion Phenomenon
In wildland fires, smoke plumes billow across hundreds of canopy meters (hence massive bounding boxes), whereas flame cores are compact and localized. Detectors with fixed large anchor priors trained on smoke must overcome this **scale inversion** during zero-shot transfer.

---

## 3. Image Resolution & Environmental Dynamics
* Original resolutions range from $640\times480$ to $1920\times1080$.
* All probe images are standardized to **$640 \times 640$** during transformation to match the training backbone input size.
* Average Laplacian variance is **482.6**, indicating sharp, in-focus aerial and terrestrial fire captures.
