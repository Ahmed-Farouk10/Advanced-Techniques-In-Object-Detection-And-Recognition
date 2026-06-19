# Task 2 — Data Understanding (EDA)

> **Phase 1: Exploratory Data Analysis | Dataset B (Boreal Watchtower/Drone)**

## Objective

Perform comprehensive exploratory data analysis on 4,954 images and 4,862 bounding boxes. Every finding must be linked to a business constraint or a pipeline action.

## Key Findings

| Finding | Value | Business Impact | Pipeline Action |
|---------|-------|----------------|-----------------|
| Total images | 4,954 (4,693 annotated + 256 empty + 5 unpaired) | Minor data gap identified | Resolved in Task 3 |
| Empty (negative) ratio | 5.17% | Watchtower sees clean forest 99.9% of the time; our dataset under-represents negatives | Focal Loss during training |
| Geographic distribution | Evo(931), Heinola(906), Karkkila(1,096), Ruokolahti(1,765) | 4 distinct watchtower locations | Location-stratified split (Task 4) |
| Large plume bias | **95.7%** of plumes >10% image area | Model will fail on early, distant smoke | Aggressive scale/crop augmentations (Task 5) |
| Small plume count | Only 1.3% <1% area (26 total boxes) | Early detection thesis at risk | Report APsmall separately |
| Resolution | 100% 4096×2160 (4K) | 21× compression to 640×640 destroys small plumes | Random cropping instead of resizing |
| Horizon bias | Smoke mean Y-center = 0.395 (top 40%) | Smoke rises, never appears at ground | Disable vertical flip augmentation |
| Daytime bias | 96% bright images (mean 112/255) | Model blind at dawn/dusk/night | HSV jitter + brightness augmentations |
| Video clips | 30 distinct drone flights | Sequential frames share background | Clip-level split (Task 4) |
| Frame displacement | Mean 0.041 normalized (4% per frame) | Quantifies temporal correlation | Justifies clip-level split |
| Aspect ratio | Mean 0.92 (roughly square) | Smoke is not elongated like COCO objects | Custom anchor design |
| Boxes per image | 99.5% single-box | NMS irrelevant for this dataset | Mosaic augmentation for multi-object |
| Anchor clusters | 5 groups: tiny(0.03) → huge(0.73) | Default COCO anchors misaligned | Custom anchors for Faster R-CNN |
| IoU overlap | Max 0.42, no pairs >0.5 | No NMS collision risk | Standard NMS thresholds safe |

## Outputs

- `explore.ipynb` — Main EDA notebook with all visualizations
- `data_understanding.md` — Findings summary with A+ business insights
- `advanced_eda_plots.png` — Spatial heatmap + illumination distribution

## Advanced EDA (Detection-Specific)

Beyond basic statistics, we performed:
- **Anchor box clustering** (k=3,5,9 via k-means)
- **Aspect ratio distribution** analysis
- **Boxes per image** histogram
- **Pairwise IoU** analysis across multi-box images
- **Frame-to-frame displacement** quantification (video-aware)

Code: `../../../shared/advanced_eda_detection.py` (outputs: `anchor_clustering.png`, `aspect_ratio.png`, `boxes_per_image.png`, `iou_analysis.png`, `displacement_analysis.png`)
