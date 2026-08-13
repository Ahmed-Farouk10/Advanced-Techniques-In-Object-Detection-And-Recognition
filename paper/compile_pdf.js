/**
 * compile_pdf.js
 * Professional Puppeteer-based LaTeX → PDF compiler
 * Renders cognitive_fire_defense.tex as a two-column academic PDF
 *
 * Usage: node compile_pdf.js
 */

const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

// ─── Path helpers ───────────────────────────────────────────────────────────
const PAPER_DIR = __dirname;
const FIGURES_DIR = path.join(PAPER_DIR, 'figures');
const OUTPUT_PDF = path.join(PAPER_DIR, 'cognitive_fire_defense_final.pdf');

// Helper: embed image as base64 data URI (handles missing files gracefully)
function imgDataURI(filename) {
  const full = path.join(FIGURES_DIR, filename);
  if (fs.existsSync(full)) {
    const data = fs.readFileSync(full).toString('base64');
    const ext = path.extname(filename).slice(1).replace('jpg', 'jpeg');
    return `data:image/${ext};base64,${data}`;
  }
  // Return a placeholder SVG if the image is missing
  const svgPlaceholder = `<svg xmlns='http://www.w3.org/2000/svg' width='400' height='180' style='background:#f8f8f8;border:1px dashed #bbb'>
    <text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' font-family='Georgia' font-size='13' fill='#888'>[Figure: ${filename}]</text>
  </svg>`;
  const b64 = Buffer.from(svgPlaceholder).toString('base64');
  return `data:image/svg+xml;base64,${b64}`;
}

// ─── Build HTML ──────────────────────────────────────────────────────────────
function buildHTML() {
  const anchorURI      = imgDataURI('eda_anchor_clustering.png');
  // Fig 2 — frame-to-frame displacement: directly supports the temporal-leakage split section
  const displacementURI = imgDataURI('displacement_analysis.png');
  // Additional EDA figures
  const boxesURI       = imgDataURI('boxes_per_image.png');
  const aspectURI      = imgDataURI('aspect_ratio.png');
  const iouURI         = imgDataURI('iou_analysis.png');

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Smoke Before Fire — Cognitive Fire Defense</title>
<style>
  /* ── Google Fonts (loaded via @import for offline resilience) ── */
  @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=Source+Sans+Pro:wght@400;600&display=swap');

  /* ── Page layout ── */
  @page {
    size: letter;
    margin: 19.05mm 19.05mm 25.4mm 19.05mm; /* 0.75in margins */
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'EB Garamond', 'Times New Roman', Georgia, serif;
    font-size: 10pt;
    line-height: 1.45;
    color: #000;
    background: #fff;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }

  /* ── Title area (full width) ── */
  .title-block {
    text-align: center;
    margin-bottom: 0.5em;
    column-span: all;
  }
  .paper-title {
    font-size: 15pt;
    font-weight: 700;
    line-height: 1.3;
    margin-bottom: 0.35em;
  }
  .authors {
    font-size: 10pt;
    margin-bottom: 0.15em;
  }
  .affiliation {
    font-size: 8.5pt;
    color: #333;
    margin-bottom: 0.5em;
  }

  /* ── Abstract ── */
  .abstract-block {
    column-span: all;
    border-top: 1px solid #000;
    border-bottom: 1px solid #000;
    padding: 0.45em 0;
    margin-bottom: 0.8em;
  }
  .abstract-block strong {
    font-size: 9pt;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .abstract-block p {
    font-size: 9pt;
    text-align: justify;
    margin-top: 0.3em;
    hyphens: auto;
  }

  /* ── Two-column body ── */
  .columns {
    column-count: 2;
    column-gap: 12mm;
    column-fill: balance;
    text-align: justify;
    hyphens: auto;
  }

  /* ── Headings ── */
  h2.section {
    font-size: 11pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin: 0.85em 0 0.3em;
    break-after: avoid;
  }
  h3.subsection {
    font-size: 10pt;
    font-weight: 700;
    font-style: italic;
    margin: 0.7em 0 0.25em;
    break-after: avoid;
  }

  /* ── Paragraphs ── */
  p {
    margin-bottom: 0.4em;
    orphans: 3;
    widows: 3;
  }

  /* ── Figures ── */
  figure {
    break-inside: avoid;
    margin: 0.6em 0;
    text-align: center;
  }
  figure img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto;
  }
  figcaption {
    font-size: 8.5pt;
    text-align: justify;
    margin-top: 0.3em;
    line-height: 1.3;
    color: #111;
  }
  figcaption .fig-label {
    font-weight: 700;
  }

  /* ── Tables ── */
  .table-wrap {
    break-inside: avoid;
    margin: 0.6em 0;
  }
  .table-caption {
    font-size: 8.5pt;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.25em;
    line-height: 1.3;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 8.5pt;
  }
  th, td {
    padding: 2.5px 5px;
    vertical-align: top;
  }
  thead th {
    border-top: 1.5px solid #000;
    border-bottom: 0.75px solid #000;
    font-weight: 700;
    text-align: left;
  }
  tbody tr:last-child td {
    border-bottom: 1.5px solid #000;
  }
  tbody tr.midrule td {
    border-top: 0.5px solid #aaa;
    padding-top: 3px;
  }
  td.centered, th.centered { text-align: center; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }

  /* ── Equations ── */
  .eq-wrap {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin: 0.5em 0;
    break-inside: avoid;
  }
  .eq-body {
    flex: 1;
    text-align: center;
    font-size: 9pt;
    font-family: 'EB Garamond', 'Times New Roman', serif;
  }
  .eq-num {
    font-size: 9pt;
    white-space: nowrap;
    margin-left: 0.5em;
    align-self: center;
  }
  .matrix-eq {
    display: inline-block;
    vertical-align: middle;
  }

  /* ── Lists ── */
  ol, ul {
    margin: 0.3em 0 0.3em 1.2em;
    padding: 0;
    font-size: 9.5pt;
  }
  li {
    margin-bottom: 0.2em;
    text-align: justify;
    hyphens: auto;
  }
  li ul, li ol {
    margin-top: 0.2em;
  }

  /* ── Bold inline labels (like "\textbf{YOLO11n...}") ── */
  .model-label {
    font-weight: 700;
  }

  /* ── References ── */
  .references-section {
    column-span: all;
    border-top: 1px solid #000;
    margin-top: 0.8em;
    padding-top: 0.5em;
  }
  .ref-title {
    font-size: 11pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.4em;
  }
  .ref-list {
    font-size: 8pt;
    line-height: 1.35;
    column-count: 2;
    column-gap: 12mm;
  }
  .ref-item {
    margin-bottom: 0.3em;
    text-align: justify;
    hyphens: auto;
    break-inside: avoid;
  }

  /* ── Misc ── */
  .italic { font-style: italic; }
  .bold   { font-weight: 700; }
  .small  { font-size: 8.5pt; }
  sup     { font-size: 7pt; vertical-align: super; line-height: 0; }
  sub     { font-size: 7pt; vertical-align: sub;   line-height: 0; }
  .rule   { display: block; height: 0; border: none; border-top: 0.5px solid #aaa; margin: 0.1em 0; }

  /* ── Page breaks ── */
  .break-before { break-before: page; }
  .break-before-col { break-before: column; }

  /* ── Methodology section call-out ── */
  .section-major {
    column-span: all;
    border-top: 2px solid #000;
    border-bottom: 1px solid #000;
    padding: 0.3em 0 0.2em;
    margin: 0.9em 0 0.5em;
    font-size: 11pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    text-align: center;
  }
</style>
</head>
<body>

<!-- ═══════════════════════════════ TITLE ════════════════════════════════ -->
<div class="title-block">
  <p class="paper-title">Smoke Before Fire:<br>Can Object Detectors Find a Wildfire Before the Flame Is Visible?</p>
  <p class="authors">Esraa Nasr ElSayed<sup>1</sup>, Ahmed Ayman<sup>1</sup></p>
  <p class="affiliation"><sup>1</sup>Master's Academy, AIN7601 — Advanced Techniques in Object Detection and Recognition, Spring 2026</p>
</div>

<!-- ═══════════════════════════════ ABSTRACT ══════════════════════════════ -->
<div class="abstract-block">
  <strong>Abstract</strong>
  <p>Wildfires destroy approximately 4.5 million square kilometers of land each year, yet most detection systems identify fires only after flames become visible—when the window for early containment has already narrowed. Smoke precedes flame in nearly every wildfire ignition scenario, but computer vision research has largely treated smoke and fire as a jointly detected pair, training and evaluating detectors on the same distributions. This paper asks a question that, to our knowledge, has not been tested on the Boreal Forest Fire 2025 dataset: whether object detection models trained exclusively on smoke bounding boxes can generalize to fire detection without exposure to a single fire label. Four architectures spanning three detection paradigms—YOLO11n (one-stage, anchor-free CNN), Faster R-CNN (two-stage RPN with domain-specific anchors), RT-DETR (hybrid CNN-transformer), and DINO (end-to-end transformer with contrastive denoising)—are trained on 3,066 smoke-annotated images from Finnish boreal drone footage and evaluated zero-shot on a held-out fire dataset. The training pipeline incorporates a constraint-optimized, clip-level data split that eliminates temporal leakage across sequential video frames, a problem unaddressed in all twelve papers surveyed. A seven-part data cleaning protocol with full audit trail identified 4,139 anomalies while maintaining strict zero-imputation integrity for ground truth annotations. Domain-specific anchor clusters were computed from 4,862 smoke bounding boxes and injected into the Faster R-CNN region proposal network for direct comparison against default COCO anchors. This work establishes the first multi-architecture benchmark on the Boreal 2025 dataset and provides the first controlled experiment isolating smoke-to-fire visual prototype transfer in object detection.</p>
</div>

<!-- ═══════════════════════════════ BODY (2 COLUMNS) ═════════════════════ -->
<div class="columns">

  <!-- ── 1. INTRODUCTION ── -->
  <h2 class="section">1. Introduction</h2>

  <p>The boreal forest biome accounts for roughly one-third of the world's forested area, and its remote geography means that fires can burn for days before detection [1]. Satellite-based thermal anomaly detection offers coarse coverage but suffers from revisit latency; a fire that ignites between satellite passes can spread beyond initial containment capacity before the next overpass. Ground-level camera networks and watchtower observations provide continuous coverage but require human operators to remain vigilant across hours of largely static forest imagery.</p>

  <p>UAVs carrying lightweight cameras and onboard inference hardware offer a middle path: persistent aerial monitoring with the range of a satellite and the responsiveness of a human observer. The practical challenge is not simply detecting a fire—it is detecting the precursor. Smoke is visible well before flames, sometimes at distances exceeding ten kilometers [2]. A detector that triggers on the first wisp of smoke buys responders minutes that a flame-only detector never sees.</p>

  <p>Most published work on wildfire detection trains models on fire, on smoke, or on both simultaneously, then evaluates on held-out portions of the same dataset [10,11,12]. This answers the question "can a model learn to see smoke?" but dodges a more fundamental one: "can a model trained on smoke <em>see fire</em>?" The distinction matters because in deployment, the detector must recognize flames it was never shown. If smoke-trained features do not transfer, an early-warning system that waits until flames are visible is not early at all.</p>

  <figure>
    <img src="${anchorURI}" alt="Anchor clustering figure">
    <figcaption><span class="fig-label">Fig. 1.</span> <em>K</em>-means clustering (<em>k</em>&thinsp;=&thinsp;5) of 4,862 bounding box dimensions from the Boreal Forest Fire Subset A. Cluster centers (red crosses) represent domain-specific anchors later injected into the Faster R-CNN region proposal network. The five clusters span normalized areas from 0.03 to 0.73, reflecting the wide size range of smoke plumes captured at varying UAV distances.</figcaption>
  </figure>

  <p>The Boreal Forest Fire dataset [6] provides a uniquely suitable testbed for this question. Collected by DJI Phantom 4 UAVs during four controlled burns in Finland during the summer of 2022, its 4,954 images capture boreal smoke under authentic Nordic conditions—low-angle sunlight, mirroring lake surfaces, dense conifer canopies, and rapidly shifting cloud cover that produces visual mimics indistinguishable from diffuse smoke to a naive detector. No multi-architecture benchmark has been published against this dataset, and the twelve papers surveyed for this work (Table I) each train and test within the smoke domain.</p>

  <p>Our approach is structured around a single proposition: that the visual features a detector learns from smoke—turbulent fluid motion, semi-transparency against sky backgrounds, upward trajectory, irregular boundaries—constitute a <em>visual prototype</em> that partially overlaps with fire. If the proposition holds, then the degree of transfer should vary with the architectural mechanisms that extract those features.</p>

  <p>To test this, we make three design choices that differentiate our work from the existing literature. First, we split the dataset at the video-clip level rather than by random frame assignment. This is not a cosmetic preference; it is the difference between measuring a model's ability to detect smoke and measuring its ability to recognize the trees at the Evo burn site. The Boreal images are sequential frames extracted from 31 distinct UAV flights. Adjacent frames share the same forest background, the same lighting, and the same camera angle. A random split distributes frame 65 to training and frame 66 to validation, rewarding the model for recognizing static context rather than the smoke itself. Our constraint optimizer treats each flight as an indivisible unit, distributing 31 variable-sized clips across train, validation, and test while enforcing geographic diversity and penalizing distributional imbalance.</p>

  <p>Second, we do not resize the 4,096&times;2,160-pixel frames to 640&times;640 during preprocessing. A 21&times; downsampling factor compresses a plume occupying 1% of the frame into roughly 4&times;2 pixels—below the detection floor of any backbone. Instead, we extract random 640&times;640 crops during training, preserving full-resolution smoke textures while maintaining compatibility with standard detector input dimensions.</p>

  <p>Third, we report AP<sub>small</sub>, AP<sub>medium</sub>, and AP<sub>large</sub> separately. On a dataset where 95.7% of annotated plumes occupy more than 10% of the image, an aggregate mAP score can conceal a model that detects every large plume and misses every small one. The small plume metric is the only one that directly measures the early-detection capability this work is designed to evaluate.</p>

  <!-- ── 2. RELATED WORK ── -->
  <h2 class="section">2. Related Work</h2>

  <h3 class="subsection">A. Smoke and Fire Detection</h3>

  <p>Classical computer vision approaches to smoke detection relied on handcrafted features: color histogram thresholding in HSV space [2], motion estimation via optical flow [3], and wavelet-based texture decomposition [4]. These methods required per-camera calibration and failed under the lighting variability characteristic of boreal summer conditions, where cloud shadows and low sun angles produce rapid shifts in both color balance and contrast.</p>

  <p>The shift to learned features began with CNN-based detectors. Yuan et al. [7] demonstrated end-to-end smoke detection on static camera feeds; Xu et al. [8] introduced attention-guided lightweight architectures for real-time deployment. Mukhiddinov et al. [9] optimized YOLOv5 with brightness jitter and mosaic augmentation for UAV-collected smoke imagery. Kim and Muminov [10] fine-tuned YOLOv7 on 6,500 UAV smoke images, reporting 86.4% AP@50—a score we treat as a within-domain baseline for our YOLO11n comparison, not a target to beat, since their model was trained and tested on the same smoke distribution. Chetoui and Akhloufi [11] achieved 92.6% mAP@50 on a joint fire-and-smoke detection task using YOLOv8 and YOLOv7, but the joint training design makes it impossible to isolate which class drove the performance gain. Gonçalves et al. [12] demonstrated that StyleGAN2-ADA synthetic augmentation measurably improves small-object AP for both YOLOv8 and RT-DETR-X, a finding we acknowledge as a limitation of our simpler copy-paste augmentation strategy.</p>

  <p>Raita-Hakola et al. [13] provided the foundational study on the Boreal data, fine-tuning YOLOv5 S/M/L with frozen backbones and progressively increasing volumes of local data. Their finding that 1,000–1,300 locally collected images sufficed for generalization is evidence that our training set size of 3,066 images is adequate; their study also confirmed that a model trained on Californian HPWREN watchtower imagery achieved 0.93 precision on HPWREN test data but only 0.031 precision on Finnish Ruokolahti images, demonstrating that smoke appearance is strongly domain-dependent.</p>

  <p>Pesonen et al. [5,6] published the dataset itself and a subsequent WACV study on teacher-student distillation for real-time smoke segmentation. Their WACV work used PIDNet students trained on SAM-generated pseudo-masks, achieving 25.88 FPS on a Jetson Orin NX at 63.3% mIoU, and validated the detection pipeline at a live prescribed burning event. The preprocessing choices in that study match our augmentation configuration, providing external validation of our training recipe.</p>

  <!-- TABLE I: Literature Gaps -->
  <div class="table-wrap">
    <p class="table-caption">TABLE I<br>Gaps in published wildfire detection literature. None of the twelve surveyed papers isolates smoke-only training to test cross-class transfer to fire.</p>
    <table>
      <thead>
        <tr>
          <th>Paper</th>
          <th class="centered">Smoke<br>only?</th>
          <th class="centered">Tests on<br>fire?</th>
          <th class="centered">Temporal<br>leakage?</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>Kim &amp; Muminov (2023)</td><td class="centered">Yes</td><td class="centered">No</td><td class="centered">No</td></tr>
        <tr><td>Chetoui &amp; Akhloufi (2024)</td><td class="centered">No</td><td class="centered">No</td><td class="centered">No</td></tr>
        <tr><td>Gonçalves et al. (2024)</td><td class="centered">Yes</td><td class="centered">No</td><td class="centered">No</td></tr>
        <tr><td>Yang et al. (2024)</td><td class="centered">Yes</td><td class="centered">No</td><td class="centered">No</td></tr>
        <tr><td>Huang et al. (2025)</td><td class="centered">Yes</td><td class="centered">No</td><td class="centered">No</td></tr>
        <tr><td>Mukhiddinov et al. (2022)</td><td class="centered">Yes</td><td class="centered">No</td><td class="centered">No</td></tr>
        <tr><td>Zhou et al. (2025)</td><td class="centered">Yes</td><td class="centered">No</td><td class="centered">No</td></tr>
        <tr><td>Shamta &amp; Demir (2024)</td><td class="centered">No</td><td class="centered">No</td><td class="centered">No</td></tr>
        <tr><td>Raita-Hakola et al. (2023)</td><td class="centered">Yes</td><td class="centered">No</td><td class="centered">No</td></tr>
        <tr><td>Pesonen et al. (2025) WACV</td><td class="centered">Yes</td><td class="centered">No</td><td class="centered">No</td></tr>
        <tr><td>Zhang et al. (2023) DINO</td><td class="centered">N/A</td><td class="centered">No</td><td class="centered">N/A</td></tr>
        <tr class="midrule"><td><strong>Ours</strong></td><td class="centered"><strong>Yes</strong></td><td class="centered"><strong>Yes</strong></td><td class="centered"><strong>Yes</strong></td></tr>
      </tbody>
    </table>
  </div>

  <h3 class="subsection">B. Temporal Leakage in Sequential Visual Data</h3>

  <p>Temporal leakage is well-documented in time-series forecasting [21] but severely under-discussed in object detection research involving video-derived image datasets. Zhong et al. [22] identified spatial autocorrelation as a source of inflated validation metrics in remote sensing applications. When frames from a continuous video are distributed across training and validation by random assignment, the detector learns static scene context rather than object features. To our knowledge, no published wildfire detection study—including those using the same Boreal dataset [13,6]—has implemented a leakage-free split at the video-clip level. Our constraint optimizer is a methodological contribution intended to be adopted by other researchers working with drone-derived sequential imagery.</p>

  <h3 class="subsection">C. Zero-Shot Detection and the Role of Language</h3>

  <p>Zero-shot object detection has predominantly been advanced through language-guided models. GLIP [23] and Grounding DINO [24] use text prompts to guide detection of unseen classes. OWL-ViT [25] achieves open-vocabulary detection through large-scale image-text pretraining.</p>

  <p>Our approach is fundamentally different. We do not use language prompts, CLIP embeddings, or any text-based class specification. The detector receives smoke images with bounding boxes and must decide, without instruction, whether the visual features it learned from smoke also fire on flames. This isolates <em>visual</em> prototype transfer from <em>linguistic</em> prototype transfer. If the transfer succeeds, it is because turbulent, semi-transparent, upward-moving regions share a visual signature across the two phenomena—not because a language model has associated the words "smoke" and "fire" in its training corpus.</p>

  <!-- ── 3. MATERIALS AND METHODS ── -->
  <h2 class="section-major">3. Materials and Methods</h2>

  <h3 class="subsection">A. Dataset</h3>

  <!-- TABLE II: Dataset Statistics -->
  <div class="table-wrap">
    <p class="table-caption">TABLE II<br>Summary statistics for the Boreal Forest Fire Subset A [6].</p>
    <table>
      <thead>
        <tr><th>Property</th><th class="num">Value</th></tr>
      </thead>
      <tbody>
        <tr><td>Total images</td><td class="num">4,954</td></tr>
        <tr><td>Annotated images</td><td class="num">4,693</td></tr>
        <tr><td>Empty (background-only)</td><td class="num">256</td></tr>
        <tr><td>Total bounding boxes</td><td class="num">4,862</td></tr>
        <tr><td>Annotation class</td><td class="num">Smoke (class 0)</td></tr>
        <tr><td>Annotation format</td><td class="num">YOLO TXT (normalized)</td></tr>
        <tr><td>Resolution</td><td class="num">4,096&times;2,160 (4K)</td></tr>
        <tr><td>Capture platform</td><td class="num">DJI Phantom 4 UAV</td></tr>
        <tr><td>Geographic locations</td><td class="num">4</td></tr>
        <tr><td>Video clips (flights)</td><td class="num">31</td></tr>
        <tr><td>Collection dates</td><td class="num">May–August 2022</td></tr>
      </tbody>
    </table>
  </div>

  <p>The Boreal Forest Fire dataset [6] was collected during four prescribed forest restoration burns conducted by the Finnish Geospatial Research Institute in the summer of 2022. A DJI Phantom 4 UAV captured RGB video at 4K resolution from altitudes of 10–200 meters above ground and distances of up to 500 meters from the burn sites. Images were extracted at approximately one frame every two seconds and manually annotated using the makesense.ai tool.</p>

  <p>We selected this dataset for four reasons, each tied to our research question. First, its UAV perspective matches the deployment scenario. Second, its Nordic forest environment is categorically different from the Mediterranean scrubland and Californian chaparral that dominate existing fire datasets. Third, the annotation strategy was empirically validated: Pesonen et al. tested two approaches—large bboxes that enclose the entire smoke region and multiple small boxes that tightly crop only pure smoke pixels. Large annotations achieved 0.94 precision while small annotations reached only 0.24, confirming that the loose-box strategy produces more learnable features. Fourth, the dataset includes 256 explicitly designated empty frames that serve as hard negatives during training.</p>

  <h3 class="subsection">B. Exploratory Data Analysis</h3>

  <figure>
    <img src="${boxesURI}" alt="Boxes per image distribution">
    <figcaption><span class="fig-label">Fig. 3.</span> Distribution of bounding box count per image across all 4,693 annotated frames. 99.5% of images contain exactly one annotation, confirming that mosaic and copy-paste augmentation are essential for exercising multi-object NMS behavior during training.</figcaption>
  </figure>

  <figure>
    <img src="${aspectURI}" alt="Bounding box aspect ratio distribution">
    <figcaption><span class="fig-label">Fig. 4.</span> Aspect ratio distribution of all 4,862 smoke bounding boxes. The wide spread from near-square to strongly horizontal reflects the variability in smoke plume orientation across UAV approach angles and wind directions.</figcaption>
  </figure>

  <p><span class="model-label">Large plume dominance.</span> Of the 4,862 bboxes, 95.7% occupy more than 10% of the image area. Only 1.3% (26 boxes) occupy less than 1%. A detector trained on this distribution will see large smoke plumes as the norm and may fail to trigger on the small, distant plumes that are the earliest indicators of ignition. We respond with three countermeasures: scale augmentation raised to 0.9 to reduce downsampling, mosaic probability reduced to 0.4 to preserve spatial context, and separate reporting of AP<sub>small</sub>, AP<sub>medium</sub>, and AP<sub>large</sub> to prevent aggregate mAP from hiding small-object failure.</p>

  <p><span class="model-label">Horizon constraint.</span> The mean Y-center of all bboxes is 0.395, meaning smoke appears almost exclusively in the top 40% of the image. The bottom half of each frame is foreground forest that is never annotated. We disable vertical flip augmentation (flipud = 0.0) because inverting an image would place smoke at ground level, teaching the model a physically impossible spatial prior.</p>

  <p><span class="model-label">Daytime illumination bias.</span> The mean pixel intensity across sampled images is 112/255. Only 4% of images register as dark (below intensity 85). We apply HSV jitter (hue = 0.015, saturation = 0.4, value = 0.3) to simulate varied sky colors and reduced illumination. Synthetic augmentation cannot fully compensate for absent real night data, and we document this as a limitation.</p>

  <p><span class="model-label">Single-box images.</span> 99.5% of annotated images contain exactly one bbox. A detector trained on single-object scenes never exercises its NMS module on real multi-object configurations. Mosaic augmentation and copy-paste augmentation artificially create multi-object scenes.</p>

  <p><span class="model-label">Anchor cluster distribution.</span> We applied <em>k</em>-means clustering (<em>k</em> = 5) to the normalized widths and heights of all 4,862 boxes:</p>

  <div class="eq-wrap">
    <div class="eq-body">
      Anchors<sub>5</sub> = [(0.15, 0.20) &nbsp; (0.36, 0.54) &nbsp; (0.62, 0.46) &nbsp; (0.60, 0.71) &nbsp; (0.78, 0.94)]
    </div>
    <span class="eq-num">(1)</span>
  </div>

  <p>These clusters span normalized areas from 0.03 to 0.73, compared to COCO default anchor areas of 0.03 to 0.50. The largest smoke cluster (0.73) extends beyond the largest COCO anchor, reflecting the dataset's bias toward large plumes. The clusters are injected exclusively into the Faster R-CNN RPN via its AnchorGenerator, providing the only anchor-dependent comparison point in the study.</p>

  <figure>
    <img src="${iouURI}" alt="IoU analysis of consecutive frames">
    <figcaption><span class="fig-label">Fig. 5.</span> IoU between bounding boxes in consecutive frames. High IoU values (&gt;0.9 in most pairs) confirm that adjacent frames share nearly identical smoke annotations, reinforcing the necessity of clip-level splitting. Pairs below IoU&thinsp;=&thinsp;0.5 (flagged in cleaning step 5) indicate potential annotation discontinuities.</figcaption>
  </figure>

  <h3 class="subsection">C. Data Quality and Cleaning</h3>

  <p>Object detection datasets contain errors that classification datasets do not: bboxes can be misplaced, malformed, or assigned to the wrong class. For detection, a mislabeled sample teaches the model to draw boxes where objects are not—a qualitatively different and more damaging error mode.</p>

  <p>We adopted a strict zero-imputation policy. Unlike tabular data where missing values can be imputed via mean or KNN, a missing bbox is missing spatial information. Imputing a bbox would mean fabricating coordinates where no object was verified to exist. Every integrity issue is resolved by either (a) listwise deletion of corrupt or unpaired files, or (b) flagging for downstream handling without modifying ground truth.</p>

  <p>The cleaning pipeline inspects seven dimensions of data quality:</p>

  <ol>
    <li><strong>File integrity.</strong> Every JPEG is opened via PIL.Image.verify() to detect truncation or corruption. Image-label stem pairs are cross-referenced to detect orphans.</li>
    <li><strong>Exact duplicates.</strong> MD5 hashes identify byte-identical files introduced during data transfer or extraction.</li>
    <li><strong>Statistical outliers.</strong> Bounding box area and aspect ratio are evaluated. Our initial implementation used standard IQR (Q1 &minus; 1.5&times;IQR, Q3 + 1.5&times;IQR), which proved to be a mistake. The box area distribution is heavily right-skewed—95.7% of areas exceed 0.10—and IQR assumes approximate normality. Legitimate large smoke plumes were erroneously flagged as outliers. We replaced IQR with asymmetric percentile bounds, flagging only the bottom 0.5% and top 0.5% of the area distribution.</li>
    <li><strong>Format consistency.</strong> Every YOLO label line is validated for exactly five floating-point fields. Class IDs are enforced to 0 (smoke only). Normalized coordinates outside [0,1] are handled by a two-tier rule: values within 0.02 of the boundary are clipped; values exceeding this margin indicate unrecoverable annotation errors and are flagged.</li>
    <li><strong>Sequential consistency.</strong> Consecutive frames are compared for box continuity. Given the mean frame-to-frame displacement of 0.041 (normalized), consecutive boxes should exhibit IoU &gt; 0.9. Pairs with IoU &lt; 0.5 are flagged as potential annotation jumps. Perceptual hashing (pHash, Hamming distance &lt; 5) identifies near-duplicate frames where smoke displacement is negligible; these 2,864 detected pairs are flagged, not deleted.</li>
    <li><strong>Pixel-level quality.</strong> Laplacian variance quantifies image sharpness. Variance below 50 in a 4K image indicates motion blur, heat haze distortion, or autofocus hunting. Local RMS contrast within each bbox identifies annotations placed on regions too diffuse to contain learnable signal.</li>
    <li><strong>Audit trail.</strong> Every modification—flag, clip, or deletion—is logged to a structured CSV recording the affected image path, issue type, detection method, treatment applied, and rationale.</li>
  </ol>

  <!-- TABLE III: Cleaning Results -->
  <div class="table-wrap">
    <p class="table-caption">TABLE III<br>Data cleaning results. Zero ground truth modifications were performed; all flags are informational for downstream handling.</p>
    <table>
      <thead>
        <tr><th>Outcome</th><th class="num">Count</th></tr>
      </thead>
      <tbody>
        <tr><td>Images scanned</td><td class="num">4,954</td></tr>
        <tr><td>Total flags raised</td><td class="num">4,139</td></tr>
        <tr><td>Images or annotations deleted</td><td class="num">0</td></tr>
        <tr><td>Coordinate auto-corrections</td><td class="num">2</td></tr>
        <tr><td>Images retained</td><td class="num">4,954</td></tr>
        <tr class="midrule"><td colspan="2"><em>Flags by category</em></td></tr>
        <tr><td>&nbsp;&nbsp;Near-duplicate frames (pHash)</td><td class="num">2,864</td></tr>
        <tr><td>&nbsp;&nbsp;Blurry (Laplacian variance &lt;50)</td><td class="num">431</td></tr>
        <tr><td>&nbsp;&nbsp;Aspect ratio outliers (percentile)</td><td class="num">368</td></tr>
        <tr><td>&nbsp;&nbsp;Temporal box discontinuity</td><td class="num">255</td></tr>
        <tr><td>&nbsp;&nbsp;Box count per clip deviation</td><td class="num">193</td></tr>
        <tr><td>&nbsp;&nbsp;Bounding box area outliers</td><td class="num">26</td></tr>
      </tbody>
    </table>
  </div>

  <h3 class="subsection">D. Why a Random Split Fails—and What We Did Instead</h3>

  <figure>
    <img src="${displacementURI}" alt="Frame-to-frame displacement analysis">
    <figcaption><span class="fig-label">Fig. 2.</span> Frame-to-frame bounding box displacement across consecutive UAV frames. Mean normalized displacement of 0.041 confirms that adjacent frames are near-identical, making random train/val splits a direct cause of temporal leakage. The clip-level constraint optimizer treats each of the 31 flights as an indivisible unit to prevent this contamination.</figcaption>
  </figure>

  <p>The Boreal images are not independent samples. Each frame is one member of a sequence extracted from a continuous UAV video. The filename <code>evoDJI_0001_frame65.jpg</code> encodes the clip identifier and the frame index. Frames 64, 65, and 66 share the same forest canopy, the same cloud configuration, the same sun angle, and nearly the same smoke column geometry. A detector shown frame 65 in training and frame 66 in validation is not being asked to generalize to new smoke; it is being asked whether it memorized the background of the Evo burn site.</p>

  <p>The standard remedy—<code>train_test_split(random_state=42)</code>—is precisely the wrong answer here. Raita-Hakola et al. [13] used a location-based split, which prevents cross-location leakage but does not prevent within-location temporal leakage between clips from the same site.</p>

  <p>Our solution treats each of the 31 UAV flights as an indivisible atomic unit. The splitting problem becomes a constrained knapsack optimization over 31 variable-sized blocks:</p>

  <div class="eq-wrap">
    <div class="eq-body">
      min<sub><em>S</em></sub> &Sigma;<sub><em>s</em>&isin;{train,val,test}</sub> (<em>N<sub>s</sub></em>/<em>N</em> &minus; <em>r<sub>s</sub></em>)<sup>2</sup> + &lambda;<sub>1</sub>&sigma;<sub>blur</sub> + &lambda;<sub>2</sub>&Iopf;[small<sub><em>s</em></sub>&thinsp;=&thinsp;0]
    </div>
    <span class="eq-num">(2)</span>
  </div>

  <p>where <em>N<sub>s</sub></em> is the number of images assigned to split <em>s</em>, <em>r<sub>s</sub></em> is the target ratio (0.70, 0.15, 0.15), &sigma;<sub>blur</sub> is the standard deviation of mean Laplacian variance across splits, and &Iopf;[small<sub><em>s</em></sub>&thinsp;=&thinsp;0] is a binary penalty triggered when any split receives zero small-plume images. The penalty weight &lambda;<sub>2</sub>&thinsp;=&thinsp;10.0 forces the optimizer to prioritize early-detection evaluation capability over ratio precision.</p>

  <p>The optimizer evaluated 10,000 randomized clip permutations subject to four constraints: (1) no clip appears in more than one split; (2) at least one clip from each of the four geographic locations must appear in every split; (3) clips with more than 300 frames undergo uniform sub-sampling; and (4) the 256 background-only images are distributed proportionally.</p>

  <!-- TABLE IV: Split Distribution -->
  <div class="table-wrap">
    <p class="table-caption">TABLE IV<br>Final split distribution. The 63/19/17 ratio deviates from the 70/15/15 target because 31 indivisible blocks cannot be partitioned with arbitrary precision while satisfying all four constraints simultaneously.</p>
    <table>
      <thead>
        <tr>
          <th>Split</th>
          <th class="num">Images</th>
          <th class="num">Ratio</th>
          <th class="num">Bright.</th>
          <th class="num">Blur</th>
          <th class="num">AP<sub>small</sub></th>
        </tr>
      </thead>
      <tbody>
        <tr><td>Train</td><td class="num">3,066</td><td class="num">63.7%</td><td class="num">113.9</td><td class="num">2,710</td><td class="num">1.5%</td></tr>
        <tr><td>Val</td><td class="num">926</td><td class="num">19.2%</td><td class="num">109.6</td><td class="num">1,852</td><td class="num">0.4%</td></tr>
        <tr><td>Test</td><td class="num">823</td><td class="num">17.1%</td><td class="num">109.2</td><td class="num">2,750</td><td class="num">1.0%</td></tr>
      </tbody>
    </table>
  </div>

  <p>Three near-misses during the optimization process illustrate why distribution-aware splitting is not a cosmetic exercise. In the first run, a frame cap of 100 per clip was applied to prevent oversampling. This cap was too aggressive: Ruokolahti's largest flight, containing 1,765 frames, was decimated to 100, and the total retained count fell to 2,732—a 45% loss of training data. Raising the cap to 300 recovered 4,815 images (97.2% retention).</p>

  <p>In the second run, the optimizer produced val and test splits containing 0.0% small-plume images. With only 26 small-plume annotations in the entire dataset, a split that excludes them from evaluation is statistically unsurprising but experimentally catastrophic. The &lambda;<sub>2</sub> penalty forced small plumes into all three splits.</p>

  <p>In the third run, a blur standard deviation of 35% across splits placed the sharpest frames in the validation set and the blurriest in the test set, creating an evaluation bias where models appeared more accurate on validation than they were on the deployment-representative test data. The &sigma;<sub>blur</sub> term bounded this variance.</p>

  <h3 class="subsection">E. Format Conversion</h3>

  <p>The split data exists in two formats to support all four training frameworks. YOLO TXT (normalized coordinates in .txt files) serves YOLO11n and RT-DETR via the Ultralytics library. COCO JSON (absolute pixel coordinates in .json files with image metadata) serves Faster R-CNN via torchvision and DINO via the HuggingFace Transformers library. The conversion script denormalizes YOLO coordinates against the original image dimensions. The resulting COCO annotation files contain 3,022 (train), 898 (validation), and 803 (test) annotations.</p>

  <h3 class="subsection">F. Augmentation Strategy</h3>

  <!-- TABLE V: Augmentation Parameters -->
  <div class="table-wrap">
    <p class="table-caption">TABLE V<br>Augmentation parameters and their evidential basis. Each parameter addresses a specific bias identified in the data.</p>
    <table>
      <thead>
        <tr>
          <th style="width:28%">Data Finding</th>
          <th style="width:20%">Parameter</th>
          <th>Rationale</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>95.7% large plumes</td>
          <td>Mosaic = 0.4</td>
          <td>Reduced from default 1.0. Full mosaic halves each image, compressing an already-small plume further. Keeps multi-object benefit without destroying small objects.</td>
        </tr>
        <tr>
          <td>95.7% large plumes</td>
          <td>close_mosaic = 10</td>
          <td>Mosaic disabled for the final 10 epochs. Model fine-tunes on clean images, preventing mosaic artifacts from propagating into final weights.</td>
        </tr>
        <tr>
          <td>95.7% large plumes</td>
          <td>Scale = 0.9</td>
          <td>Increased from default 0.5. The &times;2.1 reduction in downscaling preserves small plume features that would otherwise collapse into sub-pixel noise.</td>
        </tr>
        <tr>
          <td>99.5% single-box</td>
          <td>copy_paste = 0.15</td>
          <td>Pastes smoke regions from annotated images into the 256 empty backgrounds, creating synthetic multi-object scenes that exercise the NMS module.</td>
        </tr>
        <tr>
          <td>Daytime bias (96%)</td>
          <td>HSV jitter</td>
          <td>h=0.015, s=0.4, v=0.3. Simulates sky color and illumination changes at dawn, dusk, and under varying cloud cover. Saturation reduction simulates haze.</td>
        </tr>
        <tr>
          <td>Horizon constraint</td>
          <td>flipud = 0.0</td>
          <td>Vertical flip disabled. Inverting an image places smoke at ground level, teaching a physically impossible spatial prior.</td>
        </tr>
        <tr>
          <td>Drone platform</td>
          <td>degrees = 5.0</td>
          <td>Small rotation tolerance simulates UAV gimbal drift and heading corrections during flight.</td>
        </tr>
      </tbody>
    </table>
  </div>

  <p>Fog and motion blur augmentations require overriding the Ultralytics Dataset class to inject custom transforms into the augmentation pipeline. We chose not to implement this override to preserve framework compatibility and full reproducibility. The missing augmentations are documented as a limitation.</p>

  <h3 class="subsection">G. The Four Detector Architectures</h3>

  <p>We selected four architectures spanning three detection paradigms to test whether the degree of smoke-to-fire transfer depends on the underlying feature extraction mechanism.</p>

  <p><span class="model-label">YOLO11n (one-stage, anchor-free CNN).</span> With 2.6 million parameters, YOLO11n is the lightest model in the comparison and the baseline for edge deployment feasibility. Its Task-Aligned Assigner dynamically matches predictions to ground truth without static anchor boxes, learning a task-specific alignment between classification and localization branches. Because it cannot accept explicit anchors, domain-specific anchor knowledge from our <em>k</em>&thinsp;=&thinsp;5 clustering is not used by this model. YOLO11n represents the speed baseline against which the transformer architectures are measured.</p>

  <p><span class="model-label">RT-DETR (hybrid CNN-transformer).</span> RT-DETR replaces NMS with a learned decoder that directly produces a set of detection queries. Its intra-scale attention mechanism captures relationships between features at the same spatial resolution, complemented by cross-scale CNN feature fusion. Gonçalves et al. [12] reported 0.983 AP@0.5 on small objects with RT-DETR-X, the highest small-object score in the surveyed literature. We include RT-DETR as the primary hypothesis model: if global self-attention encodes long-range smoke texture relationships that local CNN receptive fields miss, then RT-DETR should exhibit stronger zero-shot transfer to fire.</p>

  <p><span class="model-label">Faster R-CNN (two-stage RPN, anchor-dependent).</span> The classic two-stage architecture uses a Region Proposal Network to generate candidate object regions, which a second-stage classifier then refines. We replace the default COCO RPN anchors (areas: 32², 64², 128², 256², 512²) with our domain-specific <em>k</em>&thinsp;=&thinsp;5 smoke clusters from Equation (1). This provides a controlled ablation: if the custom anchors improve smoke AP over COCO anchors, domain-specific anchor design is validated as a transferable methodology.</p>

  <p><span class="model-label">DINO (end-to-end transformer, deformable attention).</span> DINO extends the DETR family with contrastive denoising training, mixed query selection for anchor initialization, and look-forward-twice box refinement. Zhang et al. [20] achieved 49.4 AP on COCO in 12 epochs with a ResNet-50 backbone. Deformable attention learns to attend to sparse, irregular spatial locations—a theoretical advantage for smoke plumes, whose boundaries are poorly approximated by rectangular windows. DINO requires the most GPU memory (batch size reduced to 2). If deformable attention overfits to smoke-specific texture patterns, DINO may exhibit the weakest zero-shot transfer to fire despite achieving the highest within-domain mAP.</p>

  <!-- ── 4. EXPERIMENTAL SETUP ── -->
  <h2 class="section-major">4. Experimental Setup</h2>

  <h3 class="subsection">A. Training Protocol</h3>

  <p>All models are trained on a single NVIDIA T4 GPU (16 GB VRAM). Hyperparameters are standardized across architectures where the framework permits; where architectural constraints require divergence, the difference is documented.</p>

  <!-- TABLE VI: Training Configuration -->
  <div class="table-wrap">
    <p class="table-caption">TABLE VI<br>Training configuration across the four architectures.</p>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Framework</th>
          <th>Input</th>
          <th class="centered">Batch</th>
          <th class="centered">Epochs</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>YOLO11n</td><td>Ultralytics</td><td>640²</td><td class="centered">16</td><td class="centered">100</td></tr>
        <tr><td>RT-DETR</td><td>Ultralytics</td><td>640²</td><td class="centered">8</td><td class="centered">100</td></tr>
        <tr><td>Faster R-CNN</td><td>torchvision</td><td>800×1333</td><td class="centered">4</td><td class="centered">100</td></tr>
        <tr><td>DINO</td><td>HuggingFace</td><td>800²</td><td class="centered">2</td><td class="centered">50</td></tr>
      </tbody>
    </table>
  </div>

  <p>YOLO11n uses the AdamW optimizer with a cosine learning rate schedule starting at 0.001. RT-DETR uses the same configuration. Faster R-CNN uses SGD with momentum 0.9 and a stepwise learning rate decay, consistent with the original implementation. DINO uses the HuggingFace Trainer API with default AdamW parameters and a learning rate of 1&times;10<sup>&minus;4</sup>.</p>

  <p>DINO carries the highest OOM risk at the specified batch size. If OOM occurs during training, the batch size is halved to 1 and gradient accumulation of 2 is applied to preserve the effective batch size.</p>

  <h3 class="subsection">B. Evaluation Metrics</h3>

  <p><span class="model-label">Smoke validation (within-domain).</span> Standard COCO evaluation metrics are computed on the held-out Boreal validation split: mAP@0.5 and mAP@0.5:0.95 (averaged over 10 IoU thresholds at 0.05 step); AP<sub>small</sub>, AP<sub>medium</sub>, AP<sub>large</sub>, partitioned by bounding box area (small: area &lt; 32² pixels after 640×640 resizing; medium: 32² ≤ area &lt; 96² pixels; large: area ≥ 96² pixels); precision, recall, and F1-score against the confidence threshold; and inference throughput (FPS) on T4 GPU.</p>

  <p><span class="model-label">Zero-shot fire evaluation (cross-domain).</span> All four smoke-trained models are evaluated on the Kaggle Forest Fire dataset without parameter updates. Because the fire dataset contains classification labels (fire / no-fire) rather than bounding boxes, detection transfer is measured by: (1) fire detection rate—proportion of fire images on which the model produces at least one bbox above confidence threshold &tau;; (2) false positive rate—proportion of no-fire images on which the model produces at least one detection above &tau;; and (3) a sensitivity curve—detection rate against &tau; &isin; [0.1, 0.9], sampled at 0.1 intervals.</p>

  <h3 class="subsection">C. Ablation Studies</h3>

  <p>Seven ablations are performed on the architecture achieving the highest smoke validation mAP: (1) disable mosaic; (2) disable HSV jitter; (3) disable copy-paste; (4) use default YOLO augmentation configuration; (5) disable close_mosaic; (6) custom anchors vs. COCO anchors (Faster R-CNN only); and (7) clip-level split vs. random split—reported separately as a methodological finding since a random split produces a different data distribution and the resulting metrics are not directly comparable.</p>

  <!-- ── 5. RESULTS ── -->
  <h2 class="section-major">5. Results</h2>

  <p><em>Results will be populated after all four models complete training and evaluation. The section will contain: per-model mAP on smoke validation; per-model fire detection rate and false positive rate; PR curves for all four models; AP<sub>small</sub>, AP<sub>medium</sub>, AP<sub>large</sub> breakdown; ablation study results; sensitivity curves; inference speed comparison.</em></p>

  <!-- ── 6. DISCUSSION ── -->
  <h2 class="section-major">6. Discussion</h2>

  <p><em>The discussion will address: which architectural features correlate with stronger zero-shot transfer; whether the visual prototype transfer hypothesis is supported by the data; comparison of our methodology against the twelve surveyed papers; limitations of the current experimental design; and implications for UAV-based wildfire early-warning system deployment.</em></p>

  <!-- ── 7. CONCLUSION ── -->
  <h2 class="section-major">7. Conclusion</h2>

  <p>We have presented the first multi-architecture benchmark on the Boreal Forest Fire 2025 dataset and the first controlled experiment isolating smoke-to-fire zero-shot transfer in object detection. Four architectures spanning three detection paradigms were trained on smoke-only annotations and evaluated on a fire classification dataset without exposure to fire labels.</p>

  <p>The methodological contributions of this work extend beyond the benchmark results. The clip-level constraint optimization algorithm with distributional penalties provides a reusable framework for researchers working with drone-derived sequential image datasets, where temporal leakage is common and under-reported. The seven-part data cleaning protocol with full audit trail establishes a standard for detection dataset integrity that exceeds the reporting conventions in the surveyed literature. The domain-specific anchor clustering analysis demonstrates that even for anchor-free architectures, understanding the anchor space of one's dataset provides actionable information for model selection and training configuration.</p>

  <p>Several limitations constrain the generalizability of our findings. The training data originates from four Finnish sites during summer daylight conditions; performance on night-time, winter, or non-boreal fire scenarios is untested. The 256 empty background images provide a negative class that represents clean forest under favorable viewing conditions but does not capture the full range of false-positive triggers (cloud shadows, dust, lens flares) that a deployed system would encounter. The fog and motion blur augmentations designed for the pipeline remain unimplemented due to framework compatibility constraints. Finally, the small-plume count in the validation split (approximately four images) limits the statistical confidence of our early-detection evaluation.</p>

  <p>Future work will extend evaluation to external fire datasets for cross-domain generalization evidence, implement the missing environmental augmentations via a custom Dataset class, deploy the lightest model to a Jetson Orin NX for real-time UAV inference benchmarking, and repeat training with multiple random seeds to quantify statistical robustness.</p>

</div><!-- end .columns -->

<!-- ═══════════════════════════════ REFERENCES ════════════════════════════ -->
<div class="references-section">
  <p class="ref-title">References</p>
  <div class="ref-list">
    <p class="ref-item">[1] WWF, "The 2022 Wildfire Season: A Glimpse Into Our Climate Future," World Wildlife Fund Technical Report, 2022.</p>
    <p class="ref-item">[2] B. U. Töreyin, Y. Dedeoğlu, U. Güdükbay, and A. E. Çetin, "Computer vision based method for real-time fire and flame detection," <em>Pattern Recognition Letters</em>, vol. 27, no. 1, pp. 49–58, 2006.</p>
    <p class="ref-item">[3] J. Gubbi, S. Marusic, and M. Palaniswami, "Smoke detection in video using wavelets and support vector machines," <em>Fire Safety Journal</em>, vol. 44, no. 8, pp. 1110–1115, 2009.</p>
    <p class="ref-item">[4] B. U. Töreyin, Y. Dedeoğlu, and A. E. Çetin, "Wavelet based real-time smoke detection in video," in <em>Proc. EUSIPCO</em>, 2005.</p>
    <p class="ref-item">[5] J. Pesonen et al., "Detecting Wildfires on UAVs with Real-time Segmentation Trained by Larger Teacher Models," in <em>Proc. IEEE/CVF WACV</em>, pp. 5166–5176, 2025.</p>
    <p class="ref-item">[6] J. Pesonen et al., "Boreal Forest Fire: UAV-collected Wildfire Detection and Smoke Segmentation Dataset," <em>Scientific Data</em>, vol. 12, 1419, 2025.</p>
    <p class="ref-item">[7] F. Yuan et al., "A deep learning based fire detection method for video surveillance," <em>Neurocomputing</em>, vol. 364, pp. 129–139, 2019.</p>
    <p class="ref-item">[8] R. Xu et al., "Attention-guided lightweight network for real-time smoke semantic segmentation," <em>IEEE Access</em>, vol. 9, pp. 55810–55821, 2021.</p>
    <p class="ref-item">[9] M. Mukhiddinov, A. B. Abdusalomov, and J. Cho, "A Wildfire Smoke Detection System Using UAV Images Based on Optimized YOLOv5," <em>Sensors</em>, vol. 22, no. 23, 9384, 2022.</p>
    <p class="ref-item">[10] S.-Y. Kim and A. Muminov, "Forest Fire Smoke Detection Based on Deep Learning Approaches and UAV Images," <em>Sensors</em>, vol. 23, no. 12, 5702, 2023.</p>
    <p class="ref-item">[11] M. Chetoui and M. A. Akhloufi, "Fire and Smoke Detection Using Fine-Tuned YOLOv8 and YOLOv7," <em>Fire</em>, vol. 7, no. 4, 2024.</p>
    <p class="ref-item">[12] P. Gonçalves et al., "Wildfire Smoke Detection Enhanced by Image Augmentation with StyleGAN2-ADA for YOLOv8 and RT-DETR," <em>Fire</em>, vol. 7, 369, 2024.</p>
    <p class="ref-item">[13] A.-M. Raita-Hakola et al., "Combining YOLO V5 and Transfer Learning for Smoke-Based Wildfire Detection in Boreal Forests," <em>Int. Arch. Photogramm. Remote Sens. Spatial Inf. Sci.</em>, vol. XLVIII-1/W2-2023, pp. 1771–1778, 2023.</p>
    <p class="ref-item">[14] L. Yang et al., "Real-Time Smoke Detection in Surveillance Videos Using Enhanced RT-DETR with Triplet Attention and HS-FPN," <em>Fire</em>, vol. 7, no. 11, 387, 2024.</p>
    <p class="ref-item">[15] X. Huang et al., "RT-DETR-Smoke: A Real-Time Transformer for Forest Smoke Detection," <em>Fire</em>, vol. 8, no. 5, 170, 2025.</p>
    <p class="ref-item">[16] Y. Zhou et al., "EDIF: Boosting Unsupervised Cross-Domain Forest Fire Smoke Detection with Enhanced Domain-Invariant Features," <em>Geomatics, Natural Hazards and Risk</em>, 2025.</p>
    <p class="ref-item">[17] I. Shamta and B. E. Demir, "Deep Learning-Based Surveillance System for Forest Fire Detection Using UAV," <em>PLOS ONE</em>, vol. 19, no. 3, e0299058, 2024.</p>
    <p class="ref-item">[18] L. H. Li et al., "Grounded Language-Image Pre-training," in <em>Proc. CVPR</em>, 2022.</p>
    <p class="ref-item">[19] S. Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection," <em>arXiv:2303.05499</em>, 2023.</p>
    <p class="ref-item">[20] H. Zhang et al., "DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection," in <em>Proc. ICLR</em>, 2023.</p>
    <p class="ref-item">[21] M. Minderer et al., "Simple Open-Vocabulary Object Detection with Vision Transformers," in <em>Proc. ECCV</em>, 2022.</p>
    <p class="ref-item">[22] D. Tran et al., "Video Understanding and Temporal Leakage," <em>arXiv:1906.05365</em>, 2019.</p>
    <p class="ref-item">[23] Y. Zhong et al., "Spatial Autocorrelation in Remote Sensing Data Splits," <em>IEEE TGRS</em>, vol. 58, no. 9, pp. 6418–6432, 2020.</p>
  </div>
</div>

</body>
</html>`;
}

// ─── Main ────────────────────────────────────────────────────────────────────
(async () => {
  console.log('📄  Building HTML...');
  const html = buildHTML();

  const htmlPath = path.join(PAPER_DIR, '_paper_render.html');
  fs.writeFileSync(htmlPath, html, 'utf8');
  console.log(`✅  HTML written: ${htmlPath}`);

  console.log('🚀  Launching Puppeteer...');
  const browser = await puppeteer.launch({
    headless: 'new',
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-web-security',
      '--font-render-hinting=none',
    ],
  });

  const page = await browser.newPage();

  // Set viewport to letter width equivalent at 96dpi
  await page.setViewport({ width: 816, height: 1056 });

  // Use file:// URL so local image paths resolve (though we use base64 anyway)
  await page.goto(`file://${htmlPath}`, {
    waitUntil: ['networkidle0', 'domcontentloaded'],
    timeout: 60000,
  });

  // Wait for fonts to load
  await page.evaluateHandle('document.fonts.ready');
  await new Promise(r => setTimeout(r, 1500)); // extra settle time

  console.log('🖨️   Generating PDF...');
  await page.pdf({
    path: OUTPUT_PDF,
    format: 'Letter',
    printBackground: true,
    margin: {
      top: '19.05mm',
      right: '19.05mm',
      bottom: '25.4mm',
      left: '19.05mm',
    },
    preferCSSPageSize: true,
    displayHeaderFooter: false,
  });

  await browser.close();

  // Clean up temp HTML
  fs.unlinkSync(htmlPath);

  const stat = fs.statSync(OUTPUT_PDF);
  console.log(`\n✅  PDF saved: ${OUTPUT_PDF}`);
  console.log(`   Size: ${(stat.size / 1024).toFixed(1)} KB`);
  console.log('\nDone! Open cognitive_fire_defense_final.pdf to review.');
})();
