# Unseen Fire Generalization Evaluation Report
## Smoke-Only Object Detection with YOLO11n and Faster R-CNN + MobileNetV3

**Project:** Advanced Techniques in Object Detection and Recognition  
**Research Question:** Can an object detector trained only on smoke learn visual features that generalize to unseen fire?

---

## 1. Executive Summary

This experiment evaluates whether detectors trained **only on smoke** can localize **fire**, even though fire images/boxes were not used as training targets.

Two object detectors were evaluated:

1. **YOLO11n**
2. **Faster R-CNN + MobileNetV3**

The unseen-fire test set contains:

- **637 total test images**
- **459 images containing fire**
- **995 ground-truth fire bounding boxes**

A prediction is counted as an **unseen fire detection** when its predicted bounding box overlaps a ground-truth fire box by **IoU >= 0.5**, regardless of the predicted class. This is important because the models were trained only on smoke.

The results show that **Faster R-CNN generalizes better than YOLO11n**, but the absolute unseen-fire detection performance remains very low.

At the most permissive evaluated threshold (**confidence = 0.05**):

| Model | Precision | Recall | F1 | Fire Image Detection | Fire Box Detection |
|---|---:|---:|---:|---:|---:|
| YOLO11n | 0.0037 | 0.0060 | 0.0046 | 1.31% | 0.60% |
| Faster R-CNN | 0.0139 | **0.0362** | 0.0201 | **7.41%** | **3.62%** |

At confidence **0.10**, Faster R-CNN achieves its best F1 score:

- Precision = **0.0175**
- Recall = **0.0281**
- F1 = **0.0216**

The qualitative examples also show an important pattern: the successful unseen-fire detections tend to occur where **fire and smoke are visually co-located**, while many failures occur in scenes dominated by strong fire color/brightness, very large fire regions, multiple fire instances, or smoke/fire boundaries that differ substantially from the smoke patterns seen during training.

The results therefore provide **evidence of limited cross-phenomenon generalization**, but they do **not** support the claim that a smoke-only detector can reliably detect fire.

---

# 2. Research Objective

The central research question is:

> **Can a smoke-only object detector detect fire without being trained on fire images?**

This is a form of **unseen-category / cross-phenomenon generalization** experiment.

The intended reasoning is:

- Smoke and fire are physically related phenomena.
- Smoke images contain visual cues such as haze, texture, low-contrast regions, plumes, and spatial structures.
- A detector trained on smoke may learn some visual features that are also present around fire.
- If the detector localizes fire despite never receiving fire as a training target, this provides evidence that some learned visual features generalize beyond the training concept.

However, successful localization does **not** necessarily mean the model has learned the semantic concept of "fire." It may instead be responding to correlated visual cues such as smoke, brightness transitions, color gradients, or scene texture.

---

# 3. Experimental Setup

## 3.1 Training Concept

Both models were trained on **SMOKE only**.

The Faster R-CNN model was created as:

- Faster R-CNN
- MobileNetV3 backbone
- FPN
- pretrained initialization
- single target class corresponding to smoke

The relevant model construction uses a Faster R-CNN MobileNetV3 FPN model and replaces the default ROI predictor with a predictor configured for the project's number of classes.

The training configuration used for the experiment was limited to **5 epochs**.

This short training schedule is an important experimental limitation and should be considered when interpreting the results.

---

# 4. Test Dataset

The unseen-fire evaluation uses the fire-and-smoke test dataset.

### Dataset statistics

| Quantity | Value |
|---|---:|
| Total test images | 637 |
| Images containing fire | 459 |
| Ground-truth fire boxes | 995 |
| IoU threshold | 0.50 |
| Evaluation confidence thresholds | 0.50, 0.30, 0.20, 0.10, 0.05 |
| Inference confidence | 0.01 |

The test set contains images with fire annotations, but the models were not trained to detect fire.

---

# 5. Evaluation Method

## 5.1 Why IoU = 0.5?

Intersection over Union (IoU) measures the overlap between a predicted bounding box and a ground-truth bounding box.

For this experiment:

> A prediction is considered a successful unseen-fire detection when IoU >= 0.5.

This means that the predicted box must overlap the fire ground-truth box sufficiently to be considered a meaningful localization.

---

## 5.2 Important Class-Handling Decision

The detector was trained only on smoke.

Therefore, the evaluation does **not** require the model to explicitly predict the class "fire."

Instead:

1. Run the smoke-only detector.
2. Collect predicted bounding boxes.
3. Compare those predicted boxes with fire ground-truth boxes.
4. If IoU >= 0.5, count the prediction as an **UNSEEN FIRE DETECTION**.
5. The predicted class itself is ignored for this specific generalization test.

This allows the experiment to answer the intended research question:

> Does the detector produce spatially meaningful detections on unseen fire regions?

---

# 6. YOLO11n Results

The YOLO11n model was trained only on smoke.

### Confidence = 0.05

| Metric | Result |
|---|---:|
| Predictions | 1,628 |
| True Positives | 6 |
| False Positives | 1,622 |
| False Negatives | 989 |
| Precision | 0.0037 |
| Recall | 0.0060 |
| F1 | 0.0046 |
| Fire images detected | 6 / 459 |
| Image detection rate | 1.31% |
| Fire boxes detected | 6 / 995 |
| Box detection rate | 0.60% |

### YOLO11n across thresholds

| Confidence | Precision | Recall | F1 | Image Detection | Box Detection |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.0000 | 0.0000 | 0.0000 | 0.00% | 0.00% |
| 0.30 | 0.0026 | 0.0010 | 0.0014 | 0.22% | 0.10% |
| 0.20 | 0.0018 | 0.0010 | 0.0013 | 0.22% | 0.10% |
| 0.10 | 0.0042 | 0.0040 | 0.0041 | 0.87% | 0.40% |
| 0.05 | **0.0037** | **0.0060** | **0.0046** | **1.31%** | **0.60%** |

The YOLO11n results indicate very limited unseen-fire generalization.

---

# 7. Faster R-CNN Results

The Faster R-CNN + MobileNetV3 model was evaluated using the same unseen-fire protocol.

## 7.1 Results Across Confidence Thresholds

| Confidence | Predictions | TP | FP | FN | Precision | Recall | F1 | Image Detection | Box Detection |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 567 | 10 | 557 | 985 | 0.0176 | 0.0101 | 0.0128 | 1.96% | 1.01% |
| 0.30 | 785 | 14 | 771 | 981 | 0.0178 | 0.0141 | 0.0157 | 2.61% | 1.41% |
| 0.20 | 1,027 | 17 | 1,010 | 978 | 0.0166 | 0.0171 | 0.0168 | 3.27% | 1.71% |
| 0.10 | 1,603 | 28 | 1,575 | 967 | **0.0175** | 0.0281 | **0.0216** | 5.66% | 2.81% |
| 0.05 | 2,585 | 36 | 2,549 | 959 | 0.0139 | **0.0362** | 0.0201 | **7.41%** | **3.62%** |

### Best operating points

- **Best recall:** confidence = 0.05 → **3.62% box recall**
- **Best F1:** confidence = 0.10 → **0.0216**
- **Best precision:** confidence = 0.30 → **0.0178**

This demonstrates the expected precision-recall trade-off:

- Lower confidence threshold → more predictions → higher recall
- But lower confidence also introduces many false positives → precision remains extremely low

---

# 8. YOLO11n vs Faster R-CNN

Faster R-CNN performs substantially better than YOLO11n on the unseen-fire task.

At confidence = 0.05:

| Metric | YOLO11n | Faster R-CNN |
|---|---:|---:|
| Precision | 0.0037 | 0.0139 |
| Recall | 0.0060 | **0.0362** |
| F1 | 0.0046 | **0.0201** |
| Image detection | 1.31% | **7.41%** |
| Box detection | 0.60% | **3.62%** |

Relative to YOLO11n at confidence = 0.05:

- Faster R-CNN recall is approximately **6.0× higher**.
- Faster R-CNN F1 is approximately **4.4× higher**.
- Faster R-CNN image detection rate is approximately **5.7× higher**.
- Faster R-CNN box detection rate is approximately **6.0× higher**.

This is a meaningful difference between the two architectures, although both models remain weak for reliable unseen-fire localization.

---

# 9. Qualitative Analysis

Quantitative metrics alone do not explain *why* some fire instances are detected and others are missed.

The qualitative analysis therefore examines examples of:

1. False negatives
2. True positives
3. Prediction confidence
4. Bounding-box localization
5. Relationship between smoke and fire
6. Scene complexity
7. Visual similarity between training cues and unseen fire

---

# 10. False Negative Analysis

A false negative occurs when a fire ground-truth box is present but no prediction overlaps it with IoU >= 0.5.

## 10.1 Large Fire Regions

![False Negative – Fire-dominated scene](figures/fn_fire_burning_field.png)

This example contains several clearly visible fire regions. The model produces a prediction, but the predicted box is positioned over an area that does not sufficiently overlap the relevant fire ground-truth boxes.

### Interpretation

The detector appears to respond to a correlated visual region rather than accurately localizing each fire instance.

Possible causes:

- Very large fire regions
- Strong orange/yellow color
- Fire occupying a large percentage of the image
- Smoke and fire appearing together
- Weak spatial correspondence between smoke features and fire boundaries

This is especially important because an object detector trained on smoke learns **where smoke-like structures occur**, not necessarily the exact boundary of flames.

---

## 10.2 Extremely Dense Fire

![False Negative – Dense fire scene](figures/fn_dense_fire_scene.png)

This example contains intense fire activity and multiple overlapping fire regions.

The model produces smoke predictions, but many fire ground-truth boxes remain unmatched.

### Interpretation

The scene is visually far from a simple smoke-only detection problem.

The fire:

- covers large regions,
- contains strong brightness,
- contains complex textures,
- includes multiple adjacent instances,
- creates difficult boundaries.

A smoke-only model may therefore struggle to separate individual fire instances.

---

## 10.3 Smoke-Rich Scene with Multiple Fire Instances

![False Negative – Smoke-rich hillside](figures/fn_smoke_hillside.png)

This image is particularly informative because there is a large amount of visible smoke.

Several fire boxes are present along the fire line, but the predictions are large and poorly aligned.

### Interpretation

This suggests that the model may recognize the **overall smoke/fire scene** but does not necessarily learn a precise fire localization function.

This distinction is important:

> Scene-level visual similarity does not guarantee object-level localization accuracy.

The model can react to smoke-like patterns without being able to identify each individual fire bounding box.

---

## 10.4 Fire Close-Up

![False Negative – Fire close-up](figures/fn_fire_closeup.png)

This type of image contains very strong fire texture and brightness.

The model generates several smoke predictions, but they do not consistently match the fire ground truth.

### Interpretation

The visual appearance of flames can be substantially different from the smoke patterns seen during training.

The detector may therefore be reacting to:

- edges,
- texture,
- high-contrast transitions,
- haze,
- surrounding smoke,

rather than to flame appearance itself.

---

## 10.5 Multiple Fire Boxes in a Dense Scene

![False Negative – Multiple fire boxes](figures/fn_dense_fire_closeup.png)

This example contains multiple ground-truth fire regions in a visually dense scene.

The model generates broad smoke predictions, but many fire boxes remain unmatched.

### Interpretation

This highlights a recurring limitation:

**The model can sometimes detect the general phenomenon but fails at instance-level localization.**

This is consistent with the extremely low recall observed quantitatively.

---

# 11. True Positive Analysis

True positives are the most important qualitative evidence for the generalization hypothesis.

A true positive means that a prediction overlaps a fire ground-truth box with IoU >= 0.5.

---

## 11.1 True Positive: Fire with Smoke

![True Positive – Train scene](figures/tp_train_fire_smoke.png)

The model detects an unseen fire region with:

> **UNSEEN FIRE IoU = 0.71**

The predicted box overlaps the ground-truth fire region substantially.

### Why this example is important

The image contains:

- visible fire,
- visible smoke,
- strong smoke/fire spatial association.

This is a plausible example of cross-phenomenon generalization.

The model was trained on smoke, and the successful fire localization occurs in a region where smoke and fire coexist.

However, the prediction confidence is low.

That means:

> The localization is meaningful, but the model is not highly confident that it is seeing a learned smoke object.

---

## 11.2 True Positive with High IoU

![True Positive – Close fire scene](figures/tp_close_fire_scene.png)

This example reports:

> **UNSEEN FIRE IoU = 0.88**

This is a very strong spatial overlap.

The detector therefore produced a bounding box that closely corresponds to the fire ground-truth region.

### Interpretation

This is strong qualitative evidence that the smoke-only model can occasionally generate a spatially meaningful response to unseen fire.

Again, however, this should not be interpreted as proof that the model has learned "fire" as a semantic category.

It may be exploiting visual characteristics shared between smoke and fire scenes.

---

## 11.3 True Positive in a Complex Fire Scene

![True Positive – Complex fire scene](figures/tp_train_smoke_fire.png)

This example reports:

> **UNSEEN FIRE IoU = 0.75**

The model successfully overlaps a fire region even though the detector was trained only on smoke.

### Interpretation

This reinforces the main qualitative finding:

> Some fire regions contain visual structures sufficiently similar to features learned from smoke that the detector can produce a useful localization.

However, the prediction confidence remains low, again indicating that this is an unusual/generalized response rather than a strongly learned fire category.

---

# 12. What the Qualitative Results Tell Us

The qualitative examples reveal three major patterns.

## Pattern 1 — Successful detections occur where smoke and fire are spatially related

The clearest true positives contain visible smoke around or above the fire.

This supports the hypothesis that the detector may be transferring features such as:

- plume structure,
- diffuse boundaries,
- smoke texture,
- local contrast,
- haze,
- fire-associated atmospheric regions.

---

## Pattern 2 — The model often detects the scene but not the correct fire instance

Many false negatives show large prediction boxes that cover portions of the scene but do not achieve IoU >= 0.5 with individual fire boxes.

This means the model may have some **phenomenon-level sensitivity** but poor **object-level localization**.

---

## Pattern 3 — Fire-only visual regions are difficult

When the image is dominated by intense flames and there is little useful smoke structure, the model often fails.

This is expected from a smoke-only training strategy.

The model has no direct training signal telling it:

> "These flame pixels correspond to an object that should be localized."

---

# 13. Is 5 Epochs the Main Reason for the Low Results?

## Short answer

**It may be one important reason, but we cannot conclude that it is the only reason.**

Five epochs is a relatively short training schedule for a detector, especially when the goal is to learn robust visual representations.

More training can potentially improve:

- smoke localization,
- feature representation,
- convergence,
- confidence calibration,
- generalization to visually related phenomena.

However, simply increasing the number of epochs does **not** guarantee better unseen-fire generalization.

---

# 14. Why More Epochs Could Help

The current model may still be under-trained.

With only 5 epochs, the detector may not have fully learned:

- smoke shape,
- smoke boundaries,
- scale variation,
- background separation,
- small smoke regions,
- large smoke regions,
- different viewpoints,
- different illumination conditions.

If the smoke representation is weak, there is less chance for the learned representation to transfer to fire.

A controlled experiment with longer training is therefore justified.

---

# 15. Why More Epochs Might Not Solve Everything

The main challenge is a **domain/concept mismatch**.

The training target is:

> Smoke

The evaluation target is:

> Fire

Even a perfectly trained smoke detector is not expected to become a reliable fire detector automatically.

The model could become much better at detecting smoke while still failing on fire.

There is therefore an important distinction:

### Training performance

"How well can the model detect smoke?"

versus

### Unseen generalization

"How well can the learned smoke representation transfer to fire?"

These are different questions.

---

# 16. Recommended Controlled Training Experiment

To determine whether the 5-epoch limit is responsible for the poor performance, retrain Faster R-CNN using exactly the same:

- dataset,
- preprocessing,
- train/validation split,
- architecture,
- optimizer,
- learning rate,
- augmentation,
- evaluation script,

but change only the number of epochs.

Recommended experiment:

| Experiment | Epochs |
|---|---:|
| Current baseline | 5 |
| Experiment A | 10 |
| Experiment B | 20 |
| Experiment C | 30 |

The most important rule is:

> **Change one major variable at a time.**

This makes the comparison scientifically meaningful.

---

# 17. What to Compare After Retraining

For every training duration, record:

### Smoke validation performance

- Precision
- Recall
- F1
- mAP@0.5
- mAP@0.5:0.95
- Validation loss

### Unseen-fire performance

At the same thresholds:

- Precision
- Recall
- F1
- Image Detection Rate
- Box Detection Rate
- TP
- FP
- FN

A useful final table will look like:

| Epochs | Smoke mAP50 | Fire Recall | Fire Precision | Fire F1 | Fire Box Detection |
|---:|---:|---:|---:|---:|---:|
| 5 | baseline | 0.0281 | 0.0175 | 0.0216 | 2.81% |
| 10 | TBD | TBD | TBD | TBD | TBD |
| 20 | TBD | TBD | TBD | TBD | TBD |
| 30 | TBD | TBD | TBD | TBD | TBD |

This will allow us to answer:

> Does better smoke learning lead to better unseen-fire generalization?

That is a much stronger research question than simply saying "more epochs should improve the results."

---

# 18. Important Experimental Control

The **test set must remain unchanged**.

Do not train on:

- the fire test images,
- fire annotations,
- any images derived from the test set.

The fire test set should remain completely unseen during training.

Otherwise, the experiment would no longer measure unseen-fire generalization.

---

# 19. Recommended Additional Analysis

The current qualitative analysis can be extended into a more rigorous error taxonomy.

## Category A — Successful transfer

Fire + smoke together and correct localization.

## Category B — Smoke detected, fire missed

Model detects surrounding smoke but not the fire box.

## Category C — Fire detected indirectly

Prediction overlaps fire even though the predicted class is smoke.

## Category D — Scene-level false localization

Large prediction covers a broad region but does not sufficiently overlap a fire object.

## Category E — Fire-dominant failure

Strong flame appearance with limited smoke correspondence.

## Category F — Multi-instance failure

Several fire objects exist but the detector produces one broad box or misses several instances.

This taxonomy can be used in the final research paper.

---

# 20. Interpretation of Confidence Scores

A very important observation is that many true unseen-fire detections have **low confidence**.

Examples include:

- 0.064
- 0.056

Even though these predictions have strong IoU values such as:

- IoU = 0.71
- IoU = 0.88
- IoU = 0.75

This tells us that:

> **Localization quality and model confidence are not the same thing.**

The model can accidentally/generalizationally produce a geometrically correct bounding box while assigning it low confidence because fire is outside its training distribution.

This is actually useful evidence for the research hypothesis.

---

# 21. Why the Confidence Threshold Matters

At lower thresholds, more potential transfer detections become visible.

For Faster R-CNN:

- confidence 0.50 → recall 1.01%
- confidence 0.30 → recall 1.41%
- confidence 0.20 → recall 1.71%
- confidence 0.10 → recall 2.81%
- confidence 0.05 → recall 3.62%

This pattern suggests that some unseen-fire signals exist in the model, but they are assigned relatively low confidence.

However, lowering the threshold also produces many false positives.

At confidence = 0.05:

> 2,585 predictions → only 36 true positives.

Therefore, the detector is **not deployment-ready as a fire detector**.

---

# 22. Scientific Interpretation

The results should be interpreted conservatively.

### Supported conclusion

The experiments provide evidence that a smoke-only detector can occasionally produce spatially meaningful detections on unseen fire regions.

This effect is stronger for Faster R-CNN than YOLO11n.

### Not supported

The experiment does **not** show that:

- smoke-only training produces a reliable fire detector,
- the model understands the semantic concept of fire,
- the detector can replace a fire-trained detector,
- the model is suitable for real-world fire detection.

The correct interpretation is:

> **Limited cross-phenomenon visual generalization exists, but it is weak and highly unreliable.**

---

# 23. Main Findings

## Finding 1

**Faster R-CNN generalizes better than YOLO11n.**

At confidence = 0.05, Faster R-CNN achieves approximately 6× the unseen-fire box recall of YOLO11n.

---

## Finding 2

**The generalization signal is weak but non-zero.**

Faster R-CNN detects 36 of 995 fire boxes at confidence = 0.05.

That is only 3.62%, but it is clearly above zero.

---

## Finding 3

**Successful examples often contain both smoke and fire.**

This supports the possibility that shared visual features drive the transfer.

---

## Finding 4

**The detector struggles with precise instance-level localization.**

Many false negatives show broad predictions that fail the IoU >= 0.5 requirement.

---

## Finding 5

**Low confidence does not necessarily mean zero useful signal.**

Several successful unseen-fire detections have very low confidence but high IoU.

---

## Finding 6

**Five epochs may have limited the learned representation.**

A longer controlled training experiment is necessary before concluding that the observed generalization ceiling is architectural or conceptual.

---

# 24. Recommended Next Steps

### Step 1 — Retrain Faster R-CNN

Run:

- 5 epochs
- 10 epochs
- 20 epochs
- 30 epochs

with all other settings fixed.

### Step 2 — Evaluate smoke validation performance

Record mAP50, mAP50-95, precision, recall, F1, and losses.

### Step 3 — Re-run unseen-fire evaluation

Use exactly the same:

- 637 images
- 459 fire images
- 995 fire boxes
- IoU = 0.5
- confidence thresholds

### Step 4 — Compare the curves

Plot:

- epochs vs smoke mAP
- epochs vs unseen-fire recall
- epochs vs unseen-fire F1

### Step 5 — Repeat qualitative analysis

For each training duration, inspect:

- true positives,
- false negatives,
- false positives,
- confidence scores,
- IoU values.

### Step 6 — Select the best checkpoint

Do **not** select the checkpoint only according to unseen-fire performance, because the fire set is supposed to remain an unseen evaluation set.

Checkpoint selection should be based on the smoke validation set.

---

# 25. Suggested Research Narrative

The strongest way to present this experiment in the paper is:

> We investigated whether object detectors trained exclusively on smoke could exhibit cross-phenomenon generalization to unseen fire. Both YOLO11n and Faster R-CNN + MobileNetV3 were evaluated on a held-out fire dataset without fire-specific training. A prediction was considered an unseen-fire detection when its bounding box achieved IoU >= 0.5 with a fire ground-truth box, regardless of the predicted class. Faster R-CNN demonstrated stronger transfer than YOLO11n, achieving up to 3.62% fire-box recall compared with 0.60% for YOLO11n at a confidence threshold of 0.05. Qualitative analysis showed that successful detections frequently occurred in scenes where smoke and fire were spatially associated, whereas failures were common in fire-dominant, dense, and multi-instance scenes. These findings suggest limited visual feature transfer from smoke to fire, but the very low precision and recall indicate that smoke-only training is insufficient for reliable fire detection. Because the Faster R-CNN model was trained for only five epochs, additional controlled experiments with longer training are required to determine whether the observed limitation is partly attributable to under-training.

---

# 26. Final Conclusion

The unseen-fire experiment is **working as a research experiment** even though the absolute metrics are low.

The low numbers are not necessarily a failure of the project.

In fact, the experiment is answering an interesting question:

> **Can knowledge learned from smoke transfer to a visually related but unseen phenomenon, fire?**

The current evidence suggests:

**Yes, but only weakly.**

Faster R-CNN shows more evidence of this transfer than YOLO11n, and the qualitative examples demonstrate several genuine high-IoU unseen-fire detections.

However:

- precision is very low,
- recall is very low,
- false positives are extremely high,
- many fire instances are missed,
- confidence is generally low.

Therefore, the scientifically defensible conclusion is:

> **Smoke-only training produces limited cross-phenomenon generalization to unseen fire, but the effect is insufficient for reliable fire detection. Longer training should be evaluated as a controlled factor before drawing final conclusions about the model's generalization capacity.**

---

## Appendix A — Key File Locations

### Faster R-CNN checkpoint

```text
/home/esraa/Public/AIN7015/advancedTechniques/Paper_Presentation/project/Advanced-Techniques-In-Object-Detection-And-Recognition/dataset-b/model_training/faster_rcnn/runs/faster_rcnn_loss_check/best_model.pth
```

### Faster R-CNN unseen-fire results

```text
/home/esraa/Public/AIN7015/advancedTechniques/Paper_Presentation/project/Advanced-Techniques-In-Object-Detection-And-Recognition/dataset-b/model_training/faster_rcnn/runs/unseen_fire_evaluation/unseen_fire_faster_rcnn_results.csv
```

### YOLO11n unseen-fire results

```text
/home/esraa/Public/AIN7015/advancedTechniques/Paper_Presentation/project/Advanced-Techniques-In-Object-Detection-And-Recognition/dataset-b/model_training/yolo11n/runs/unseen_fire_evaluation/unseen_fire_results.csv
```

### Test images

```text
dataset-a/test_dataset/Fire-and-Smoke-Detection-Dataset/Fire-and-Smoke-Detection-Dataset/dataset/test/images
```

### Test labels

```text
dataset-a/test_dataset/Fire-and-Smoke-Detection-Dataset/Fire-and-Smoke-Detection-Dataset/dataset/test/labels
```

---

## Appendix B — Qualitative Figures Included in This Report

The `figures/` directory contains the qualitative examples used in this report:

- False-negative fire-dominated scene
- False-negative dense fire scene
- False-negative smoke-rich hillside
- False-negative close-up fire scene
- False-negative multiple-fire scene
- True-positive train/fire/smoke scene
- True-positive close fire scene
- True-positive complex fire scene

These images are included so the report can be reviewed independently of the chat conversation.

---

## Appendix C — Key Terminology

**IoU (Intersection over Union):** Measures overlap between a predicted bounding box and a ground-truth box.

**True Positive (TP):** A prediction that overlaps a fire ground-truth box with IoU >= 0.5.

**False Positive (FP):** A prediction that does not correctly match a fire ground-truth box.

**False Negative (FN):** A fire ground-truth box that was not successfully matched by a prediction.

**Precision:** Proportion of predictions that are correct.

**Recall:** Proportion of ground-truth fire boxes that were detected.

**F1 Score:** Harmonic mean of precision and recall.

**Image Detection Rate:** Percentage of fire-containing images in which at least one fire region was successfully detected.

**Box Detection Rate:** Percentage of all ground-truth fire boxes that were successfully detected.

**Unseen Fire Detection:** A prediction generated by a smoke-only detector whose bounding box overlaps an unseen fire ground-truth box with IoU >= 0.5.

---

**Report status:** Current results and qualitative evidence are based on the 5-epoch Faster R-CNN and YOLO11n experiments described above. Longer-epoch experiments are recommended before the final research conclusion is locked.