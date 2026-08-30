# Theoretical Framework Changes — cognitive_fire_defense.tex

> **Purpose:** This file documents every theoretical addition made to the paper after the initial empirical draft.
> Use it to re-apply these changes to any new version of the paper (e.g., journal extension, v4 rewrite).
>
> **Branch:** v3
> **Commits:** cd405044 -> 5dd0c9ab -> latest
> **Author:** Ahmed Ayman

---

## Overview: Three Research Gaps Addressed

The initial draft described the work purely as an empirical benchmark.
The following changes elevate it by explicitly naming and closing three theoretical gaps:

| Gap | Status Before | Status After |
|---|---|---|
| 1. Classification gap — what kind of inference is the model performing? | Unnamed / implied | CLOSED — named as visual proxy inference in Introduction |
| 2. Purpose gap — why use relaxed IoU and low confidence? | Looked like a weakness | CLOSED — justified via Bayesian asymmetric risk framing in Discussion |
| 3. Future work gap — where does this lead? | Absent | CLOSED — Two-Stage Alarm Architecture proposed + 3 future directions in Conclusion |

---

## Change 1: Introduction — Visual Proxy Inference Paragraph (Classification Gap)

**File:** paper/cognitive_fire_defense.tex
**Location:** Section 1 (Introduction), after the evaluation-objective gap paragraph.

**What was added:**

```
A third gap concerns the characterization of the transfer process itself.
Existing zero-shot detection literature frames cross-class generalization
as a semantic embedding problem, relying on language-vision alignment to
bridge unseen categories [18], [19], [21]. This study examines a distinct
case: purely visual proxy inference, in which a model generalizes from a
learned precursor state (smoke) to an unobserved co-occurring state (fire)
without linguistic supervision or target-class labels. It remains unknown
whether any detection architecture can perform this form of inference and,
if so, which architectural properties enable it.
```

**Why this was needed:**
Without this, the paper had no formal name for what it was studying.
The term "visual proxy inference" gives reviewers a theoretical handle on
the contribution, distinct from language-vision zero-shot work (GLIP, Grounding DINO, OWL-ViT).

**References it relies on:** [18] GLIP, [19] Grounding DINO, [21] OWL-ViT
— already in bibliography, no new citations needed.

**Adaptation notes for future versions:**
- If the paper adds attention visualization (GradCAM / DINO attention maps),
  strengthen this paragraph: change "It remains unknown whether..." to
  "This study demonstrates that transformer architectures perform visual proxy
  inference, as evidenced by attention map analysis in Section X."
- Keep the term "visual proxy inference" consistent across abstract,
  introduction, and discussion for reviewer clarity.

---

## Change 2: Introduction — Evaluation Objective Gap Paragraph (Purpose Gap Setup)

**File:** paper/cognitive_fire_defense.tex
**Location:** Section 1 (Introduction), planted before Change 1.

**What was added:**

```
A further gap exists in the framing of evaluation objectives. Standard
object detection benchmarks optimize for precision under strict geometric
localization criteria (e.g., IoU >= 0.50). For a system whose operational
purpose is reducing the probability of undetected ignition events, this
criterion is misaligned with the underlying risk profile: in remote boreal
terrain, the cost of a missed fire detection substantially outweighs the
cost of a false positive. No prior study has reported zero-shot fire
sensitivity under an evaluation protocol explicitly designed around this
asymmetric cost structure.
```

**Why this was needed:**
Sets up the Bayesian Discussion section so it reads as a payoff, not an
afterthought. Without this in the Introduction, the relaxed-IoU design
looks like a post-hoc rationalization of poor results.

**Adaptation notes for future versions:**
- If a formal Limitations section is added, include the inverse:
  "The relaxed-IoU protocol inflates image-level detection rates relative
  to deployments requiring precise localization for suppression targeting."

---

## Change 3: section{Implications} -> section{Discussion} (Full Replacement)

**File:** paper/cognitive_fire_defense.tex
**Location:** After Ablation Study, before Conclusion.

**What replaced the original single paragraph:** Three subsections.

### Subsection A: Architectural Dependency in Cross-Domain Transfer

Explains mechanistically why each architecture fails or succeeds:

- CNNs (YOLO11n, Faster R-CNN): locality bias -> texture-dependent features ->
  smoke plume textures != flame textures -> transfer failure under strict IoU
- Deformable DETR: sparse multi-scale attention -> learns spatial relational
  structure (geometry, not texture) -> partial transfer to compact flame geometry
- RT-DETR: intermediate position — NMS-free decoder (transformer property)
  but CNN backbone (texture dependency) -> intermediate zero-shot rates

Key phrase to preserve in future versions:
  "learning spatial relational structure rather than texture identity"
  — this is the one-sentence architectural explanation for the CNN/transformer gap.

### Subsection B: Bayesian Risk Framing of the Evaluation Design (Purpose Gap)

Justifies the recall-over-precision design decision WITHOUT equations
(Option B — safe for IEEE/CV venues, avoids scope rejection):

Core argument chain:
  1. P(F) is extremely low in boreal forest patrol frames
  2. Cost of false negative (L_FN) >> cost of false positive (L_FP)
  3. Lowering confidence to 0.05 + relaxing IoU to 0.10 = minimizing
     the fatal posterior P(F | no alert)
  4. 46,552 FP boxes = accepted cost of suppressing missed detections
  5. 43.32% box recall + 58.82% image detection rate = upper bound of
     purely visual unimodal proxy transfer

IF a future journal version wants full equations (Option A), insert this
BEFORE the existing paragraph:

```latex
Let $F$ denote fire present, $A$ denote a model alert.
The fatal posterior is:
$$P(F \mid \neg A) = \frac{P(\neg A \mid F) \cdot P(F)}{P(\neg A)}$$
where $P(\neg A \mid F) = 1 - \text{Recall}$ is the false negative rate.
Minimizing $P(F \mid \neg A)$ requires maximizing Recall, motivating
the recall-centered evaluation protocol adopted in this study.
```

### Subsection C: Proposed Two-Stage Alarm Architecture (Future Work Gap)

Reframes the paper from a benchmark to a system proposal.

Pipeline sketch (NOT implemented — described as future direction):

```
Stage 1: Deformable DETR at confidence 0.05
         -> high-recall anomaly proposals
         -> 58.82% of fire images flagged
         -> 46,552 FP boxes = investigation candidates, not failures

Stage 2: Lightweight verifier (future work)
         Candidate approaches:
         a) Spectral ratio analysis (SWIR/NIR combustion signature)
         b) Temporal frame differencing (moving flame vs static terrain)
         c) Compact binary classifier trained on small labeled fire sample

Output: Proposals surviving Stage 2 trigger human alert
        or autonomous UAV waypoint redirect
```

Key sentence to preserve verbatim in future versions:
  "The empirical results presented here characterize the first-stage recall
  ceiling achievable through purely smoke-trained visual transfer, establishing
  the performance envelope within which a two-stage system must operate."

---

## Change 4: Conclusion — Expanded from 1 to 3 Paragraphs

**File:** paper/cognitive_fire_defense.tex
**Location:** section{Conclusion}

**Structure of new Conclusion:**

Paragraph 1 — Result summary
  All 4 architectures named, both strict and relaxed localization protocols
  mentioned, Deformable DETR peak number (58.8%) cited.

Paragraph 2 — Bayesian payoff + system contribution named
  FPs reinterpreted as first-stage proposals, Two-Stage Architecture named,
  asymmetric risk structure explicitly referenced.

Paragraph 3 — Three future work directions:
  (1) Second-stage verifier implementation and evaluation
  (2) Extension to nighttime and multi-spectral UAV imagery
  (3) Integration with autonomous flight path replanning

**Adaptation notes for future versions:**
- When Stage 2 verifier is implemented: move direction (1) from Conclusion
  into Results. Update Conclusion to: "A two-stage alarm architecture was
  implemented and evaluated; [results summary]."
- When journal version adds Limitations section: move boreal-only and
  daytime-only caveats from inline text to the dedicated section.

---

## Gap Closure Scorecard

```
BEFORE these changes:
  Paper identity  = "We benchmarked 4 models on smoke -> fire transfer"
  Gaps closed     = 0 / 3

AFTER these changes:
  Paper identity  = "We characterized visual proxy inference under Bayesian-
                     justified recall-centered criteria and proposed a
                     Two-Stage Alarm Architecture"
  Classification gap  = CLOSED (visual proxy inference named, Intro paragraph 4)
  Purpose gap         = CLOSED (Bayesian risk framing, Discussion subsection B)
  Future work gap     = CLOSED (Two-Stage Architecture + 3 directions, Conclusion)
  Gaps closed         = 3 / 3
```

---

## Files Modified in This Round

| File | What changed |
|---|---|
| paper/cognitive_fire_defense.tex | Intro: 2 new paragraphs; Discussion: full rewrite; Conclusion: expanded |
| paper/cognitive_fire_defense.pdf | Recompiled — all changes reflected |
| CHANGELOG.md | RT-DETR row added to within-domain table; zero-shot RT-DETR metrics updated |

---

## Vocabulary Glossary — Keep These Terms Consistent

If adapting to a new paper version, preserve these exact terms:

| Term | Meaning | First appears |
|---|---|---|
| visual proxy inference | Model infers unseen target (fire) from learned precursor (smoke) without text supervision | Introduction |
| asymmetric cost structure | L_FN >> L_FP in early-warning contexts | Introduction |
| recall ceiling | Upper bound of detection rate achievable through purely visual proxy transfer | Discussion + Conclusion |
| first-stage anomaly proposer | Role of Deformable DETR in Two-Stage Architecture | Discussion |
| second-stage verifier | Lightweight module filtering Stage 1 proposals | Discussion + Conclusion |
| fatal posterior | P(F | no alert) — probability fire exists given model is silent | Discussion |
| performance envelope | Range within which a two-stage system must operate, defined by Stage 1 recall | Conclusion |
