# Task 1 — Business Logic

> **Phase 1: Problem Understanding | Dataset B (Boreal Watchtower/Drone Footage)**

## Objective

Translate the operational problem of wildfire early detection into a machine learning task. Define what success means, what constraints exist, and what failure costs.

## Key Questions Answered

| Question | Answer |
|----------|--------|
| What is the business problem? | Detect smoke as a fire precursor — before flames are visible. Stationary watchtower cameras provide continuous coverage. |
| What is the ML task? | Supervised Object Detection. Single class: `Smoke` (class 0). |
| What matters more: Precision or Recall? | **Recall.** A missed smoke plume (False Negative) = forest fire. A false alarm = human checks and dismisses it. |
| What's the ultimate goal? | Prove that models trained on smoke can detect fire zero-shot (cross-domain transfer to Dataset A). |
| What is the #1 risk? | Temporal leakage. Sequential video frames MUST be split by clip ID, not randomly. |
| What's the success criterion? | >85% Recall on Dataset B validation. Measurable Fire Detection Rate >0% on Dataset A zero-shot. |

## Outputs

- `business_logic.md` — Full business translation with constraints and expected outcomes

## For First-Time Students

This task answers "WHY are we doing this?" before any code is written. Every technical decision from Task 2-6 traces back to a business constraint defined here.
