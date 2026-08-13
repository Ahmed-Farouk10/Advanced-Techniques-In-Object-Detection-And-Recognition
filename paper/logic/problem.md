# Problem Definition — Cognitive Fire Defense

## Core Question
Does a model trained exclusively on smoke learn visual features that transfer to fire detection without any fire-specific training data?

## Observations
- O1: Wildfire detection systems typically identify fires after flames are visible, not before.
- O2: Twelve published papers on smoke detection train and test within the smoke domain. None isolate smoke training and test on fire.
- O3: The Boreal Forest Fire 2025 dataset contains 4,954 sequential drone frames but has never been used for a multi-architecture benchmark or a smoke-to-fire transfer experiment.
- O4: Sequential video frames cause temporal leakage when split randomly — a problem unaddressed in the smoke detection literature.
- O5: 95.7% of smoke plumes occupy >10% of the image. Aggregate mAP hides small-plume detection failure.

## Gaps
- G1: No controlled experiment testing whether smoke-trained features transfer to fire.
- G2: No multi-architecture benchmark on the Boreal 2025 dataset.
- G3: No published split methodology that prevents temporal leakage in sequential drone footage.
- G4: No domain-specific anchor analysis for smoke detection.

## Research Question (Single, Falsifiable)
Do object detectors trained on smoke bounding boxes detect fire zero-shot, and does the degree of transfer depend on architectural paradigm?
