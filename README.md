# PneumoScore-XAI

**Deep Learning for Chest X-ray Pneumonia Classification**  
Medical Image Classification Challenge 2026 · AI project

## Overview

This project classifies chest X-ray images into two classes: **Normal** and **Pneumonia**. The current selected inference candidate is a 50/50 probability ensemble of a ResNet18 baseline and a ResNet18 model with Efficient Channel Attention (ECA).

The repository contains the **final inference and verification code**. The training notebooks and organizer-provided images are **not included in this source snapshot**.

**Status:** Development and validation complete. The official organizer-provided hidden test set has not yet been evaluated.

## Technology

Python, PyTorch, torchvision, NumPy, pandas, Pillow, Kaggle Jupyter Notebook.

## Method

1. Preprocess chest X-ray inputs with the documented image transform.
2. Load two previously trained ResNet18-family model checkpoints.
3. Average their predicted probabilities with equal weights (50/50).
4. Apply the fixed decision threshold of **0.55**.
5. Generate `Image_id` / `Prediction` CSV outputs (`Normal` or `Pneumonia`).
6. Audit inputs, output formatting, and reproducibility.

## Validation results

These results come from a single selected **seed 2026** pair on **1,028 validation images**, not the official hidden test set.

| Model | Accuracy | Sensitivity | Specificity | F1 | P (weighted score) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet18 baseline | 98.15% | 99.61% | 93.87% | 98.77% | 97.81% |
| ResNet18 + ECA | 98.05% | 98.96% | 95.40% | 98.70% | 97.91% |
| **Ensemble (50/50)** | **98.15%** | **99.09%** | **95.40%** | **98.77%** | **97.99%** |

The performance score is `P = 0.30 × Accuracy + 0.25 × Sensitivity + 0.20 × Specificity + 0.25 × F1`.

These are **validation results used during model development**, not independent clinical validation or final competition results. This model is a research/competition prototype and **must not be used for clinical diagnosis**.

## Repository structure

```text
PneumoScore-XAI/
├── src/
│   └── ensemble_inference.py       # Final inference source
├── notebooks/
│   ├── submission.ipynb            # Inference and submission generation
│   ├── verify_export.ipynb         # Checkpoint/export verification
│   └── verify_pipeline.ipynb       # Pipeline and reproducibility checks
├── results/
│   ├── validation_metrics.csv
│   ├── benchmark_forward.csv
│   ├── benchmark_end_to_end.csv
│   ├── environment.json
│   ├── patch_verification.json
│   └── structural_checks.json
├── docs/
│   ├── VALIDATION_REPORT_VI.md
│   └── CHANGELOG.md
├── requirements.txt
└── .gitignore
```

## Running the project

The notebooks are designed for the **Kaggle** environment. They require authorized access to the matching team model checkpoint (`model.pkl`) and the appropriate input dataset.

1. Open `notebooks/submission.ipynb` in Kaggle.
2. Add the authorized model checkpoint and image dataset as Kaggle inputs.
3. Configure the paths (`TEST_DIR`, `MODEL_PATH`, and optional `SAMPLE_SUBMISSION_CSV`).
4. Run all cells from top to bottom, and inspect the generated run status and audit files.
5. Confirm the expected output labels and CSV schema.

**Note:** This repository does not include `model.pkl`, the organizer's dataset, test images, hidden labels, or per-image organizer data. The complete inference workflow cannot be reproduced without those inputs.

## Verification

The source archive reports successful verification that:
- the two selected architectures and weights remained unchanged after the patch;
- original and reloaded model predictions agreed across 1,028 validation images;
- the inference notebook produced the expected validation-format CSV;
- the official hidden test set was **not** evaluated in this verification run.

For the complete recorded methodology and limitations, see [`docs/VALIDATION_REPORT_VI.md`](docs/VALIDATION_REPORT_VI.md).

## Authorship and competition

Student team project prepared for Medical Image Classification Challenge 2026. Individual contributions should be stated accurately when this work is referenced in a CV.

No dataset, model checkpoint, or third-party training artifacts are redistributed in this repository. No license is granted for reuse unless the authors agree on one.
