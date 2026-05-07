# LUNA16 — Pulmonary Nodule Detection
### Deep Learning Pipeline: Candidate Detection + False Positive Reduction

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Dataset — LUNA16](#2-dataset--luna16)
3. [Repository Structure](#3-repository-structure)
4. [Lung Mask Generation](#4-lung-mask-generation-lung_maskpy)
5. [DenseNet-121 Architecture](#5-densenet-121-architecture-densenet121py)
6. [Stage 1 — Candidate Detector Training](#6-stage-1--candidate-detector-training)
7. [Stage 2 — False Positive Reduction](#7-stage-2--false-positive-reduction-fpr-training)
8. [Completed Work Summary](#8-completed-work-summary)
9. [Next Steps — Enhance FPR](#9-next-steps--enhance-false-positive-reduction-fpr)
10. [Requirements](#10-requirements)
11. [References](#11-references)

---
<img width="1233" height="848" alt="pipeline" src="https://github.com/user-attachments/assets/bfb5ce87-26e9-40ae-a2f4-1de241b041d1" />

## 1. Project Overview

This project implements a full end-to-end deep learning pipeline for automatic **pulmonary nodule detection** in 3D CT scans, using the **LUNA16** (LUng Nodule Analysis 2016) benchmark dataset. The system follows a two-stage architecture widely validated in clinical literature:

- **Stage 1 — Candidate Detection:** A DenseNet-121 classifier trained to detect every possible nodule with very high sensitivity, producing many candidates (including false positives).
- **Stage 2 — False Positive Reduction (FPR):** A second DenseNet-121 trained to distinguish true nodules from false positives, dramatically reducing the false-positive rate while preserving sensitivity.

The pipeline covers every step from raw DICOM data acquisition, through lung segmentation, 3D patch extraction, model training, full-scan sliding-window inference, and final prediction refinement.

---

## 2. Dataset — LUNA16

The LUNA16 dataset is a public benchmark derived from the LIDC-IDRI collection. It contains **888 CT scans** with radiologist-annotated nodule locations.

### Key Files

| File | Description |
|------|-------------|
| `candidates.csv` | All nodule candidates with coordinates (x, y, z) and class label (0 = negative, 1 = positive) |
| `annotations.csv` | Ground-truth nodule locations with diameter (mm) — used as positives in Stage 2 |
| `Subsets 0–9` | 10 subsets of CT scans (`.mhd` / `.raw` format) downloaded from Zenodo / Grand Challenge |

### Preprocessing Steps (Stage 0)

1. Download all 10 subsets from Zenodo using the LUNA16 dataset download notebooks.
2. Load `.mhd` volumes and convert to Hounsfield Unit (HU) arrays.
3. Apply lung mask segmentation to remove non-lung tissue.
4. Extract `64×64×64` voxel patches centered on each candidate coordinate.
5. Label patches: `class=1` (nodule) from `annotations.csv`, `class=0` (background) from `candidates.csv`.

---

## 3. Repository Structure

| File / Notebook | Purpose |
|-----------------|---------|
| `EDA_and_Visualization.ipynb` | Exploratory Data Analysis: distribution of nodule sizes, class balance, HU histograms, slice visualizations |
| `LungMask_Generation.ipynb` | Applies the lung segmentation pipeline to all CT scans and saves binary lung masks |
| `lung_mask.py` | Core lung mask module — 7-step morphological pipeline (threshold, close, CC label, border removal, top-2 lobes, fill, dilate) |
| `DenseNet121_Architecture.ipynb` | Architecture exploration notebook — verifies layer structure, parameter count, forward pass shapes |
| `densenet121.py` | Custom DenseNet-121 implementation from scratch in PyTorch (Bottleneck, DenseBlock, TransitionLayer, full model) |
| `LUNA16_Stage1_Complete.ipynb` | Stage 1 full training pipeline — data loading, augmentation, training loop, checkpointing, evaluation |
| `LUNA16_Stage2_Complete.ipynb` | Stage 2 FPR training pipeline — hard-negative mining from Stage 1 false positives, DenseNet-121 FPR model |
| `Testing_Pipeline_Updated.ipynb` | End-to-end inference: full 3D scan → lung mask → sliding window → Stage 1 → Stage 2 → final predictions |
| `pipeline.jpeg` | Visual diagram of the complete 3-stage pipeline (data acquisition, Stage 1 inference, Stage 2 training) |

---

## 4. Lung Mask Generation (`lung_mask.py`)

A custom **7-step morphological pipeline** extracts binary lung masks from raw 3D CT volumes in Hounsfield Units. No deep learning — pure classical image processing using `numpy` and `scipy`.

### Pipeline Steps

1. **Threshold Air** — binary mask where HU < -400 (air density)
2. **Morphological Closing** — 6-connected 3D closing (2 iterations) to seal vessel/noise gaps
3. **Connected Component Labelling** — 26-connected labelling of all air regions
4. **Remove Border Components** — strips any component touching the volume border (scanner background, room air)
5. **Keep Top-2 Largest** — retains only left lung + right lung lobes by voxel count
6. **Fill Interior (slice-by-slice)** — `binary_fill_holes` per axial slice to include dense intra-lung structures
7. **Dilate** — 2-voxel outward dilation to capture subpleural and pleural nodules

### Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `hu_thresh` | `-400 HU` | Air detection cutoff |
| `close_iters` | `2` | Gap-filling aggressiveness |
| `dilate_voxels` | `2` | Subpleural margin expansion |

---

## 5. DenseNet-121 Architecture (`densenet121.py`)

A full PyTorch implementation of **DenseNet-121 from scratch**, following Huang et al. (2017) *"Densely Connected Convolutional Networks"*. Two separate instances of this model are trained for Stage 1 and Stage 2.

### Architecture Summary

| Component | Details |
|-----------|---------|
| **Stem** | 7×7 Conv (stride 2) → BN → ReLU → 3×3 MaxPool (stride 2) → 64 channels |
| **Dense Block 1** | 6 bottleneck layers, growth rate k=32 → 256 channels |
| **Transition 1** | BN → ReLU → Conv 1×1 (θ=0.5) → AvgPool 2×2 → 128 channels |
| **Dense Block 2** | 12 bottleneck layers → 512 channels |
| **Transition 2** | BN → ReLU → Conv 1×1 (θ=0.5) → AvgPool 2×2 → 256 channels |
| **Dense Block 3** | 24 bottleneck layers → 1024 channels |
| **Transition 3** | BN → ReLU → Conv 1×1 (θ=0.5) → AvgPool 2×2 → 512 channels |
| **Dense Block 4** | 16 bottleneck layers → 1024 channels |
| **Classification Head** | BN → ReLU → Global AvgPool → Dropout → Linear(1024, num_classes) |

### Bottleneck Layer

Each layer follows: `BN → ReLU → Conv 1×1 (→ 4k channels) → BN → ReLU → Conv 3×3 (→ k channels)`, then concatenates its output with all previous layers' feature maps (dense skip connection).

### Total Parameters

≈ **7.98 million** trainable parameters (standard DenseNet-121 count).  
Weight initialization: Kaiming Normal for Conv layers, constant 1/0 for BN, Normal(0, 0.01) for Linear.

---

## 6. Stage 1 — Candidate Detector Training

### Objective

Train DenseNet-121 to classify `64×64×64` raw 3D patches as **nodule** (class=1) or **non-nodule** (class=0) with very high sensitivity. A low decision threshold (0.1–0.3) is intentionally used to maximize recall at the cost of precision.

### Training Details

- **Input:** 64×64×64 voxel patches in HU, center-cropped on each candidate coordinate
- **Positives:** candidates flagged `class=1` in `candidates.csv` (confirmed nodules)
- **Negatives:** candidates flagged `class=0` in `candidates.csv` (random lung locations)
- **Augmentation:** random flips, rotations, intensity jitter to combat class imbalance
- **Loss:** Binary Cross-Entropy with class weighting for imbalance
- **Optimizer:** Adam with learning rate scheduling
- **Output:** probability score per patch; threshold 0.1–0.3 for candidate selection

### Inference (Full Scan, Sliding Window)

- Apply lung mask to the full 3D CT volume
- Slide a `64×64×64` window with stride strictly inside the lung mask region
- Run every window through the trained Stage 1 model
- Collect all locations exceeding the low threshold → candidate list
- **Result:** high sensitivity, vast number of false positives *(expected behavior)*

---

## 7. Stage 2 — False Positive Reduction (FPR) Training

### Objective

Train a second DenseNet-121 specifically to distinguish **true nodules** from the large number of false positives produced by Stage 1. This model acts as a refinement filter.

### Training Data Construction

- **Positives:** patches centered on ground-truth nodule locations from `annotations.csv`
- **Negatives (hard negatives):** patches from Stage 1 inference that were classified positive but are actually false positives — these are the hardest, most informative negatives
- This **hard-negative mining** strategy forces the FPR model to learn exactly where Stage 1 fails

### Training Details

- Same DenseNet-121 architecture, freshly initialized
- Higher decision threshold used (model trained to be more discriminative)
- Standard train/validation split with early stopping on validation loss
- **Output:** final per-candidate probability; only high-confidence predictions retained

### Final Output

The testing pipeline chains both models:

```
Lung Mask → Sliding Window → Stage 1 Filter → Stage 2 FPR Filter → Final Nodule Locations
```

The result is a significantly reduced set of predicted nodule locations with far fewer false positives than Stage 1 alone.

---

## 8. Completed Work Summary

| # | Task | Status |
|---|------|--------|
| 1 | LUNA16 dataset download (10 subsets, Zenodo) | ✅ Done |
| 2 | Exploratory Data Analysis & Visualization | ✅ Done |
| 3 | Lung mask generation — 7-step morphological pipeline (`lung_mask.py`) | ✅ Done |
| 4 | DenseNet-121 implementation from scratch (`densenet121.py`) | ✅ Done |
| 5 | Stage 1 training — candidate detector on 64×64×64 patches | ✅ Done |
| 6 | Stage 1 inference — full scan sliding window, low-threshold candidate generation | ✅ Done |
| 7 | Stage 2 training — FPR model with hard-negative mining from Stage 1 false positives | ✅ Done |
| 8 | End-to-end testing pipeline (lung mask → Stage 1 → Stage 2 → predictions) | ✅ Done |

---

## 9. Next Steps — Enhance False Positive Reduction (FPR)

The current two-stage pipeline is functional, but Stage 2 FPR performance can be substantially improved. The following enhancements are recommended in priority order:

### 9.1 Better Hard-Negative Mining

The quality of negatives fed to Stage 2 directly determines its performance.

- **Iterative / bootstrapped hard-negative mining:** run Stage 1 inference → collect new false positives → retrain Stage 2 → repeat. Each iteration produces harder, more informative negatives.
- Stratify negatives by anatomical region (vessels, airways, lymph nodes, chest wall) so the FPR model learns to reject each false-positive type explicitly.
- Balance the ratio of hard negatives to true positives carefully — too many negatives causes the model to become over-conservative.

### 9.2 3D-Aware Architecture

The current DenseNet-121 processes 2D slices or treats the 3D patch as a channel stack. True 3D convolutions capture spatial context in all three dimensions.

- Replace 2D convolutions with **3D convolutions** (`Conv3d`) throughout the DenseNet-121 stem, bottleneck layers, and transitions.
- Alternatively, use a 3D variant such as **3D ResNet-18** or **3D SE-ResNet** for the FPR stage, which are lighter and better studied for nodule classification.
- Input: `64×64×64` or `48×48×48` voxel patches directly as 3D tensors without slice-by-slice processing.

### 9.3 Multi-Scale / Multi-View Input

Nodules range from 3 mm to 30 mm. A single patch size cannot capture context at all scales.

- Extract patches at multiple resolutions (e.g., `64×64×64` at 1mm, `32×32×32` at 2mm, `20×20×20` at 4mm) centered on each candidate.
- Pass each scale through a separate encoder branch and fuse feature maps before the classification head.
- This makes the FPR model robust to both small and large nodules.

### 9.4 Stronger Data Augmentation

Medical imaging datasets are small; aggressive augmentation is critical.

- Elastic deformation and random affine transforms on 3D patches
- HU windowing augmentation (random window width / level shifts to simulate scanner variability)
- Mixup or CutMix adapted for 3D medical patches
- Random noise injection (Gaussian noise, Rician noise for MRI-like behavior)

### 9.5 Ensemble of FPR Models

Ensembling is one of the most reliable improvements in medical image analysis competitions.

- Train 3–5 FPR models with different random seeds, architectures, or hyperparameters.
- Average their softmax probabilities before thresholding.
- Ensemble across folds from cross-validation for maximum data utilization.
- **Expected gain:** 2–5% improvement in FROC AUC without any architectural change.

### 9.6 Evaluation — FROC Curve

Use the official LUNA16 evaluation metric: **Free-Response ROC (FROC)** curve.

- FROC measures sensitivity at fixed false-positive-per-scan rates: `0.125, 0.25, 0.5, 1, 2, 4, 8`.
- **Target:** average FROC sensitivity ≥ 0.85 across all 7 FP/scan thresholds.
- Re-evaluate after every FPR enhancement using the LUNA16 official evaluation script.
- Track **CPM** (Competition Performance Metric) as the single summary number for comparison.

### 9.7 Post-Processing

After the FPR model scores all candidates, apply post-processing to further clean predictions.

- **Non-Maximum Suppression (NMS) in 3D:** merge overlapping predictions within a radius (e.g., 6 mm) by keeping the highest-confidence detection.
- **Minimum size filtering:** discard candidates smaller than 3 mm equivalent diameter.
- **Lung boundary validation:** discard candidates outside the lung mask with margin check.

---

## 10. Requirements

### Python Packages

```bash
pip install torch torchvision numpy scipy scikit-learn SimpleITK matplotlib pandas tqdm
```

### Hardware

- GPU with ≥ 8 GB VRAM recommended for training (NVIDIA CUDA)
- ≥ 32 GB RAM for loading full CT volumes during sliding-window inference
- ≥ 500 GB disk space for all 10 LUNA16 subsets

---

## 11. References

- Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). *Densely Connected Convolutional Networks.* CVPR 2017.
- Setio, A. A. A. et al. (2017). *Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in CT images: The LUNA16 challenge.* Medical Image Analysis.
- Armato, S. G. et al. (2011). *The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI).* Medical Physics.
- [LUNA16 Grand Challenge](https://luna16.grand-challenge.org)

---

*End of README*
