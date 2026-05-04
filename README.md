# 🫁 LUNA16 — Lung Nodule Detection with DenseNet-121

A full end-to-end deep learning pipeline for **automated lung nodule detection** from 3D CT scans, built on the [LUNA16 Grand Challenge](https://luna16.grand-challenge.org/) dataset.

---

## 📋 Pipeline Overview

![Pipeline](pipeline.jpeg)

The system is divided into three main stages:

| Stage | Description |
|-------|-------------|
| **Stage 0** | Data acquisition from Zenodo (subsets 0–9, ~888 CT scans) |
| **Stage 1** | Train a DenseNet-121 **detector** on 64×64×64 nodule patches |
| **Stage 1 Inference** | Sliding window across full CT volumes using a lung mask |
| **Stage 2** | Train a second DenseNet-121 **FPR model** to reduce false positives |

---

## 🗂️ Project Structure

```
LUNA16_PROJECT/
│
├── densenet121.py                  # DenseNet-121 from scratch (PyTorch)
├── lung_mask.py                    # Lung mask generation from scratch (NumPy/SciPy)
│
├── EDA_and_Visualizationnn.ipynb   # Exploratory data analysis & visualization
├── LungMask_Generation.ipynb       # Lung mask generation notebook
├── DenseNet121_Architecture.ipynb  # DenseNet-121 architecture exploration
├── LUNA16_Stage1_Completeee.ipynb  # Stage 1 training + inference
├── LUNA16_Stage2_Completeee.ipynb  # Stage 2 FPR training
├── Testing_Pipeline_Updated.ipynb  # End-to-end testing pipeline
│
├── pipeline.jpeg                   # Full pipeline diagram
├── requirements.txt
└── .gitignore
```

---

## 🧠 Model Architecture

### DenseNet-121
- Block configuration: `[6, 12, 24, 16]`
- Growth rate: `k = 32`
- Channel compression: `θ = 0.5`
- Input: `1 × 64 × 64 × 64` (single-channel 3D patch)
- Output: binary classification (nodule / non-nodule)

### Lung Mask Pipeline (`lung_mask.py`)
1. **Threshold** — isolate air voxels (HU < −400)
2. **Morphological closing** — fill small holes from vessels/noise
3. **Remove border components** — strip external air (scanner background)
4. **Keep top-2 components** — left lung + right lung
5. **Fill interior** — close dense structures inside each lung
6. **Dilate** — expand to capture subpleural nodules

---

## ⚙️ Key Hyperparameters

### Stage 1 — Detector Training
| Parameter | Value |
|-----------|-------|
| Patch size | 64×64×64 voxels |
| HU clip range | [−1000, 400] |
| Train split | 80% |
| Neg:Pos ratio | 10:1 |
| Batch size | 16 |
| Epochs | 20 |
| Learning rate | 1e-4 |
| Inference threshold | 0.15 (high sensitivity) |
| Sliding window stride | 16 voxels |

### Stage 2 — False Positive Reduction
| Parameter | Value |
|-----------|-------|
| Augmentation per TP | ×12 |
| Hard negatives per scan | 15 |
| Lung mask overlap threshold | 30% |
| Min match radius | 15 mm |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/LUNA16-Lung-Nodule-Detection.git
cd LUNA16-Lung-Nodule-Detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the LUNA16 dataset
Download subsets 0–9 from [Zenodo](https://zenodo.org/record/3723295) and place them under:
```
LUNA16_Project/
├── subset0/
├── subset1/
...
└── subset9/
```
Also download `annotations.csv` and `candidates.csv`.

### 4. Run the notebooks in order
```
1. EDA_and_Visualizationnn.ipynb        ← understand the data
2. LungMask_Generation.ipynb            ← precompute lung masks
3. DenseNet121_Architecture.ipynb       ← explore the model
4. LUNA16_Stage1_Completeee.ipynb       ← train detector
5. LUNA16_Stage2_Completeee.ipynb       ← train FPR model
6. Testing_Pipeline_Updated.ipynb       ← evaluate end-to-end
```

> 💡 All notebooks are designed to run on **Google Colab** with Google Drive mounted.

---

## 📦 Dataset

| Item | Details |
|------|---------|
| Dataset | [LUNA16](https://luna16.grand-challenge.org/) |
| CT scans | ~888 scans across 10 subsets |
| Annotations | 6,446 nodule annotations |
| Format | `.mhd` / `.raw` (MetaImage) |
| Source | [Zenodo](https://zenodo.org/record/3723295) |

---

## 🛠️ Tech Stack

- **PyTorch** — model training
- **MONAI** — medical imaging transforms & DenseNet-121
- **SimpleITK** — CT scan loading & resampling
- **lungmask** — pretrained lung segmentation
- **NumPy / SciPy** — preprocessing & morphological operations
- **scikit-learn** — metrics & evaluation

---

## 📄 License

This project is for research and educational purposes.  
The LUNA16 dataset is subject to its own [terms of use](https://luna16.grand-challenge.org/Data/).

---

## 📚 References

- Huang et al. (2017) — [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- [LUNA16 Grand Challenge](https://luna16.grand-challenge.org/)
- [MONAI Framework](https://monai.io/)
