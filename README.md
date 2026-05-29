# Aggregate Shape Classification using Deep Learning

**Final Year Project — Department of Computer Engineering, University of Jaffna, Sri Lanka**

Deep learning pipeline for classifying construction aggregate shapes produced at different milling revolution speeds, using transfer learning CNNs, custom CNN architectures, and a hybrid CNN + SVM approach — implemented in PyTorch.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Authors](#authors)

---

## Problem Statement

Construction aggregates are processed at different milling revolution speeds — 0, 100, 500, 1000, 1500, and 2000 RPM — each producing a distinct particle shape and surface texture. As milling intensity increases, particles progressively become rounder and smoother, which directly affects the mechanical behaviour of concrete mixtures.

This project builds a **six-class automated image classification system** that identifies the milling class of an aggregate particle from a single photograph, replacing manual and time-consuming sieve-based analysis with a fast, camera-based deep learning approach.

**Classes:** `0 RPM` · `100 RPM` · `500 RPM` · `1000 RPM` · `1500 RPM` · `2000 RPM`

---

## Dataset

| Property | Detail |
|---|---|
| Number of classes | 6 (0, 100, 500, 1000, 1500, 2000 RPM) |
| Images per class | 500 |
| Total images | 3,000 |
| Image type | RGB photographs of individual aggregate particles |
| Preprocessing | CLAHE enhancement + binary segmentation + white background isolation |
| Input resolution | 224 × 224 px (resized for all models) |

The dataset is not included in this repository due to size. Contact the authors for access.

---

## Preprocessing Pipeline

All images were preprocessed **offline** using **Contrast Limited Adaptive Histogram Equalization (CLAHE)** before dataset construction. CLAHE enhances local contrast while suppressing noise amplification — essential for aggregate images with non-uniform illumination, irregular particle boundaries and varying surface textures.

**Steps applied to every image:**

1. Convert RGB image to grayscale
2. Apply CLAHE — clip limit `2.0`, tile grid `8 × 8` (64 tiles per image), via OpenCV `createCLAHE`
3. Binary thresholding to separate particle from background
4. Morphological opening to remove small noise regions
5. Largest-contour extraction to isolate the single aggregate particle
6. Replace background with uniform white fill

This pipeline ensures consistent illumination, enhanced surface texture contrast, and clean background separation across all six milling classes.

---

## Model Architectures

### 1. Pretrained CNNs — Transfer Learning

Four pretrained architectures were fine-tuned for the six-class aggregate classification task. All models were initialised with **ImageNet pretrained weights** and extended with a shared custom classification head:

```
Backbone (frozen initial layers)
    → Dropout(p=0.3)
    → Linear(in_features → 128)   # 128-dimensional embedding layer
    → ReLU
    → Linear(128 → 6)             # six-class output
```

| Model | Backbone Parameters | Model Size |
|---|---|---|
| ResNet50 | 23.8 M | ~91 MB |
| EfficientNetB0 | 4.2 M | ~17 MB |
| MobileNetV2 | 2.4 M | ~9 MB |
| Xception | 21.1 M | ~81 MB |

**Training configuration (all pretrained models):**

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1 × 10⁻⁵ |
| Batch size | 32 |
| Early stopping patience | 5 epochs (min 20 epochs) |
| Best-weight restoration | Lowest validation loss |
| Augmentation | Horizontal flip (p=0.5), rotation ±20°, zoom 0.8–1.2 |
| Normalisation | ImageNet mean/std |

---

### 2. Custom CNNs — Trained from Scratch

Seven custom CNN architectures (CNN2 through CNN8) were designed and trained from scratch to investigate how network depth and parameter count affect classification performance on this domain-specific task.

All custom CNNs follow this general pattern:

```
Input (3, 224, 224)
    → [Conv2d(3×3, padding=1) → ReLU → MaxPool2d(2×2)] × N layers
    → AdaptiveAvgPool2d(1×1)
    → Linear(→ 512)   # embedding
    → Dropout(0.5)
    → Linear(512 → 6) # classifier
```

| Model | Conv Layers | Channel Progression | Parameters | Model Size |
|---|---|---|---|---|
| CNN2 | 2 | 3→32→64 | ~115 K | ~0.5 MB |
| CNN3 | 3 | 3→32→64→128 | ~436 K | ~1.8 MB |
| CNN4 | 4 | 3→32→64→128→256 | ~1.8 M | ~7 MB |
| CNN5 | 5 | 3→32→64→128→256→512 | ~6.5 M | ~26 MB |
| **CNN6** | **6** | **3→32→64→128→256→512→1024** | **~6.8 M** | **~27 MB** |
| CNN7 | 7 | 3→32→64→128→256→512→512→1024 | ~101 M | ~101 MB |
| CNN8 | 8 | deeper extension of CNN7 | ~245 M | ~245 MB |

CNN6 achieved the best balance of accuracy and model size among custom architectures.

---

### 3. Hybrid CNN + SVM

CNN6 was extended as a feature extractor for a hybrid pipeline:

```
Image
  → CNN6 backbone → 512-d embedding (from penultimate FC layer)
  → mRMR feature selection → 96 features selected from 512
  → StandardScaler
  → RBF-kernel SVM (C=10, gamma='scale')
  → Class prediction
```

mRMR (Minimum Redundancy Maximum Relevance) greedily selects features that maximise mutual information with the class label while minimising redundancy among selected features. 96 features were selected from the 512-dimensional embedding.

---

## Results

### Six-Class Classification (All Milling Levels)

| Model | Overall Accuracy | Macro F1 |
|---|---|---|
| **ResNet50** | **85.67%** | — |
| MobileNetV2 | 84.50% | — |
| Xception | 84.37% | — |
| EfficientNetB0 | 83.33% | — |
| CNN6 + SVM (mRMR-96) | 82.33% | — |
| CNN6 (plain) | 82.20% | — |
| CNN7 | 81.13% | — |
| CNN5 | 73.53% | — |
| CNN4 | 68.40% | — |
| CNN8 | 67.07% | — |
| CNN3 | 62.60% | — |
| CNN2 | 61.27% | — |

ResNet50 achieved the highest overall accuracy at **85.67%** on the full 3,000-image test set.

---

### Subclass Experiments — ResNet50

When the six-class problem was reduced to targeted four-class and three-class subsets, accuracy improved significantly, demonstrating that morphologically similar intermediate classes (500–1500 RPM) are the primary source of confusion.

| Experiment | Classes | Test Samples | Accuracy |
|---|---|---|---|
| 4-class | 0, 100, 1000, 2000 RPM | 2,000 | **97.50%** |
| 4-class | 0, 100, 1500, 2000 RPM | 2,000 | **97.50%** |
| 4-class | 0, 100, 500, 2000 RPM | 2,000 | 95.35% |
| 3-class | 500, 1000, 1500 RPM | 1,500 | 78.00% |

---

## Project Structure

```
aggregate-shape-classification/
│
├── preprocessing/
│   ├── clahe_pipeline.ipynb              # CLAHE preprocessing walkthrough
│   ├── basic_he_illustration.ipynb       # Histogram equalization illustration
│   ├── clahe_schematic.ipynb             # CLAHE schematic diagram generation
│   ├── contrast_stretching_diagram.ipynb # Contrast stretching visualization
│   ├── he_local_vs_global.ipynb          # HE local vs global comparison
│   └── histogram_figure.ipynb            # Histogram type figure generation
│
├── pretrained_models/
│   ├── ResNet50.ipynb                    # ResNet50 fine-tuning (best: 85.67%)
│   ├── EfficientNetB0.ipynb              # EfficientNetB0 fine-tuning (83.33%)
│   ├── MobileNetV2.ipynb                 # MobileNetV2 fine-tuning (84.50%)
│   └── Xception.ipynb                    # Xception fine-tuning (84.37%)
│
├── custom_cnn/
│   ├── CNN2.ipynb                        # 2-layer custom CNN (61.27%)
│   ├── CNN3.ipynb                        # 3-layer custom CNN (62.60%)
│   ├── CNN4.ipynb                        # 4-layer custom CNN (68.40%)
│   ├── CNN5.ipynb                        # 5-layer custom CNN (73.53%)
│   ├── CNN6.ipynb                        # 6-layer custom CNN — best (82.20%)
│   ├── CNN7.ipynb                        # 7-layer custom CNN (81.13%)
│   ├── CNN8.ipynb                        # 8-layer custom CNN (67.07%)
│   └── Hybrid.ipynb                     # CNN6 + mRMR + SVM (82.33%)
│
├── subclass_experiments/
│   ├── ResNet50_4Class_0_100_500_2000.ipynb    # 4-class: 95.35%
│   ├── ResNet50_4Class_0_100_1000_2000.ipynb   # 4-class: 97.50%
│   ├── ResNet50_4Class_0_100_1500_2000.ipynb   # 4-class: 97.50%
│   └── ResNet50_3Class_500_1000_1500.ipynb     # 3-class: 78.00%
│
├── analysis/
│   ├── ROC_PR_All_Models.py              # ROC and PR curve generation for all models
│   └── ResNet50_Probability_BarCharts.ipynb  # Per-class probability analysis
│
└── requirements.txt
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Jenarththan2001/aggregate-shape-classification.git
cd aggregate-shape-classification

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

Each notebook is self-contained. Open in Jupyter Lab or Jupyter Notebook and update the dataset path variable at the top of the notebook before running:

```python
DATASET_DIR = "/path/to/your/dataset"   # update this to your local path
```

**Recommended run order:**

1. `preprocessing/clahe_pipeline.ipynb` — understand the preprocessing before training
2. `pretrained_models/ResNet50.ipynb` — start with the best pretrained model
3. `pretrained_models/EfficientNetB0.ipynb`, `MobileNetV2.ipynb`, `Xception.ipynb`
4. `custom_cnn/CNN6.ipynb` — best custom architecture
5. `custom_cnn/Hybrid.ipynb` — CNN + mRMR + SVM pipeline
6. `subclass_experiments/` — explore targeted class subsets
7. `analysis/ROC_PR_All_Models.py` — generate final evaluation plots

---

## Requirements

| Package | Version | Purpose |
|---|---|---|
| torch | ≥ 2.0.0 | Model training and inference |
| torchvision | ≥ 0.15.0 | Pretrained models and transforms |
| timm | ≥ 0.9.0 | Xception architecture |
| scikit-learn | ≥ 1.3.0 | SVM classifier and mRMR selection |
| joblib | ≥ 1.3.0 | Model serialisation |
| opencv-python | ≥ 4.8.0 | CLAHE preprocessing |
| matplotlib | ≥ 3.7.0 | Training curves and plots |
| seaborn | ≥ 0.12.0 | Confusion matrix visualisation |
| Pillow | ≥ 10.0.0 | Image loading |
| numpy | ≥ 1.24.0 | Numerical operations |
| pandas | ≥ 2.0.0 | Results tabulation |
| tqdm | ≥ 4.65.0 | Training progress bars |

---

## Authors

**A. Jenarththan · S. Nathiskar · V. Aarthy · P. Jeyananthan**
Department of Computer Engineering, University of Jaffna, Sri Lanka

**Supervisor: D. N. Subramaniam**
Department of Civil Engineering, University of Jaffna, Sri Lanka
