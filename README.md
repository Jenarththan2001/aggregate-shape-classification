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

Construction aggregates are processed at different numbers of milling revolutions — 0, 100, 500, 1000, 1500, and 2000 — each producing a distinct particle shape and surface texture. As the number of milling revolutions increases, particles progressively become rounder and smoother, which directly affects the mechanical behaviour of concrete mixtures.

This project builds a **six-class automated image classification system** that identifies the milling class of an aggregate particle from a single photograph. Traditional shape-factor-based methods and Fourier analysis-based classification have shown limited accuracy in distinguishing within milled classes — this system addresses that gap by using CNNs to learn morphological features directly from images with higher multiclass classification accuracy.

**Classes:** `0` · `100` · `500` · `1000` · `1500` · `2000` revolutions

---

## Dataset

| Property | Detail |
|---|---|
| Number of classes | 6 (0, 100, 500, 1000, 1500, 2000 revolutions) |
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

| Model | Conv Layers | Channel Progression | Parameters | Size | CNN Acc | Best (+ SVM) |
|---|---|---|---|---|---|---|
| CNN2 | 2 | 3→32→64 | 115 K | 0.1 MB | 61.27% | 61.93% |
| CNN3 | 3 | 3→32→64→128 | 436 K | 0.5 MB | 62.60% | 64.03% |
| CNN4 | 4 | 3→32→64→128→256 | 1.8 M | 1.8 MB | 68.40% | 69.47% |
| CNN5 | 5 | 3→32→64→128→256→512 | 6.8 M | 6.8 MB | 73.53% | 75.67% |
| **CNN6** | **6** | **3→32→64→128→256→512→1024** | **6.8 M** | **27.3 MB** | **82.20%** | **82.33%** |
| CNN7 | 7 | 3→32→64→128→256→512→512→1024 | 101 M | 104.9 MB | 81.13% | 81.63% |
| CNN8 | 8 | deeper extension of CNN7 | 256 M | 256.0 MB | 67.07% | 73.63% |

CNN6 achieved the best accuracy among custom models. Performance peaks at 6 layers — CNN7 and CNN8 show degraded CNN-only accuracy without residual connections, though CNN8 benefits significantly from mRMR + SVM (+6.56%).

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

### Pretrained Models — 6-Class

| Model | Accuracy | Macro F1 | Parameters | Size |
|---|---|---|---|---|
| **ResNet50** | **85.67%** | **85.64%** | 23.8 M | 91.0 MB |
| MobileNetV2 | 84.50% | 84.43% | 2.4 M | 9.4 MB |
| Xception | 84.37% | 84.41% | 21.1 M | 80.7 MB |
| EfficientNetB0 | 83.33% | 83.49% | 4.2 M | 16.2 MB |

ResNet50 achieved the highest overall accuracy at **85.67%** on the full 3,000-image test set. MobileNetV2 is the most efficient — second-best accuracy at only 9.4 MB (10× smaller than ResNet50).

### Custom CNNs — 6-Class (CNN alone vs CNN + mRMR + SVM)

| Model | CNN Accuracy | CNN Macro F1 | +SVM Accuracy | +SVM Macro F1 | Size |
|---|---|---|---|---|---|
| **CNN6** | 82.20% | 82.24% | **82.33%** | **82.23%** | 27.3 MB |
| CNN7 | 81.13% | 80.90% | 81.63% | 81.59% | 104.9 MB |
| CNN5 | 73.53% | 73.54% | 75.67% | 75.66% | 6.8 MB |
| CNN8 | 67.07% | 65.67% | 73.63% | 73.68% | 256.0 MB |
| CNN4 | 68.40% | 68.13% | 69.47% | 69.48% | 1.8 MB |
| CNN3 | 62.60% | 61.78% | 64.03% | 62.94% | 0.5 MB |
| CNN2 | 61.27% | 58.94% | 61.93% | 60.22% | 0.1 MB |

CNN6 achieves the best custom-model accuracy. mRMR + SVM consistently improves over CNN-alone — the largest gain is CNN8 (+6.56%), which benefits most from dimensionality reduction via mRMR.

---

### Subclass Experiments — ResNet50

When the six-class problem was reduced to targeted four-class and three-class subsets, accuracy improved significantly, demonstrating that morphologically similar intermediate classes (500–1500 revolutions) are the primary source of confusion.

| Experiment | Classes | Test Samples | Accuracy |
|---|---|---|---|
| 4-class | 0, 100, 1000, 2000 revolutions | 2,000 | **97.50%** |
| 4-class | 0, 100, 1500, 2000 revolutions | 2,000 | **97.50%** |
| 4-class | 0, 100, 500, 2000 revolutions | 2,000 | 95.35% |
| 3-class | 500, 1000, 1500 revolutions | 1,500 | 78.00% |

---

## Project Structure

```
aggregate-shape-classification/
│
├── preprocessing/
│   ├── 01_Crop_Aggregates.ipynb          # Grid detection and individual specimen cropping
│   └── 02_CLAHE.ipynb                    # CLAHE enhancement + white background pipeline
│
├── pretrained_models/
│   ├── 01_ResNet50.ipynb                 # ResNet50 fine-tuning (best: 85.67%)
│   ├── 02_EfficientNetB0.ipynb           # EfficientNetB0 fine-tuning (83.33%)
│   ├── 03_MobileNetV2.ipynb              # MobileNetV2 fine-tuning (84.50%)
│   └── 04_Xception.ipynb                 # Xception fine-tuning (84.37%)
│
├── custom_cnn/
│   ├── 01_CNN2.ipynb                     # 2-layer custom CNN (61.27%)
│   ├── 02_CNN3.ipynb                     # 3-layer custom CNN (62.60%)
│   ├── 03_CNN4.ipynb                     # 4-layer custom CNN (68.40%)
│   ├── 04_CNN5.ipynb                     # 5-layer custom CNN (73.53%)
│   ├── 05_CNN6.ipynb                     # 6-layer custom CNN — best (82.20%)
│   ├── 06_CNN7.ipynb                     # 7-layer custom CNN (81.13%)
│   ├── 07_CNN8.ipynb                     # 8-layer custom CNN (67.07%)
│   └── 08_Hybrid_CNN_SVM.ipynb           # CNN6 + mRMR + SVM (82.33%)
│
├── subclass_experiments/
│   ├── 01_ResNet50_4Class_0_100_500_2000.ipynb    # 4-class subset: 95.35%
│   ├── 02_ResNet50_4Class_0_100_1000_2000.ipynb   # 4-class subset: 97.50%
│   ├── 03_ResNet50_4Class_0_100_1500_2000.ipynb   # 4-class subset: 97.50%
│   └── 04_ResNet50_3Class_500_1000_1500.ipynb     # 3-class subset: 78.00%
│
├── analysis/
│   ├── 01_ROC_PR_All_Models.py           # ROC and PR curve generation for all models
│   └── 02_ResNet50_Probability_BarCharts.ipynb  # Per-class probability analysis
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

1. `preprocessing/01_Crop_Aggregates.ipynb` — grid detection and specimen cropping
2. `preprocessing/02_CLAHE.ipynb` — CLAHE enhancement and background removal
3. `pretrained_models/01_ResNet50.ipynb` — best pretrained model (85.67%)
4. `pretrained_models/02_EfficientNetB0.ipynb`, `03_MobileNetV2.ipynb`, `04_Xception.ipynb`
5. `custom_cnn/05_CNN6.ipynb` — best custom architecture (82.20%)
6. `custom_cnn/08_Hybrid_CNN_SVM.ipynb` — CNN + mRMR + SVM pipeline
7. `subclass_experiments/` — targeted 4-class and 3-class experiments
8. `analysis/01_ROC_PR_All_Models.py` — generate final evaluation plots

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

**A. Jenarththan · S. Nathiskar**
Department of Computer Engineering, University of Jaffna, Sri Lanka

**Supervisor**
Dr. (Mrs.) P. Jeyananthan — Department of Computer Engineering, University of Jaffna, Sri Lanka

**Co-Supervisors**
- Prof. N. Sathiparan — Department of Civil Engineering, University of Jaffna, Sri Lanka
- Prof. D. N. Subramaniam — Department of Civil Engineering, University of Jaffna, Sri Lanka
- Miss. V. Aarthy — Department of Computer Engineering, University of Jaffna, Sri Lanka
