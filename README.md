# Chromosome Classification

**Semi-supervised chromosome classification (classes 1–22, X, Y)** using PCA and blob-based features. Pipeline for training and inference from src/ or scripts/; scikit-learn, OpenCV.

---

## Problem Statement

- **Real-world problem**: Cytogenetics labs need to identify and group chromosomes (1–22, X, Y) from microscopy images. Manual sorting is time-consuming; semi-supervised or feature-based methods can assist when labeled data is limited.
- **Why it matters**: Supports karyotyping and genetic diagnosis; PCA and blob features provide interpretable, low-dimensional representation.
- **Constraints**: Image data and possible limited labels; feature extraction (blob, shape) and PCA; users are researchers or lab technicians.

---

## System Architecture

```
Raw karyotype sheet images
    → Preprocessing pipeline (`src/preprocessing/`, unified via `src/pipeline.py`)
    → Cropped chromosome images (folders 1–22, X, Y)
    → Feature extraction / PCA
    → Classifier (e.g. sklearn / semi-supervised)
    → Class label (1–22, X, Y)
```

- **No web/API by default**: Run training and inference from src/ or scripts/ as needed.
- **Stack**: Python, scikit-learn, OpenCV; data and models on disk.

---

## Key Features

### AI Features

- **Feature extraction**: Blob and shape features from chromosome images (OpenCV).
- **Dimensionality reduction**: PCA for compact representation.
- **Model**: Semi-supervised or supervised classifier (see src/ or scripts/) for 24 classes (1–22, X, Y).
- **Evaluation**: Accuracy or per-class metrics (implement in training script).

### Application Features

- **CLI/scripts**: Training and inference as per project scripts; no built-in UI.

### Engineering Features

- **Modular**: src/ or scripts/ for pipeline steps; requirements.txt for deps.

---

## Model & Methodology

- **Features**: Blob and shape descriptors (OpenCV); PCA for reduction.
- **Algorithm**: Classifier (e.g. SVM, Random Forest, or semi-supervised method) on feature vector.
- **Evaluation**: Accuracy, per-class precision/recall (run training to generate).

---

## Results

Dataset: 42,722 chromosome images (24 classes: 1–22, X, Y)

| Model              | Accuracy | F1 (weighted) | F1 (macro) |
|--------------------|----------|---------------|------------|
| Supervised         | 89.90%   | 89.92%        | 89.29%     |
| Semi-supervised    | 89.44%   | 89.43%        | —          |

The supervised model slightly outperforms the semi-supervised approach by ~0.5%.

### Evaluation outputs (after running `scripts/evaluate.py`)

Generated under `results/`:

- **Confusion matrix**: `confusion_matrix_supervised.png`, `confusion_matrix_semi_supervised.png`
- **Per-class performance**: `per_class_performance_supervised.png`, `per_class_performance_semi_supervised.png`
- **Training curves**: `training_curves.png` (from `train.py`)
- **PCA variance**: `pca_variance.png`
- **Metrics**: `comparison.json`

Example placement in docs: you can add a `results/examples/` folder and commit sample plots for the README, e.g.:

| Confusion matrix (supervised) | Per-class performance (supervised) |
|-------------------------------|------------------------------------|
| ![Confusion matrix](results/examples/confusion_matrix_supervised.png) | ![Per-class](results/examples/per_class_performance_supervised.png) |

*(Create `results/examples/` and copy the PNGs there if you want these images in the repo; `results/` is gitignored by default.)*

---

## Project Structure

```
chromosome-classification/
├── data/                    # Training data: 1–22, X, Y, unlabeled/
├── models/                  # Saved PCA + classifier weights
├── results/                 # Plots, metrics, evaluation artifacts
├── scripts/
│   ├── train.py             # Training entrypoint
│   ├── evaluate.py          # Evaluation entrypoint
│   └── infer.py             # Inference on new images
├── src/
│   ├── config.py            # Global paths + hyperparameters
│   ├── pipeline.py          # Unified preprocessing wrapper (from raw sheets → cropped chromosomes)
│   ├── preprocessing/       # Ported from `cv-karyotype-preprocessing`
│   │   ├── part1_preproc.py
│   │   ├── part2_blobs.py
│   │   ├── part3_map_and_crop.py
│   │   ├── blob_processing_utils.py
│   │   ├── preprocessing_utils.py
│   │   ├── utils.py
│   │   └── config.py
│   ├── datasets/            # Dataset utilities for classification
│   ├── features/            # Feature extraction (blob, PCA, etc.)
│   ├── models/              # Model definitions
│   ├── training/            # Training loop / semi-supervised logic
│   ├── evaluation/          # Evaluation helpers
│   ├── inference/           # Inference helpers
│   └── utils/               # Shared utilities
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

Typical: scikit-learn, OpenCV, numpy, pandas. Prepare chromosome image data as expected by the pipeline.

---

## Usage

### 1. Preprocess raw karyotype sheets (from former `cv-karyotype-preprocessing`)

```bash
# Option A: one-command wrapper
python scripts/run_pipeline.py

# Option B: more control via module CLI
python -m src.pipeline --input data/raw --out data/preprocessed
```

This will:

- Run segmentation, blob extraction, and grid mapping from the old project.
- Produce cropped chromosome images grouped into subfolders (1–22, X, Y) under `data/preprocessed/` (or a custom `--out`).

### 2. Train chromosome classifier

Move/link the cropped chromosome folders from `data/preprocessed/` vào `data/` đúng cấu trúc (xem bên dưới), rồi chạy:

```bash
python scripts/train.py
```

### 3. Evaluate / infer

```bash
python scripts/evaluate.py
```

**Single-image inference (demo):**

```bash
python scripts/infer.py --image sample.png
```

Example output:

```text
================================================================================
CHROMOSOME CLASSIFICATION - INFERENCE
================================================================================

Loading trained model and feature extractor...

Chromosome 7 (confidence 0.91)
```

**Batch inference** (directory of images):

```bash
python scripts/infer.py --dir data/unlabeled --output results/predictions.csv
```

---

## Dataset structure

Recommended layout for data used in this project:

```text
data/
 ├── raw/
 │    └── karyotype_sheets...           # ảnh tấm karyotype gốc
 │
 ├── preprocessed/
 │    └── (tự động tạo bởi preprocessing)
 │         ├── 1/
 │         ├── 2/
 │         ├── ...
 │         ├── X/
 │         └── Y/
 │
 ├── unlabeled/                         # (optional) ảnh chưa gán nhãn cho semi-supervised
 │    └── *.png / *.jpg
 │
 └── (optional) train/                  # nếu muốn tách riêng so với preprocessed
      ├── 1/
      ├── 2/
      ├── ...
      ├── X/
      └── Y/
```

By default, `src.config` dùng `data/` làm thư mục chứa các lớp 1–22, X, Y; bạn có thể:

- Dùng trực tiếp `data/preprocessed/` làm `data/`, hoặc
- Copy/symlink từ `data/preprocessed/` sang `data/` trước khi chạy `train.py`.

---

## Demo

Run `python scripts/infer.py --image sample.png` to see a single-image prediction (e.g. `Chromosome 7 (confidence 0.91)`). See **Results** above for accuracy and **Evaluation outputs** for generated plots.

---

## Deployment

- **Local/research**: Run on workstation or server; add REST API (e.g. FastAPI) if serving is needed.

---

## Future Improvements

- **CNN baseline**: ResNet18 or MobileNetV2 for comparison with classical ML (feature + PCA + MLP). *Future work: CNN baseline (ResNet18) for comparison with classical ML.*
- Full pipeline script (single command from raw images to labels).
- Export to standard karyotype format or report.

---

## For your CV / portfolio

Short bullet you can use (e.g. for AI internship applications):

**Chromosome Classification (Computer Vision)**

- Built an end-to-end pipeline for chromosome classification (24 classes: 1–22, X, Y) from microscopy images.
- Implemented preprocessing and segmentation using OpenCV blob detection.
- Extracted shape-based features and applied PCA for dimensionality reduction.
- Trained supervised and semi-supervised classifiers using PyTorch/scikit-learn.
- Achieved **89.9% accuracy** on a 42k-image dataset.
