# Dry Bean Classification with PyTorch

A modular, tabular deep-learning project that classifies dry beans into **seven varieties** using **16 numerical shape features**. The workflow covers exploratory data analysis (EDA), preprocessing, PyTorch training, evaluation, error analysis, and reusable single-observation inference.

> **Scope:** This model works with numerical features extracted from bean images. It does **not** classify raw image files directly and does not use a convolutional neural network (CNN).

## Results

The final notebook's **latest saved execution** reports:

| Metric | Value |
| --- | ---: |
| Test macro F1-score | **0.9333** |
| Test cross-entropy loss | **0.2255** |
| Best validation macro F1-score | **0.9359** |
| Best checkpoint epoch | **10** |
| Test-set size | **2,032** |

Per-class test scores from the same notebook execution:

| Class | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| BARBUNYA | 0.9516 | 0.8939 | 0.9219 |
| BOMBAY | 1.0000 | 1.0000 | 1.0000 |
| CALI | 0.9388 | 0.9388 | 0.9388 |
| DERMASON | 0.9349 | 0.8910 | 0.9124 |
| HOROZ | 0.9406 | 0.9642 | 0.9522 |
| SEKER | 0.9628 | 0.9375 | 0.9500 |
| SIRA | 0.8203 | 0.8990 | 0.8578 |

**Interpretation:** BOMBAY achieved an F1-score of 1.0 on this test split, while SIRA had the lowest F1-score. A perfect result on one class in one test set is not a guarantee of perfect performance on future data. The notebook includes a confusion-matrix heatmap and a per-class F1 bar chart for further error analysis.

> **Notebook consistency note:** Some handwritten analysis cells still refer to *earlier* training/evaluation results (such as 0.9323 test F1 and 0.9381 validation F1). The tables above intentionally use the **latest saved execution outputs**. Update those prose cells to match the current run before publishing.

## Dataset

- **Source:** UCI Machine Learning Repository — [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- **UCI dataset ID:** `602` (loaded using `ucimlrepo`)
- **Original observations:** 13,611
- **Input features:** 16 numerical morphological measurements
- **Target classes:** BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA
- **Cleaning:** The notebook reports 68 duplicate observations and no missing values.
- **Split after deduplication:** 9,480 training / 2,031 validation / 2,032 test observations (~70% / 15% / 15%), stratified by target class.

The 16 input features, in training order, are:

`Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRatio`, `Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`, `Solidity`, `Roundness`, `Compactness`, `ShapeFactor1`, `ShapeFactor2`, `ShapeFactor3`, `ShapeFactor4`.

## Approach

1. **EDA:** Inspect class frequencies, missing values, duplicates, feature distributions, and correlations.
2. **Preprocessing:** Remove duplicates, encode class labels, create stratified train/validation/test splits, and fit `StandardScaler` **on training data only**.
3. **Data loading:** Build a custom `torch.utils.data.Dataset` and `DataLoader` for minibatch training.
4. **Model:** A fully connected neural network with architecture **16 → 64 → 32 → 7**, hidden-layer batch normalization, ReLU, and dropout (`p=0.2`). The model outputs logits.
5. **Training:** Cross-entropy loss, Adam optimizer, and early stopping monitored using validation macro F1; save the best checkpoint.
6. **Evaluation:** Report independent test loss and macro F1, per-class precision/recall/F1, and confusion matrix.
7. **Inference:** Reload the checkpoint, fitted scaler, label encoder, and ordered feature names to classify one new observation without retraining.

## Project layout

Arrange the repository as follows (some files are generated when you run the notebook):

```text
.
├── main.ipynb
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/                        # generated checkpoints/preprocessing artifacts
│   ├── best_model.pth
│   ├── scaler.joblib
│   ├── encoder.joblib
│   └── features_names.joblib
└── reports/                       # generated evaluation outputs
    ├── per_class_metrics.csv
    ├── confusion_matrix.png
    └── perclass_f1.png
```

**Important:** `src/*.py` modules are required to run the notebook; the notebook alone is not a standalone implementation. Include the **final matching versions** of those modules when publishing. The above `models/` and `reports/` filenames match the paths used in the submitted notebook.

## Installation

Use Python 3.11 (the notebook's recorded environment) or another compatible Python version.

```bash
# From the repository root
python -m venv .venv
```

**Windows (Command Prompt):**

```bat
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

**macOS / Linux:**

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Open `main.ipynb` and run the cells in sequence. The first data-loading step uses `fetch_ucirepo(id=602)` and requires internet access on its first fetch. The notebook contains a `%pip install ucimlrepo` cell; that installation is already covered by `requirements.txt` and may be skipped after setup.

PyTorch runs on CPU, so an NVIDIA GPU is **not required**. For CUDA-specific installation, consult the [official PyTorch instructions](https://pytorch.org/get-started/locally/).

## Reusable inference

Once the trained checkpoint and preprocessing files are present in `models/`, the `src.predict.predict()` function accepts either a **one-row pandas DataFrame with named columns** or **16 unnamed numerical values already in training feature order**.

Example (using a *raw*, unstandardized observation fetched from UCI):

```python
from ucimlrepo import fetch_ucirepo
from src.predict import predict

beans = fetch_ucirepo(id=602)
raw_sample = beans.data.features.iloc[[0]]

predicted_class, class_index, probabilities = predict(
    X=raw_sample,
    scaler="models/scaler.joblib",
    encoder="models/encoder.joblib",
    checkpoint="models/best_model.pth",
    features="models/features_names.joblib",
    device="cpu",
)

print("Predicted class:", predicted_class)
print("Class index:", class_index)
print("Class probabilities:", probabilities)
```

For named DataFrames, inference checks for missing/extra/duplicate feature names and reorders columns to the saved training order. For unnamed arrays or lists, it can only verify feature count; the caller must ensure correct ordering. The input must contain **exactly one observation with 16 features**. Do **not** pass an already standardized sample to this function, because it applies the saved scaler itself.

Only load `.joblib` artifacts and model files from trusted sources.

## Outputs and reproducibility

- `models/best_model.pth`: trained model state and architecture metadata
- `models/scaler.joblib`: scaler fitted on the training split
- `models/encoder.joblib`: fitted label encoder
- `models/features_names.joblib`: ordered input feature names
- `reports/per_class_metrics.csv`: per-class test metrics
- `reports/confusion_matrix.png`: labeled confusion matrix
- `reports/perclass_f1.png`: F1-score comparison by class

The notebook demonstrates single-observation inference, a successful check of prediction invariance to DataFrame column reordering, and a rejected 15-feature input. A fully independent **fresh-process inference test** should also be run before claiming end-to-end reproducibility.

**Before publishing:** Restart the notebook kernel and run all cells in order, confirm the final `src/` modules and checkpoint/preprocessing artifacts agree, and refresh the stale narrative metrics in the notebook.

## Limitations

- Predicts from 16 pre-extracted measurements, not image pixels.
- Some varieties, notably SIRA, are harder to distinguish with the available features.
- Observed test performance depends on the supplied dataset and split; deployment to other acquisition conditions is not established.
- Unnamed input arrays cannot be checked for semantic feature order.

## Acknowledgment

Dataset: [Dry Bean Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset). Please consult the dataset page for its original authors, licensing, and citation details before redistribution.
