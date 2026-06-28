# Data Science Internship Project 2
## Supervised Learning (Fraud Detection Pipeline)
**DecodeLabs Internship 2026 - Amashi Fernando**

## Overview
This project builds a leak-free machine learning pipeline to detect
fraudulent credit card transactions in a highly imbalanced dataset.
The focus is on **algorithmic precision** — correctly handling class
imbalance, avoiding data leakage, and choosing evaluation metrics that
actually matter for fraud detection, rather than chasing accuracy.

## Objectives
- Handle extreme class imbalance using **SMOTE (Synthetic Minority Over-sampling)**
- Train and tune **classification models** (Logistic Regression, Random Forest) using Scikit-Learn and Imbalanced-learn
- Evaluate models using **Precision, Recall, F1-score, and ROC-AUC** instead of Accuracy
- Build a pipeline that is provably **free of data leakage**

## Dataset
**Source:** [Credit Card Fraud Detection — Kaggle (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
**Size:** 284,807 rows × 31 columns (283,726 rows after removing 1,081 duplicates)

| Column | Description |
|---|---|
| Time | Seconds elapsed between this transaction and the first transaction in the dataset |
| V1–V28 | Anonymized features obtained via PCA (original meaning withheld for confidentiality) |
| Amount | Transaction amount |
| Class | Target variable — 1 = Fraudulent, 0 = Legitimate |

## Methodology

### 1. Data Cleaning
Checked for missing values (**0 found**) and duplicate rows
(**1,081 found and removed**), leaving 283,726 clean transactions.

### 2. Train/Test Split — Before Any Preprocessing
The dataset was split **80/20** using a stratified split, preserving
the real-world 0.17% fraud ratio in both sets:
```
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
```
This split happens **before** any scaling or resampling, so the test
set remains an untouched, realistic blind exam.

### 3. Handling Class Imbalance — SMOTE
Applied **SMOTE (Synthetic Minority Over-sampling Technique)** inside
an `imblearn.pipeline.Pipeline`, ensuring resampling only ever touches
the training fold — never the test set, and never before the split

SMOTE interpolates between minority-class neighbors to generate new
synthetic fraud examples, rather than simply duplicating existing rows.

### 4. Model Pipelines
| Pipeline | Steps | Notes |
|---|---|---|
| Logistic Regression | `StandardScaler → SMOTE → LogisticRegression` | Scaler required — LR is sensitive to feature magnitude |
| Random Forest | `SMOTE → RandomForestClassifier` | No scaler needed — tree splits are scale-invariant |

### 5. Hyperparameter Tuning
- **Logistic Regression:** Tuned with `GridSearchCV` (5-fold stratified CV, scored on ROC-AUC) across `C`, `penalty`, and `smote__k_neighbors`.
- **Random Forest:** Trained with a single, sensible configuration (`n_estimators=100`, `max_depth=10`) instead of an exhaustive grid search, since full GridSearchCV with SMOTE on ~227,000 training rows was computationally expensive. This was a deliberate time-vs-tuning tradeoff rather than an oversight.

### 6. Threshold Sensitivity Analysis
Since Logistic Regression's default-threshold precision was very low,
decision thresholds from 0.3 to 0.9 were tested:

| Threshold | Precision | Recall | F1-Score | False Positives |
|---|---|---|---|---|
| 0.3 | 0.0266 | 0.8842 | 0.0516 | 3,075 |
| 0.5 | 0.0532 | 0.8737 | 0.1002 | 1,478 |
| 0.7 | 0.1016 | 0.8632 | 0.1818 | 725 |
| 0.9 | 0.2000 | 0.8421 | 0.3232 | 320 |

Raising the threshold to 0.9 more than tripled Logistic Regression's
F1-score while barely affecting recall — proving that ROC-AUC alone
cannot tell you which threshold to deploy with, since it stayed
identical (0.9634) across every threshold tested.

## Results
| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (tuned, threshold=0.5) | 0.0532 | 0.8737 | 0.1002 | 0.9634 |
| Random Forest (single fit, threshold=0.5) | 0.6016 | 0.8105 | 0.6906 | 0.9729 |

- **Final dataset:** 283,726 rows after deduplication, 0 missing values
- **Target variable:** `Class` (Fraud vs Legitimate)
- Random Forest outperforms Logistic Regression on F1-score and
  Precision at the default threshold, while both models achieve
  strong ROC-AUC (>0.96), confirming both rank fraud risk well even
  though their default-threshold usability differs sharply.

## Repository Structure
```
DecodeLabs-Task2/
├── DecodeLabs_Task2.ipynb                   # Main analysis notebook
├── Dataset-creditcard.csv                   # Raw input dataset
├── Visualizations
   ├── 01_class_distribution.png             # EDA visual
   ├── 02_amount_distribution.png            # EDA visual
   ├── 03_correlation_heatmap.png            # EDA visual
   ├── 04_time_distribution.png              # EDA visual
   ├── 05_confusion_matrices.png             # Model evaluation visual
   ├── 06_roc_curves.png                     # Model evaluation visual
   ├── 07_precision_recall_curves.png        # Model evaluation visual
   ├── 08_feature_importance.png             # Model evaluation visual
└── model_comparison_summary.csv             # Final metrics table
README.md
```

## How to Run

**Option 1 — Google Colab**
1. Open `DecodeLabs_Task2.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Upload `Dataset-creditcard.csv` to the Colab session
   (or mount Google Drive if stored there)
3. Run all cells: **Runtime → Run all**

Required libraries (pre-installed on Colab, except imbalanced-learn):
```
pandas, numpy, matplotlib, seaborn, scikit-learn
```
```bash
pip install imbalanced-learn
```

**Option 2 — VS Code**
1. Open the project folder in VS Code
2. Install the **Jupyter** extension (if not already installed)
3. Open `DecodeLabs_Task2.ipynb` and select your Python interpreter (e.g. Anaconda)
4. Make sure the dataset file is in the same folder as the notebook
5. Run all cells using the ▶️ **Run All** button at the top of the notebook

Install dependencies (if not using Anaconda's base environment):
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

## Tech Stack
- Python (Pandas, NumPy)
- Scikit-Learn, Imbalanced-learn (SMOTE, Pipeline)
- Matplotlib, Seaborn
- Statistical/ML methods: SMOTE, Stratified K-Fold CV, GridSearchCV, threshold tuning

## Author
Amashi Fernando — DecodeLabs Data Science Internship, June 10 - July 10 2026
