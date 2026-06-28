# Credit Card Fraud Detection — Leak-Free SMOTE Pipeline

A supervised learning project that detects fraudulent credit card transactions in a highly imbalanced dataset, built as part of the DecodeLabs Data Science Internship (Project 2).

## Problem

Credit card fraud datasets are extremely imbalanced — in this dataset, only **0.17%** of transactions are fraudulent. A naive model that predicts "legitimate" for every transaction would score over **99.8% accuracy** while catching **zero fraud**. This project builds a pipeline that avoids that trap entirely by ditching accuracy as a metric and handling class imbalance correctly.

## Dataset

- **Source:** [Credit Card Fraud Detection — Kaggle (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Rows:** 284,807 transactions (283,726 after removing 1,081 exact duplicates)
- **Features:** `Time`, `V1`–`V28` (anonymized PCA components), `Amount`
- **Target:** `Class` (1 = fraud, 0 = legitimate)
- **Class balance:** 473 frauds (0.17%) vs. 283,253 legitimate transactions

## Project Goals

- Handle extreme class imbalance correctly using **SMOTE** (Synthetic Minority Over-sampling)
- Train and compare **Logistic Regression** and **Random Forest** classifiers
- Avoid **data leakage** by keeping resampling and scaling strictly inside the training fold
- Evaluate using **Precision, Recall, F1-score, and ROC-AUC** — never raw accuracy

## Key Techniques

| Concept | How it's handled |
|---|---|
| Class imbalance | SMOTE inside an `imblearn.pipeline.Pipeline` |
| Data leakage | Train/test split performed **before** any scaling or SMOTE |
| Feature scaling | `StandardScaler` used for Logistic Regression only (Random Forest is scale-invariant) |
| Hyperparameter tuning | `GridSearchCV` (5-fold stratified CV, scored on ROC-AUC) for Logistic Regression |
| Random Forest tuning | A single fit with sensible defaults (`n_estimators=100`, `max_depth=10`) was used instead of an exhaustive grid search, since SMOTE + Random Forest on ~227k training rows made full GridSearchCV computationally expensive. This was a deliberate tradeoff between tuning depth and training time. |
| Evaluation | Precision, Recall, F1, ROC-AUC, confusion matrix, ROC & Precision-Recall curves, feature importance |

## Tech Stack

- Python 3
- pandas, numpy
- scikit-learn
- imbalanced-learn (`imblearn`)
- matplotlib, seaborn

## How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
   ```

3. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root.

4. Open and run `DecodeLabs_Task2.ipynb` top to bottom (Kernel → Restart & Run All recommended).

The notebook prints EDA summaries and model metrics inline, and saves the following files:

- `01_class_distribution.png`
- `02_amount_distribution.png`
- `03_correlation_heatmap.png`
- `04_time_distribution.png`
- `05_confusion_matrices.png`
- `06_roc_curves.png`
- `07_precision_recall_curves.png`
- `08_feature_importance.png`
- `model_comparison_summary.csv`

## Results

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (tuned, threshold=0.5) | 0.0532 | 0.8737 | 0.1002 | 0.9634 |
| Random Forest (single fit, threshold=0.5) | 0.6016 | 0.8105 | 0.6906 | 0.9729 |

**Interpretation:** Both models achieve strong ROC-AUC (>0.96), showing both can separate fraud from legitimate transactions well in terms of *ranking*. However, at the default 0.5 decision threshold, Logistic Regression's precision (0.053) is very low — it generates many false positives (1,478) for every 83 frauds it correctly catches. Random Forest performs much better in practice, with far fewer false positives (51) and a noticeably higher F1-score (0.69 vs 0.10). This highlights an important lesson: a high ROC-AUC alone doesn't guarantee a usable model — the precision/recall tradeoff at the chosen threshold matters just as much, especially in a fraud context where false positives mean blocking real customers.

### Threshold Sensitivity Analysis (Logistic Regression)

Since Logistic Regression's default-threshold precision was so low, decision thresholds from 0.3 to 0.9 were tested to see if a better operating point exists:

| Threshold | Precision | Recall | F1-Score | False Positives |
|---|---|---|---|---|
| 0.3 | 0.0266 | 0.8842 | 0.0516 | 3,075 |
| 0.5 | 0.0532 | 0.8737 | 0.1002 | 1,478 |
| 0.7 | 0.1016 | 0.8632 | 0.1818 | 725 |
| 0.9 | 0.2000 | 0.8421 | 0.3232 | 320 |

Raising the threshold from 0.5 to 0.9 more than tripled the F1-score (0.100 → 0.323) and cut false positives by over 75%, while recall dropped only slightly (0.874 → 0.842). ROC-AUC stayed identical (0.9634) across every threshold, since ROC-AUC measures ranking ability across *all* thresholds at once — it does not indicate which threshold to actually use in production. Even at its best threshold, Logistic Regression's F1-score (0.323) still did not match Random Forest's single-fit F1-score (0.691), confirming Random Forest as the stronger model for this task. This analysis demonstrates that threshold selection is a necessary tuning step before deploying a probabilistic classifier like Logistic Regression in a real fraud detection system.

## What I Learned

- Why accuracy is a misleading metric for imbalanced classification problems
- How SMOTE generates synthetic minority-class samples through interpolation rather than duplication
- How to prevent data leakage by isolating resampling and scaling inside the pipeline, applied only to training data
- How ROC-AUC and the Precision/Recall tradeoff can tell different stories about the same model, and why threshold choice matters in a real fraud-detection context
- That ROC-AUC is threshold-independent, so it cannot tell you which decision threshold to deploy with — only F1-score (or a business cost analysis) at specific thresholds can reveal the right operating point
- How to make and justify practical tradeoffs (e.g. skipping exhaustive Random Forest tuning) based on computational cost versus benefit

