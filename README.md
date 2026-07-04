# Customer Churn Prediction

Predicting which bank customers are likely to leave (churn), using both classical
ML models and a tuned neural network. The goal isn't just to hit a high accuracy
number — for churn, the model that matters most is the one that actually catches
the customers who are about to leave, so recall on the churn class gets called
out explicitly below rather than buried in an overall accuracy figure.

**Dataset**: [Bank Customer Churn Modeling](https://www.kaggle.com/datasets/barelydedicated/bank-customer-churn-modeling)
(Kaggle) — 10,000 customers, 14 features, ~20% churn rate.

## Approach

1. **EDA** — checked for nulls (none), reviewed distributions of credit score, age,
   balance, tenure, and the categorical splits (geography, gender, active member status).
2. **Preprocessing** — dropped identifier columns (`RowNumber`, `CustomerId`, `Surname`),
   label-encoded `Gender`, one-hot encoded `Geography`, min-max scaled the numeric features.
3. **Class imbalance** — the raw data is ~80/20 (stayed/churned). Classical models were
   trained on the raw split; the ANN was additionally trained with SMOTE applied to the
   **training split only** (not the full dataset, to avoid leaking synthetic samples into
   the test set).
4. **Models** — Logistic Regression (baseline), Random Forest, Gradient Boosting, XGBoost,
   and an ANN (MLP with batch normalization), the last tuned via Keras Tuner (Hyperband)
   over layer sizes and learning rate.
5. **Evaluation** — accuracy, precision/recall/F1 per class, Cohen's Kappa, and ROC-AUC,
   since accuracy alone is a poor signal on this imbalanced a target.

## Results

**Classical ML models** — test set: 2,000 samples, 393 churned (~19.7%), class 1 = churned:

| Model | Accuracy | Precision (churn) | Recall (churn) | F1 (churn) | Cohen's Kappa |
|---|---|---|---|---|---|
| Logistic Regression | 0.811 | 0.56 | 0.19 | 0.28 | 0.205 |
| Random Forest | 0.866 | 0.76 | 0.47 | 0.58 | 0.504 |
| Gradient Boosting | 0.864 | 0.74 | 0.47 | 0.58 | 0.501 |
| **XGBoost** | **0.870** | 0.76 | 0.50 | 0.60 | **0.528** |

**ANN (MLP + BatchNorm)** — test set: 2,000 samples, 407 churned (~20.4%), a genuine,
untouched hold-out split (no leakage):

| Stage | Accuracy | Precision (churn) | Recall (churn) | F1 (churn) | Kappa | AUC |
|---|---|---|---|---|---|---|
| Baseline (32-16-8, no SMOTE) | 0.857 | 0.696 | 0.484 | 0.571 | 0.488 | *pending re-run* |
| Rebuilt architecture (no SMOTE) | 0.867 | 0.798 | 0.464 | 0.587 | 0.514 | *pending re-run* |
| Deeper rebuild (no SMOTE) | 0.833 | 0.595 | 0.555 | 0.574 | 0.470 | *pending re-run* |
| **Tuned (SMOTE-train + Keras Tuner Hyperband)** | 0.820 | 0.54 | **0.69** | 0.61 | 0.494 | **0.837** |

*AUC was only computed for the tuned model in the original notebook; ROC-AUC code has
now been added after each of the first three stages too (mirroring the tuned model's
evaluation), so those values will populate the next time the notebook is run.*

Best hyperparameters found (Hyperband): `neurons1=32, neurons2=64, neurons3=32, learning_rate=0.005`.

**Takeaway**: XGBoost gives the best overall balance of accuracy and Kappa among the
classical models. The tuned ANN trades about 4-5 points of raw accuracy for a large
jump in churn recall (0.69 vs. 0.46-0.48 for the non-SMOTE models) — for a churn
use case, that's the more useful model in practice, since missing an actual churner
is usually far more costly than a false alarm. Across every model here, though,
recall on the churn class tops out around 0.5-0.7, so there's real room for further
work: threshold tuning, a cost-sensitive objective, or additional feature engineering
would all be reasonable next steps before treating any of these as production-ready.

## Repo structure

```
Customer_Churn_Prediction_with_ANN_MLP.ipynb   # EDA + ANN (BatchNorm, SMOTE, hyperparameter tuning)
Customer_Churn_with_ML_Algos.ipynb             # Logistic Regression, Random Forest, Gradient Boosting, XGBoost
```

## Requirements

```
pandas
numpy
scikit-learn
tensorflow
keras
scikeras
keras-tuner
xgboost
imbalanced-learn
seaborn
matplotlib
```

## Notes on known issues (fixed)

- A copy-paste bug meant every classical model's Cohen's Kappa was computed against
  Random Forest's predictions instead of its own — fixed, and the corrected values
  are what's reported in the results table above.
- Confusion matrix labels in the ANN notebook were leftover from a different project
  (`"No Insurance"/"Insurance"`) and, later, listed in the wrong order — both fixed.
- SMOTE was being applied to the full dataset before the train/test split, leaking
  synthetic minority samples into the test set — fixed so SMOTE only touches the
  training split. Confirmed by re-running: the test set now correctly reflects the
  true ~20% churn rate instead of an artificially balanced 50/50 split.
