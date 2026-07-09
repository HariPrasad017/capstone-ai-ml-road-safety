# Part 2 — Supervised Machine Learning — Build, Train, and Evaluate
## Road Accident Severity & Casualty Prediction (India)

## Setup

Loaded `cleaned_data.csv` from Part 1. Defined:

- **Feature matrix X**: all columns except `Number of Casualties` and
  `Accident Severity`.
- **Regression label `y_reg`**: `Number of Casualties` (continuous numeric).
- **Classification label `y_clf`**: binary, derived from `Accident
  Severity` — `1` if Fatal, `0` otherwise (Minor or Serious). This was
  chosen over a median-split of `y_reg` because it's a more meaningful,
  real-world-relevant binary outcome for a road-safety context ("was this
  accident fatal or not"), and it ties directly into the risk-assessment
  system built in Part 4.

## How to run

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook train_model.ipynb
```

---

## Encoding

- `Time of Day` (1,263 unique timestamps) was converted into 4 time
  buckets (Morning/Afternoon/Evening/Night) — too high-cardinality to
  one-hot encode directly.
- `City Name` was dropped (redundant with `State Name`, high cardinality).
- All remaining categorical columns have **no natural order** (state
  names, weather, road type, etc.), so **one-hot encoding** (`pd.get_dummies()`,
  `drop_first=True`) was used throughout — no ordinal/label encoding was
  needed, since none of the categorical features here have a meaningful
  inherent ranking (e.g., there's no natural order to weather conditions
  like Rainy vs Foggy). One-hot encoding avoids implying a false ordinal
  relationship that label encoding would otherwise introduce.

## Leak-Free Train-Test Split & Scaling

- Split 80/20 using `train_test_split(..., stratify=y_clf, random_state=42)`.
- `StandardScaler` was fit **only on `X_train`** (`scaler.fit(X_train)`),
  then applied to both `X_train` and `X_test` via `.transform()`.

Fitting the scaler on the full dataset (including test data) would
constitute **data leakage** — the scaler's mean/std would then encode
information about the test set's distribution into preprocessing, giving
the model indirect access to test-set statistics before evaluation and
inflating apparent performance.

---

## Regression — Linear vs Ridge

| Model | MSE | R² |
|---|---|---|
| Linear Regression | 10.7689 | -0.0295 |
| Ridge Regression | 10.7686 | -0.0295 |

Both models produce a **negative R²**, meaning they perform *worse* than a
naive baseline that simply predicts the mean casualty count for every row.
This is consistent with Part 1's EDA findings (near-zero skewness,
near-zero correlations, near-zero Spearman/Pearson differences): `Number
of Casualties` has no learnable linear relationship with any available
feature in this dataset.

**Top 3 coefficients by magnitude** (Linear Regression): Weather
Conditions (Stormy), State Name (Bihar), State Name (Puducherry) — all
with small coefficients (~0.19–0.23). A coefficient this size means even a
1-standard-deviation change in that feature is associated with less than a
quarter-casualty change in the prediction — negligible in practical terms.
A large positive coefficient would mean that feature is associated with
*more* predicted casualties as it increases; a large negative coefficient
would mean *fewer* — but no coefficient here is large enough to be
practically meaningful.

**Ridge vs OLS**: Ridge (alpha=1.0) produces nearly identical MSE and R²
to plain Linear Regression. Ridge adds an L2 penalty that shrinks large
coefficients toward zero, controlled by the `alpha` parameter (higher
alpha = stronger shrinkage). Since the OLS coefficients were already
small, Ridge had very little to shrink, explaining the near-identical
performance. Ridge would diverge more visibly from OLS on data with strong
multicollinearity or large/unstable coefficients — neither is present here.

---

## Classification — Logistic Regression

### Class Imbalance

`y_clf_train` distribution: **1,612 Not-Fatal (67.2%) vs 788 Fatal
(32.8%)**. Since the minority class falls below the 35% threshold,
**`class_weight='balanced'`** was used (chosen over SMOTE for simplicity —
it requires no additional resampling library and directly reweights the
loss function to compensate for the imbalance).

### Results

Confusion Matrix:
|  | Predicted Not-Fatal | Predicted Fatal |
|---|---|---|
| **Actual Not-Fatal** | 202 | 201 |
| **Actual Fatal** | 94 | 103 |

Accuracy: **50.8%** — statistically no better than a coin flip for a
2-class problem.

**Precision** = TP / (TP + FP) — of all accidents predicted Fatal, what
fraction actually were fatal.
**Recall** = TP / (TP + FN) — of all actually fatal accidents, what
fraction were correctly caught.

For the Fatal class: Precision = 0.34, Recall = 0.52, F1 = 0.41.

**Which metric matters more for this task?** Recall matters more than
Precision here — in a real road-safety application, missing a genuinely
fatal-risk scenario (a false negative) is costlier than incorrectly
flagging a non-fatal scenario as high-risk (a false positive), since the
former means a missed safety-intervention opportunity. Our model's Fatal
recall (0.52) is barely better than random, reinforcing that this dataset
cannot support a reliable safety-critical classifier as-is.

### ROC Curve & AUC

**AUC = 0.523** — only marginally above 0.5 (pure random guessing). The
ROC curve closely tracks the diagonal "random guess" line throughout.

An AUC of 0.523 means the model has almost no ability to rank a randomly
chosen Fatal accident as higher-risk than a randomly chosen Not-Fatal one
— practically indistinguishable from a coin flip.

---

## Decision Threshold Sensitivity

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.331 | 0.985 | 0.496 |
| 0.40 | 0.338 | 0.812 | 0.478 |
| 0.50 | 0.339 | 0.523 | 0.411 |
| 0.60 | 0.359 | 0.168 | 0.228 |
| 0.70 | 0.364 | 0.020 | 0.038 |

**F1-maximizing threshold: 0.30** (F1 = 0.496).

As the threshold rises, Precision stays roughly flat (~0.33–0.36) while
Recall collapses from 0.985 to 0.020 — the model rarely gains real
confidence in its Fatal predictions even at high thresholds, because it
isn't learning a genuine signal.

Since Recall matters more for this safety-critical task (see above), a
**lower threshold (0.30)** is preferable — it catches 98.5% of actual
fatal cases, at the cost of low precision (33%, meaning two-thirds of
"Fatal" predictions would be false alarms). The cost of lowering the
threshold is a high false-positive rate, which would need to be weighed
against real-world operational costs of over-flagging.

---

## Regularization Experiment — C=0.01 vs C=1.0

| Model | Precision | Recall | AUC |
|---|---|---|---|
| C=1.0 (baseline) | 0.339 | 0.523 | 0.523 |
| C=0.01 (strong regularization) | 0.347 | 0.543 | 0.528 |

`C` is the inverse of regularization strength — a smaller C means a
stronger L2 penalty, shrinking coefficients more aggressively toward zero
for a simpler decision boundary.

Performance is nearly identical between the two settings; C=0.01 is
marginally better across all three metrics, but the difference is small
enough to likely be noise (confirmed below). Since neither model was
learning real signal to begin with, stronger regularization simply shrinks
already-weak, uninformative coefficients further toward zero without
meaningfully changing predictive capability.

## Bootstrap Confidence Interval — AUC Difference

- Mean AUC difference (C=1.0 − C=0.01): **-0.0044**
- 95% CI (500 bootstrap resamples): **[-0.0137, 0.0034]**
- CI excludes zero: **No**

Since the 95% confidence interval **includes zero**, the small AUC
difference between the two regularization strengths is **not statistically
reliable** — consistent with the conclusion that neither C setting
produces a meaningfully better classifier, because neither extracts a real
signal from this dataset.

---

## Overall Conclusion

Across regression (negative R²), classification (50.8% accuracy, AUC
0.523), threshold sensitivity, and regularization experiments, every model
consistently performs at or near random-chance level. The bootstrap
confidence interval confirms even the small differences between
regularization strengths are not statistically reliable.

This is **not a flaw in the modeling approach** — preprocessing, encoding,
scaling, and imbalance handling were all done correctly and are
demonstrated in full — but a property of the dataset itself, consistent
with Part 1's finding that this data appears synthetically/uniformly
generated rather than reflecting genuine real-world accident dynamics
(where factors like alcohol, speed, and lighting would typically show much
stronger relationships to outcomes).

## Known Limitations

- The apparent absence of real signal in this dataset means the trained
  models here demonstrate correct ML methodology rather than a deployable,
  accurate predictor.
- With more time, sourcing a real (non-synthetic) accident dataset from an
  official government road-safety database would be the natural next step
  to validate whether these techniques perform better on genuine data.

## Files

- `train_model.ipynb` — full modeling notebook (outputs cleared)
- `plots/` — confusion matrix and ROC curve visualizations
- `requirements.txt` — dependencies