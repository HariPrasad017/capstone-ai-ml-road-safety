# Part 3 — Advanced Modeling: Ensembles, Tuning, and Full ML Pipeline
## Road Accident Fatal-Risk Classification (India)

## Setup

Uses `cleaned_data.csv` from Part 1, with the same preprocessing pipeline
established in Part 2 (null handling, time-bucket feature engineering,
one-hot encoding, leak-free train/test split and scaling), and the same
binary classification target: `y_clf` = 1 if Accident Severity is Fatal,
else 0.

## How to run

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook ensemble_modeling.ipynb
```

---

## 1. Decision Tree — Unconstrained Baseline

| Metric | Value |
|---|---|
| Training Accuracy | 1.0000 (100%) |
| Test Accuracy | 0.6150 (61.5%) |

The unconstrained tree achieves perfect training accuracy but drops to
61.5% on the test set — a large train/test gap indicating clear
**overfitting**. Decision trees are described as high-variance models
because they fit the training data greedily: at each split, the tree picks
the feature/threshold that best separates the current training subset,
without ever revisiting or reconsidering earlier splits. Left
unconstrained, it keeps splitting until every training example is
perfectly classified, capturing noise and dataset-specific quirks rather
than a generalizable pattern.

## 2. Decision Tree — Controlled (max_depth=5, min_samples_split=20)

| Metric | Unconstrained | Controlled |
|---|---|---|
| Training Accuracy | 1.0000 | 0.6867 |
| Test Accuracy | 0.6150 | 0.6367 |
| Train-Test Gap | 0.3850 | 0.0500 |

Constraining the tree dramatically reduces the train-test gap from 0.385
to 0.050. The controlled tree's test accuracy is actually slightly higher
(0.637 vs 0.615) despite lower training accuracy — the unconstrained
tree's "perfect" training fit was pure memorization, not useful learning.

**`max_depth`** limits how many levels deep the tree can grow, trading a
small amount of bias for a large reduction in variance. **`min_samples_split`**
prevents a node from splitting unless it has at least 20 samples, stopping
splits based on tiny, noisy subsets that wouldn't generalize.

## 3. Gini vs Entropy Comparison

| Criterion | Test Accuracy |
|---|---|
| Gini | 0.6367 |
| Entropy | 0.6367 |

**Gini Impurity:** 1 − Σ pᵢ² &nbsp;&nbsp; **Entropy:** −Σ pᵢ log₂(pᵢ)

A node with **Gini = 0** means all samples in that node belong to a single
class — perfectly "pure." Both criteria produced identical accuracy here;
Gini and Entropy typically select very similar splits in practice, and
Gini is often preferred by default for its slightly lower computational cost.

## 4. Random Forest

| Metric | Value |
|---|---|
| Training Accuracy | 0.7300 |
| Test Accuracy | 0.6717 |
| Test ROC-AUC | 0.5064 |

Random Forest achieves the highest raw test accuracy so far (67.2%), but
its **ROC-AUC of 0.506 is essentially at the random-guess baseline** —
confirming, as with every model in this project, that it cannot genuinely
distinguish Fatal from Not-Fatal beyond chance. Higher accuracy alone is
misleading here given the class imbalance; AUC is the more honest measure.

**Feature Importance (Top 5):**

| Feature | Importance |
|---|---|
| Speed Limit (km/h) | 0.087 |
| Driver Age | 0.074 |
| Year | 0.044 |
| Number of Fatalities | 0.040 |
| Number of Vehicles Involved | 0.037 |

Random Forest computes feature importance as the **average reduction in
Gini impurity** contributed by each feature across all splits and all
trees. This differs from a linear regression coefficient, which measures a
linear, additive effect — importance instead reflects how useful a feature
is for *splitting* data into purer groups, capturing non-linear effects a
coefficient cannot. Even the top feature has a low importance score (0.087)
— no feature dominates.

**Bagging & variance reduction:** Random Forest uses bootstrap aggregating
— each of the 100 trees trains on an independent bootstrap sample, and at
each split only a random subset of √(features) is considered. This double
randomization means individual trees make different errors; averaging
their predictions cancels out much of this variance, making the ensemble
far more stable than any single deep tree, even though no individual tree
is more accurate on its own.

## 4a. Gradient Boosting

| Metric | Value |
|---|---|
| Training Accuracy | 0.7425 |
| Test Accuracy | 0.6600 |
| Test ROC-AUC | 0.4950 |

Gradient Boosting's AUC (0.495) is essentially identical to random
guessing — even slightly below 0.5 — reinforcing that no ensemble method,
however sophisticated, can extract signal that isn't present in the data.

## 4b. Feature Ablation Study

5 lowest-importance features removed: `State Name_Madhya Pradesh`,
`State Name_Himachal Pradesh`, `State Name_Jharkhand`, `State Name_Tripura`,
`State Name_Tamil Nadu`.

| Model | Test AUC |
|---|---|
| Full (all features) | 0.5064 |
| Reduced (5 features removed) | 0.4976 |

The AUC drop (0.0088) is tiny — well within noise given both scores hover
near 0.5. These 5 features were **genuinely uninformative**: removing them
didn't meaningfully hurt performance.

**Production trade-off:** dropping uninformative features would normally
reduce model size and inference cost with negligible accuracy loss — a
reasonable simplification in general. However, since the *entire* feature
set here shows no strong signal (best AUC ≈ 0.51 across all models), this
ablation doesn't change the broader conclusion: a simpler model is only
preferable *if* the dataset were otherwise usable for prediction, which it
is not.

## 5. Cross-Validated Comparison — 5-Fold AUC

| Model | Mean AUC | Std AUC |
|---|---|---|
| Logistic Regression | 0.5207 | 0.0187 |
| Decision Tree (max_depth=5) | 0.5222 | 0.0361 |
| Random Forest | 0.5197 | 0.0247 |
| Gradient Boosting | 0.5290 | 0.0297 |

All four models cluster tightly around 0.52 AUC — indistinguishable from
random guessing and from each other, especially given the standard
deviations are comparable in size to the differences between means.

**Why cross-validation is more reliable than a single split:** a single
80/20 split's result depends heavily on which rows happened to land in the
test set. 5-fold CV trains and evaluates 5 times over different held-out
folds and averages the results, giving both a more stable performance
estimate and a measure of variability that a single split cannot provide.

## 6. Hyperparameter Tuning — GridSearchCV

**Best Parameters:** `max_depth=10`, `min_samples_leaf=5`, `n_estimators=100`
**Best CV Score (AUC):** 0.5252

**Total configurations evaluated:** 3 × 3 × 2 = 18 configurations × 5 folds
= **90 model fits**.

Even after exhaustively searching 18 hyperparameter combinations, the best
achievable AUC (0.525) is barely above the manually-chosen Random Forest's
earlier CV score (0.520) — tuning cannot compensate for the absence of
real signal in the features.

**Grid Search vs Randomized Search:** Grid Search exhaustively evaluates
every combination, guaranteeing the best combination *within the grid* is
found, but cost grows multiplicatively with parameters/values tested (90
fits for a modest 3-parameter grid here). Randomized Search samples a
fixed number of random combinations instead, scaling far better to larger
search spaces, at the cost of no longer guaranteeing the single best
combination is tested.

## 7. Manual Learning Curve

| Training Fraction | Training AUC | Test AUC |
|---|---|---|
| 20% | 0.9893 | 0.4753 |
| 40% | 0.9786 | 0.4830 |
| 60% | 0.9823 | 0.5115 |
| 80% | 0.9752 | 0.4936 |
| 100% | 0.9636 | 0.4903 |

**(i) Training AUC trend:** decreases modestly (0.989 → 0.964) as training
size grows — expected for a high-variance model that can nearly perfectly
fit a small dataset but has more trouble memorizing a larger one.

**(ii) Test AUC trend:** stays flat and noisy around 0.48–0.51 regardless
of training set size, with no upward trend — if more data reliably helped,
we'd expect a steady climb toward the training AUC.

**(iii) Conclusion:** the persistent, unclosing gap between training AUC
(~0.96–0.99) and test AUC (~0.48–0.51) at *every* training fraction —
including 100% of available data — is the signature of a dataset with no
learnable relationship between features and target. Collecting more data
of the same kind would not help; the model fits noise at every scale
rather than a real pattern. **This is not a data-quantity problem — it is
a fundamental absence of signal**, consistent with every other finding
across Parts 1–3.

## 8. Model Serialization

The best GridSearchCV pipeline was retrained on the full training set and
saved via `joblib.dump()` as `best_model.pkl` (1.5 MB, committed to the
repository).

**Reload verification:** the saved model was reloaded with `joblib.load()`
and used to predict on two held-out test samples:
- Sample 1: predicted 0 (Not-Fatal), actual 0 — correct
- Sample 2: predicted 0 (Not-Fatal), actual 1 (Fatal) — incorrect

The reload-and-predict pipeline runs without errors. The one incorrect
prediction is consistent with the model's near-random overall performance
(AUC ≈ 0.51).

---

## Summary Comparison — All Models (Parts 2 & 3)

| Model | 5-Fold CV Mean AUC | 5-Fold CV Std AUC | Test-Set AUC |
|---|---|---|---|
| Logistic Regression | 0.5207 | 0.0187 | 0.5232 |
| Decision Tree (max_depth=5) | 0.5222 | 0.0361 | — |
| Random Forest | 0.5197 | 0.0247 | 0.5064 |
| Gradient Boosting | 0.5290 | 0.0297 | 0.4950 |
| Random Forest (GridSearch-tuned) | 0.5252 | — | — |

## Final Model Recommendation

**Recommendation: Logistic Regression**, with the important caveat that
**no model in this comparison is genuinely deployable for this task**.

All five models cluster within a narrow 0.495–0.529 AUC band —
statistically indistinguishable from random guessing (AUC = 0.5), and
Part 2's bootstrap confidence interval confirmed even small differences
between models are not statistically reliable. Given this, Logistic
Regression is the practical choice: it matches the AUC of the more complex
ensemble methods, trains far faster, is fully interpretable (unlike Random
Forest or Gradient Boosting), and has the lowest cross-validation variance
(std = 0.0187) — the most stable estimate of the five. The added
complexity of ensemble methods buys no accuracy improvement here.

**The more important recommendation to the client** is that **no model
should be deployed on this dataset for accident-severity prediction**.
Every technique tested — linear/regularized regression, logistic
regression with imbalance handling, decision trees at multiple constraint
levels, random forests, gradient boosting, and exhaustive hyperparameter
tuning — converges on the same conclusion from Part 1's EDA: this
dataset's features carry no statistically learnable relationship with
either casualty count or fatal/non-fatal outcome. Sourcing genuine,
non-synthetic accident data should be the priority before building any
predictive model intended for real-world use.

## Known Limitations

- As established across all three parts, this dataset appears
  synthetically/uniformly generated, so all models here demonstrate
  correct ML methodology rather than deployable predictive accuracy.
- `best_model.pkl` was tuned and serialized purely to satisfy the
  technical pipeline requirements of this task; it should not be used for
  real predictions given its near-random performance.

## Files

- `ensemble_modeling.ipynb` — full notebook (outputs cleared)
- `best_model.pkl` — serialized best pipeline from GridSearchCV
- `plots/` — feature importance visualization
- `requirements.txt` — dependencies