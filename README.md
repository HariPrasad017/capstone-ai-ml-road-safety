# Applied AI & ML Capstone — Road Safety in India

An end-to-end AI/ML capstone project analyzing road accident data in India,
built across four parts: rigorous data preparation and exploration, a
supervised machine learning model (regression + classification), advanced
ensemble modeling with hyperparameter tuning, and an LLM-powered batch risk
scoring feature with safety guardrails.

## Project Parts

- **[Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis](./part1)**
  Cleaning and exploring 3,000 road accident records from across India —
  null analysis, dtype correction, skewness, IQR outlier detection, 10
  visualizations, Spearman/Pearson correlation, and grouped aggregation.

- **[Part 2 — Supervised Machine Learning Model](./part2)**
  Regression (Linear + Ridge) predicting casualty count, and classification
  (Logistic Regression) predicting fatal vs non-fatal outcomes — with
  leak-free preprocessing, class-imbalance handling, ROC/AUC, threshold
  sensitivity analysis, and bootstrap confidence intervals.

- **[Part 3 — Advanced Modeling: Ensembles, Tuning, and Full ML Pipeline](./part3)**
  Decision Trees, Random Forest, Gradient Boosting, feature importance and
  ablation, cross-validation, GridSearchCV hyperparameter tuning, a manual
  learning curve, and a serialized, reloadable model pipeline.

- **[Part 4 — LLM-Powered Feature: Batch Risk Scoring](./part4)**
  An LLM-powered batch scoring system (Track B) that assesses accident
  records against a safety rubric, with a PII guardrail, JSON schema
  validation, and a temperature A/B comparison.

## Dataset

[Road Accident Severity in India (Kaggle)](https://www.kaggle.com/datasets/s3programmer/road-accident-severity-in-india) —
3,000 accident records across Indian states, used consistently across all
four parts (Part 1 produces `cleaned_data.csv`, used as the input to
Parts 2–4).

## A Note on the Core Finding

Across all four parts — EDA, regression, classification, ensemble models,
and hyperparameter tuning — this project consistently finds that the
dataset's recorded features (weather, alcohol involvement, lighting, speed,
etc.) carry **no statistically learnable relationship** with accident
severity or casualty count. Every model, from simple linear regression to
tuned ensembles, performs at or near random-chance level. This is discussed
in detail in each part's README, and is treated as a genuine data-science
finding in its own right — correctly diagnosing the absence of signal,
rather than over-fitting in pursuit of one, is itself the point of Parts
1–3. Part 4's LLM feature uses an authored safety rubric rather than
learned patterns for this reason.

## How to Explore

Each part folder is self-contained with its own `README.md`, setup
instructions, and dependencies. Start with whichever part you're
interested in — no need to run them in order to understand any single part,
though Parts 2–4 do depend on `cleaned_data.csv` produced in Part 1.
