# Part 2 — Supervised Machine Learning
## Predicting Road Accident Severity (India)

## What is this project about?

Building on Part 1's data exploration, this part attempts to build a machine
learning model that predicts how severe a road accident will be (Minor,
Serious, or Fatal) based on conditions like weather, lighting, alcohol
involvement, road type, and driver details.

Just as important as building the model is being honest about what the
results actually show — and this project surfaced an important, real
finding, explained below.

---

## What's inside this folder?

| File / Folder | What it contains |
|---|---|
| `data/accident_prediction_india.csv` | The dataset (same as Part 1) |
| `train_model.ipynb` | Full notebook — feature engineering, model training, evaluation |
| `plots/` | Confusion matrix and feature importance charts |
| `requirements.txt` | List of Python tools needed to run this project |
| `README.md` | This file |

---

## How to run this yourself

1. **Install Python** (3.9+) if not already installed.
2. **Open a terminal inside this folder**, then set up an isolated environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. **Install required tools**:
   ```
   pip install -r requirements.txt
   ```
4. **Open and run the notebook**:
   ```
   jupyter notebook
   ```
   Open `train_model.ipynb` and run each cell top to bottom with `Shift + Enter`.

---

## What was done, step by step

1. **Cleaned missing values** — same approach as Part 1 (`Traffic Control
   Presence` and `Driver License Status` nulls filled with meaningful
   placeholder categories rather than dropped).

2. **Feature engineering** — the raw `Time of Day` column had 1,263 unique
   timestamp values (e.g. `14:35`), which is unusable for a model directly.
   This was converted into four meaningful buckets: Morning, Afternoon,
   Evening, and Night.

3. **Removed redundant columns** — `City Name` was dropped since it largely
   duplicates the information already captured in `State Name`, and keeping
   both would have added many extra columns without adding real value.

4. **Encoding** — all remaining categorical columns (state, weather, road
   type, etc.) were converted into numeric form using one-hot encoding, since
   machine learning models require numeric input.

5. **Feature scaling** — for Logistic Regression specifically, all numeric
   features were standardized (put on the same scale), since this type of
   model is sensitive to differences in feature ranges. Random Forest does
   not need this step, since tree-based models are unaffected by feature scale.

6. **Train/test split** — 80% of the data used for training, 20% held out for
   testing, with stratification to keep the proportion of each severity class
   balanced in both sets.

7. **Trained two models**:
   - **Logistic Regression** (baseline) — a simple, interpretable linear model
   - **Random Forest** (tuned) — a more complex ensemble model, better suited
     to capturing non-linear patterns

---

## Results

| Model | Accuracy | F1 Score (weighted) |
|---|---|---|
| Logistic Regression (Baseline) | 33.3% | 0.333 |
| Random Forest (Tuned) | 33.2% | 0.329 |

**With 3 severity classes (Minor/Serious/Fatal), a model that guesses
completely at random would also score around 33% accuracy on average.** Both
of our trained models perform almost exactly at this random-chance level —
meaning neither model actually learned any useful pattern from the data.

---

## Key Finding: Why didn't the models perform better?

Rather than assuming the modeling approach was flawed, we investigated
*why* performance was this low by checking whether the dataset's features
actually relate to the severity outcome at all.

We compared the proportion of Minor/Serious/Fatal outcomes across different
categories of key features:

- **Alcohol Involvement:** ~33% Fatal whether alcohol was involved or not
  (32.7% vs 33.0%)
- **Lighting Conditions:** ~33% Fatal across Dark, Dawn, Daylight, and Dusk
  conditions alike
- **Weather Conditions:** ~33% Fatal across Clear, Foggy, Hazy, Rainy, and
  Stormy conditions alike

In real-world accident data, we would expect these factors to noticeably
shift the likelihood of a fatal outcome (for example, alcohol involvement
typically increases fatal accident risk substantially). Here, every category
lands within a percentage point or two of the overall average — for every
feature tested.

**Conclusion:** This dataset's `Accident Severity` labels appear to be
randomly or independently assigned, with no statistical relationship to the
recorded accident conditions. This means no machine learning model — no
matter how well-tuned — could achieve meaningfully better-than-random
performance on this specific prediction task, because the signal simply
isn't present in the data.

This is a common and important scenario in real-world data science: not
every dataset contains a learnable relationship for the question being
asked, and correctly diagnosing that (rather than over-tuning a model
chasing a signal that doesn't exist) is itself a valuable and necessary
skill.

---

## What would be done differently with more time / real data

- Source a real (non-synthetic) accident dataset, ideally from an official
  government road safety database, where severity outcomes are grounded in
  actual reported incidents.
- If restricted to this dataset, pivot to a task the data can actually
  support — for example, analyzing correlations and general trends (as in
  Part 1) rather than building a predictive model, since prediction requires
  a genuine signal to learn from.

---

## Known Limitations

- The dataset appears to be synthetically generated for practice purposes,
  which limits how meaningful any trained model's predictions are.
- Given this, model comparison here demonstrates correct ML methodology
  (proper train/test split, encoding, scaling, evaluation) rather than a
  deployable, accurate predictor.