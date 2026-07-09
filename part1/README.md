# Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis
## Road Accident Severity Analysis (India)

## Dataset

- **Source:** [Road Accident Severity in India, Kaggle](https://www.kaggle.com/datasets/s3programmer/road-accident-severity-in-india)
- **Size:** 3,000 rows, 22 columns
- **Why this dataset:** Road safety is a genuine public-interest problem in
  India, and this dataset has a healthy mix of numeric columns (casualties,
  fatalities, speed limit, driver age) and categorical columns (state,
  weather, road type, lighting, alcohol involvement), satisfying the task
  requirements while supporting a meaningful real-world narrative.

## How to run

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook data_prep_eda.ipynb
```

Run all cells top to bottom. Plots are saved automatically to `plots/`, and
the cleaned dataset is saved as `cleaned_data.csv` at the end.

---

## 1. Load & Inspect

Loaded with `pd.read_csv()`. Shape: **(3000, 22)**. `df.head()`, `df.dtypes`,
and `df.describe()` were used to get a first look at the data.

## 2. Null Value Analysis

| Column | Null Count | Null % |
|---|---|---|
| Driver License Status | 975 | 32.50% |
| Traffic Control Presence | 716 | 23.87% |

**Two columns exceed the 20% missing-value threshold**, both categorical.

All numeric columns (Year, Number of Vehicles Involved, Number of
Casualties, Number of Fatalities, Speed Limit, Driver Age) have **zero**
missing values, so median imputation (intended for numeric columns) was not
applicable anywhere in this dataset — there was nothing to impute
numerically.

For the two high-null categorical columns, missing values were filled with
meaningful placeholder categories rather than the mode:
- `Traffic Control Presence` → filled with `'None'` (assumed no control was
  present when unrecorded)
- `Driver License Status` → filled with `'Unknown'` (status genuinely not
  recorded)

This was chosen over mode-imputation because assuming the most frequent
category for 24–33% of the data would introduce significant bias, whereas
an explicit "Unknown"/"None" category preserves the honesty of what's
actually known versus missing.

## 3. Duplicate Detection

`df.duplicated().sum()` returned **0** — no duplicate rows were found, so
no rows were removed and null percentages were unaffected.

## 4. Data Type Correction & Memory Usage

All columns already had correct dtypes (numeric fields as int64/float64,
categorical fields as object) — there was no genuinely miscast column in
this dataset. To demonstrate the required techniques:

- **Category conversion**: 12 repetitive string columns (State Name,
  Weather Conditions, Road Type, Road Condition, Lighting Conditions,
  Traffic Control Presence, Driver Gender, Driver License Status, Alcohol
  Involvement, Vehicle Type Involved, Accident Severity, Day of Week) were
  converted to the `category` dtype, reducing memory usage (see notebook
  output for exact before/after figures).
- **`pd.to_numeric()` demonstration**: a temporary copy of `Speed Limit
  (km/h)` was cast to string and converted back using
  `pd.to_numeric(..., errors='coerce')` to demonstrate the technique; no
  values were lost, confirming the column was cleanly numeric to begin with.

## 5. Descriptive Statistics & Skewness

All numeric columns show skewness very close to 0:

| Column | Skewness |
|---|---|
| Driver Age | -0.044 (highest absolute) |
| Year | -0.036 |
| Number of Casualties | -0.027 |
| Speed Limit (km/h) | -0.013 |
| Number of Vehicles Involved | 0.001 |
| Number of Fatalities | 0.033 |

**Driver Age** has the highest absolute skewness at -0.044, but this is
negligible — a skew this close to zero indicates an almost perfectly
symmetric distribution, not a meaningfully skewed one. Real-world
distributions typically show skew well above ±0.5 when genuinely skewed.

**Consequence for imputation:** since no column shows meaningful skew, the
choice between mean and median would make very little practical difference
here — for a symmetric distribution, mean ≈ median.

## 6. IQR Outlier Detection

| Column | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outliers Found |
|---|---|---|---|---|---|---|
| Speed Limit (km/h) | 51.0 | 99.0 | 48.0 | -21.0 | 171.0 | 0 |
| Driver Age | 31.0 | 57.0 | 26.0 | -8.0 | 96.0 | 0 |

No outliers were found in either column. Values stay tightly within
realistic bounds with no extreme cases, consistent with the near-zero
skewness observed above. Since there are no outliers to handle, no capping
or removal strategy is needed for Part 2.

## 7. Visualizations

Ten plots were produced in total (five required types, plus five
additional plots for deeper analysis):

**Required types:**
1. **Line plot** (Plot 7) — Number of Casualties across the first 200
   accident records. No clear trend or pattern is visible — casualties
   fluctuate randomly across records with no time-based structure.
2. **Bar chart, mean by category** (Plot 10) — Average casualties per
   accident by road type. Values cluster tightly around 5 across nearly
   all road types, showing little differentiation.
3. **Histogram** (Plot 4, updated) — Distribution of Driver Age (most
   skewed column), bins=20. The distribution is roughly flat/uniform
   across the 18–70 age range, consistent with the near-zero skewness.
4. **Scatter plot** (Plot 8) — Speed Limit vs Number of Casualties. No
   visible linear or curved pattern; points are scattered uniformly,
   confirming the near-zero Pearson correlation (0.023) between these
   variables.
5. **Box plot** (Plot 9) — Driver Age by Accident Severity. Median and
   interquartile range are nearly identical across Minor, Serious, and
   Fatal categories — the boxes overlap almost completely, indicating
   driver age alone does not distinguish severity.

**Additional plots:** Accident Severity Distribution (Plot 1), Severity by
Alcohol Involvement (Plot 2), Severity by Lighting Conditions (Plot 3),
Top 10 States by Accident Count (Plot 6).

## 8. Correlation Heat Map

The highest absolute correlation among numeric features is between Number
of Vehicles Involved and Number of Fatalities (0.039) — still very weak.

Even if this were a stronger correlation, it would not necessarily imply
causation: a plausible third variable is road type or traffic density,
since highways with more vehicles involved in a pileup may also
independently have higher speed limits, which could separately influence
fatality counts. In this dataset specifically, since all correlations are
close to zero, no pair shows strong enough association to draw even a
tentative causal hypothesis.

## 9. Imputation Strategy Comparison (Mean vs Median)

For the two highest-skew columns:

| Column | Mean | Median | Skew |
|---|---|---|---|
| Driver Age | (see notebook output) | (see notebook output) | -0.044 |
| Year | (see notebook output) | (see notebook output) | -0.036 |

Both columns had zero nulls to begin with, confirmed via
`df[cols].isnull().sum()`. Since skew is negligible for both columns, mean
and median are nearly identical — the choice between them would have made
no practical difference had imputation been necessary.

## 10. Spearman vs Pearson Correlation

Top 3 column pairs by |Spearman − Pearson| difference:

| Pair | Difference |
|---|---|
| Number of Vehicles Involved & Driver Age | 0.000667 |
| Number of Vehicles Involved & Number of Casualties | 0.000507 |
| Number of Fatalities & Speed Limit (km/h) | 0.000420 |

These differences are negligible (all under 0.001) — Spearman and Pearson
agree almost perfectly across every pair. Since |Pearson| ≥ |Spearman| in
all three cases, none of these pairs show a monotonic-but-non-linear
pattern that Pearson would miss — there is simply no relationship of
either kind to detect.

**For Part 2 feature selection:** neither correlation measure provides
useful guidance here — no numeric feature shows a linear or rank-based
relationship strong enough to be considered predictive on its own.

## 11. Grouped Aggregation — Casualties by State

- **Highest mean:** Bihar (5.71 average casualties per accident)
- **Lowest mean:** Andhra Pradesh (4.36 average casualties per accident)
- **Highest standard deviation:** Odisha (3.46)

**Is high within-group standard deviation a concern?** Yes — Odisha's std
of 3.46 is close to its own mean (~4.74), meaning individual accidents
within Odisha vary wildly in severity. This high within-group variance
means that knowing "the accident happened in Odisha" alone gives very
little predictive power about how many casualties to expect.

**Ratio of highest to lowest group mean:** 5.71 / 4.36 = **1.307**

A ratio of 1.31 means the highest-casualty state only sees about 31% more
casualties on average than the lowest — a fairly small spread. Combined
with the high within-group standard deviation across nearly every state
(std ≈ 3.0–3.5, close to the mean itself), this suggests **State Name
carries weak predictive signal** for casualty count.

---

## Overall Conclusion

Across every analysis performed in this part — near-zero skewness, zero
IQR outliers, near-zero Pearson and Spearman correlations, and weak
between-state variation relative to within-state variation — this dataset
consistently shows values that appear close to uniformly/randomly
generated rather than reflecting genuine real-world accident dynamics
(where factors like alcohol, speed, and lighting would typically show much
stronger relationships to outcomes). This finding is carried forward and
further tested in Part 2.

## Known Limitations

- The dataset appears to be synthetically generated for practice purposes,
  limiting how meaningful downstream predictive modeling can be (explored
  further in Part 2).
- No exact GPS location is available, limiting analysis to state/city level.

## Files

- `data_prep_eda.ipynb` — full analysis notebook (outputs cleared)
- `plots/` — 10 saved visualizations
- `data/` — raw dataset
- `cleaned_data.csv` — cleaned dataset used in Part 2
- `requirements.txt` — dependencies