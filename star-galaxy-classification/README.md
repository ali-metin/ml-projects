# SDSS Star vs Galaxy Classification

Binary classification of astronomical objects from the Sloan Digital Sky Survey (SDSS) as **STAR** or **GALAXY** using photometric magnitude features. Six classifiers are compared end-to-end, from EDA and multicollinearity diagnostics through tuned models and error analysis.

---

## Problem Definition

Star/galaxy separation is a foundational task in survey astronomy: every downstream science (galaxy clustering, stellar population studies, transient detection) depends on it. SDSS provides multi-band photometry in five filters (u, g, r, i, z), and the morphological differences between point sources (stars) and extended sources (galaxies) leave a measurable signature across these bands — which is exactly what a classifier can learn from.

## Dataset

- **Source:** Sloan Digital Sky Survey (SDSS) photometric catalogue
- **Size:** 45,000 observations, filtered to 14,561 after applying the SDSS `clean` quality flag
- **Target:** `object_type` — `STAR` (0) or `GALAXY` (1)
- **Features used:**
  - `ra`, `dec` — sky coordinates (J2000)
  - `modelMag_u`, `modelMag_g`, `modelMag_i`, `modelMag_z` — SDSS model magnitudes
  - (`modelMag_r` and `psfMag_r` dropped due to high VIF — see below)

## Pipeline

1. **Data cleaning** — drop non-predictive identifier columns (`objID`, `flags`), remove a clear outlier in `modelMag_u`.
2. **EDA** — target distribution, per-class boxplots and pairplots across all magnitude bands, correlation heatmap.
3. **Multicollinearity diagnosis (VIF)** — `modelMag_r` showed VIF = 29.04 and `psfMag_r` VIF = 12.03, indicating severe collinearity with the other magnitude bands. Both dropped before modelling.
4. **Data filtering** — restrict to `clean == 1` observations (14,561 rows) to ensure photometric reliability.
5. **Modelling** — six classifiers, each tuned with `GridSearchCV` (5-fold `StratifiedKFold`, scoring on `f1_macro`):
   - Logistic Regression (elastic-net, scaled)
   - Random Forest
   - K-Nearest Neighbors (scaled)
   - XGBoost
   - CatBoost
6. **Evaluation** — accuracy, precision, recall, F1, and confusion matrices on a stratified 80/20 test split.

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.766 | 0.689 | 0.632 | 0.659 |
| Random Forest | 0.835 | 0.775 | 0.761 | 0.768 |
| KNN | 0.834 | 0.778 | 0.752 | 0.765 |
| XGBoost | 0.838 | 0.780 | 0.763 | 0.772 |
| **CatBoost** | **0.840** | **0.778** | **0.776** | **0.777** |

### Key findings

- Tree-based ensembles (CatBoost, XGBoost, Random Forest) all clustered around 0.77–0.78 F1, decisively beating the linear baseline (0.659). The decision boundary between stars and galaxies in magnitude space is non-linear.
- Dropping `modelMag_r` and `psfMag_r` based on VIF kept the feature set well-conditioned without measurable loss in performance.
- Filtering to clean observations cost ~68% of the rows but removed unreliable measurements that would have added label noise.
- The target is imbalanced (more stars than galaxies), which is why macro F1 was chosen over plain accuracy as the tuning criterion.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── objects.ipynb           # main analysis notebook
├── Objects.csv             # SDSS photometric catalogue
└── .gitignore
```

## Getting started

```bash
git clone https://github.com/ali-metin/ml-projects.git
cd ml-projects/star-galaxy-classification

python -m venv .venv
source .venv/bin/activate         # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook objects.ipynb
```

The notebook loads `Objects.csv` from the same directory.

## Next steps

- **Colour-index features** (`u−g`, `g−r`, `r−i`, `i−z`) — these are the standard photometric separators in stellar classification and are known to improve linear models substantially.
- **Train on the full dataset** with `clean` as a feature rather than a filter, to see whether the model can learn around lower-quality measurements.
- **Class imbalance handling** — test SMOTE or class weighting and quantify the effect on minority-class recall.
- **Error analysis** — characterise misclassified galaxies (false negatives), particularly faint or compact ones that look point-source-like.
- **CNN on image cutouts** — move from photometric features to the SDSS image stamps for a morphological model.

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · CatBoost · statsmodels · matplotlib · seaborn

## License

MIT
