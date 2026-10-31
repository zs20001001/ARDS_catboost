# ARDS 28-Day Mortality Risk Calculator

A Streamlit web calculator for the CatBoost model in
`catboost_theta6_model.pkl`. It estimates the individual probability of
28-day mortality for invasively ventilated adults with moderate-to-severe ARDS,
using four Day-3 variables, and explains each prediction with SHAP.

## Repository contents

| File | Purpose |
| --- | --- |
| `predictor7.py` | Streamlit application (**main script**) |
| `catboost_theta6_model.pkl` | Trained `CatBoostClassifier`, saved with `joblib` as a dictionary bundle: `model`, `features`, `target`, `params`, `cutoff`, `train_info`, `versions`, `saved_at` |
| `cohort_features.csv` | Development-cohort covariates (4 columns, 284 rows), used as the SHAP background and percentile reference |
| `requirements.txt` | Pinned dependencies |
| `.streamlit/config.toml` | Light theme settings |

## Model summary

* **Algorithm**: CatBoost, `CatBoostClassifier`
  (`depth` = 6, `learning_rate` = 0.06, `iterations` = 200,
  `l2_leaf_reg` = 120, `random_strength` = 6, `border_count` = 96,
  `loss_function` = `Logloss`, `random_seed` = 42, `thread_count` = 1)
* **Development cohort**: 284 invasively ventilated ARDS patients (111 deaths,
  39.1%). The model was fitted on the complete cohort.
* **Outcome**: 28-day all-cause mortality
* **Predictors** (Day 3 of invasive ventilation), in the model's column order:
  A-aDO2 (mmHg), PEEP (cmH2O), SOFA score, arterial lactate (mmol/L)
* **Discrimination**: apparent AUROC 0.954; 10-fold cross-validated AUROC 0.903
  (SD 0.085, fold range 0.687-0.970). Internal (n = 123) and external (n = 206)
  validation are reported in the manuscript, not in this app.
* **Calibration on the development cohort**: mean predicted risk 38.9% vs
  observed 39.1%; apparent Brier 0.100, cross-validated Brier 0.131
* **Ranked variable importance (CatBoost, prediction values change)**: PEEP
  75.2%, SOFA 11.2%, A-aDO2 7.2%, lactate 6.5%
* **Decision threshold**: 0.50

## Run locally

```bash
pip install -r requirements.txt
streamlit run predictor7.py
```

The app opens at <http://localhost:8501>.

## Deploy on Streamlit Community Cloud

1. Create a public GitHub repository and upload the five files listed above,
   keeping them in the repository root. `catboost_theta6_model.pkl` (~47 KB) and
   `cohort_features.csv` must be committed - they are the only artefacts the app
   needs, and neither contains patient identifiers (the `.csv` holds the four
   model covariates only, with no outcome, dates, or identifiers).
2. Sign in to <https://share.streamlit.io> with GitHub, click **New app**, pick
   the repository and branch.
3. Set **Main file path** to `predictor7.py` (not `app.py`).
4. In *Advanced settings* choose Python **3.13**, the version used to fit and
   verify the model. `numpy` 2.x and `pandas` 2.3 are not installable on older
   interpreters.
5. Click **Deploy**. The first build takes a few minutes because `catboost`,
   `shap` and `matplotlib` are large wheels.

## Why catboost is pinned exactly

A CatBoost pickle stores the booster together with its Python wrapper, so the
library version used at load time should match the version used at fit time.
Loading this file under a different version either raises an error or shifts the
predicted probabilities. `requirements.txt` therefore pins `catboost==1.2.10`,
the version recorded inside the pickle as `versions.catboost` (the app reads that
field and warns if the runtime differs). If you retrain the model, update both
the `.pkl` and the pin together.

`scikit-learn` is deliberately **not** pinned any more: it is not used for
prediction, only pulled in as a dependency of `shap`. The earlier random-forest
version of this app needed an exact pin because a scikit-learn pickle stores the
estimator object itself.

## How the explanation works

`shap.TreeExplainer` is built once and cached. The baseline of a SHAP waterfall is
not a universal constant - it is the model output averaged over a reference
population, so both the chosen reference and the number of rows matter:

| Reference population | E[f(X)] for a test patient | Note |
| --- | --- | --- |
| Path-dependent (no background) | 32.0% | classic TreeSHAP; averages over tree coverage rather than over patients, computed in log-odds and mapped to probability |
| 100-row subsample (SHAP default, original row order) | 43.1% | silently applied when the background has more than 100 rows |
| 100-row subsample (same data, shuffled row order) | 35.9% | the draw follows row order, so the baseline is not stable across re-uploads |
| Full cohort, n = 284 | 38.9% | equals the cohort's mean predicted risk; what this app uses |

That is why the background is passed through `shap.maskers.Independent` with
`max_samples = len(background)`: the baseline then equals the mean predicted risk
of the development cohort exactly, and no longer depends on row order or on the
`shap` release. Contributions are reported on the probability scale, in
percentage points, and satisfy `E[f(X)] + sum(SHAP) ≈ predicted probability` up
to numerical error (0.00 percentage points for the model shipped here).

CatBoost returns a 2-D matrix `(n_samples, n_features)` for a binary target,
while some wrappers return one value per feature **per class**. The app handles
both layouts and locates the death class through `model.classes_` rather than
assuming a column position.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `Model file not found` | `catboost_theta6_model.pkl` is not in the same folder as `predictor7.py`. |
| Warning: CatBoost version mismatch | Pin `catboost==1.2.10` in `requirements.txt` and redeploy (or re-export the pickle in the runtime version). The version the file was fitted with is stored in the pickle (`versions.catboost`) and shown in the expander. |
| Warning: `cohort_features.csv` not found | The app falls back to a small built-in reference table. Predictions are unchanged, but the waterfall baseline will not match the published value. |
| `ModuleNotFoundError: No module named 'sklearn'` | You are running an older copy of `predictor7.py` or an old `requirements.txt`. This CatBoost version of the app does not import scikit-learn directly. |
| Build fails on dependency resolution | Repin to a single consistent stack; the versions in `requirements.txt` are known to work together. |

## Disclaimer

Research and educational use only. The model was developed in a single centre
and has not been prospectively validated; inputs outside the observed cohort
range are extrapolations. The output must not replace clinical judgement and must
not be the sole basis for treatment-limitation decisions.
