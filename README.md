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
| `cohort_features.csv` | Development-cohort covariates (4 columns, 284 rows), used for the percentile notes under each input; SHAP itself needs no background sample |
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

SHAP values come from CatBoost's own path-dependent TreeSHAP,
`get_feature_importance(pool, type="ShapValues")`, on the raw margin (log-odds)
scale. This is the routine that produced Figure 7B of the manuscript, and the app
redraws that panel with the figure's own plotting function, so the two agree digit
for digit: base value `E[f(X)] = -0.75`, contributions `+0.72`, `+0.65`, `+0.30`,
`-0.16`, and `f(x) = 0.76` for the reference patient (A-aDO2 277 mmHg, PEEP
9 cmH2O, SOFA 8, lactate 12.0 mmol/L; `predict_proba` = 0.681).

The baseline is a quantity stored inside the fitted booster, not an average over a
background sample. That makes it independent of `cohort_features.csv`, of row
order and of the installed `shap` release - the failure modes of an interventional
`shap.maskers.Independent` masker, which silently subsamples a 284-row cohort to
its default 100 rows and shifts `E[f(X)]` accordingly.

Because the waterfall is on the log-odds scale, both ends are also translated into
probabilities for the reader:

| End of the waterfall | log-odds | probability |
| --- | --- | --- |
| `E[f(X)]` (cohort average) | -0.75 | 32.0% |
| `f(x)` (this patient) | 0.76 | 68.1% |

One caveat, because the two percentages below are both correct and are easily
confused: the cohort's **mean predicted probability** is 38.9%, whereas the
probability implied by the mean log-odds, `sigma(E[f(X)])`, is 32.0%. The former
averages the individual probabilities, the latter averages the log-odds first; by
Jensen's inequality the second is always the lower of the two. The app uses the
log-odds baseline for the waterfall - it is the only scale on which the four
contributions are exactly additive - and quotes 38.9% only as a calibration
statistic.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `Model file not found` | `catboost_theta6_model.pkl` is not in the same folder as `predictor7.py`. |
| Warning: CatBoost version mismatch | Pin `catboost==1.2.10` in `requirements.txt` and redeploy (or re-export the pickle in the runtime version). The version the file was fitted with is stored in the pickle (`versions.catboost`) and shown in the expander. |
| Warning: `cohort_features.csv` not found | The app falls back to a small built-in reference table for the percentile notes. Predictions and SHAP values are unaffected - the baseline comes from the model itself. |
| `ModuleNotFoundError: No module named 'sklearn'` | You are running an older copy of `predictor7.py` or an old `requirements.txt`. This CatBoost version of the app does not import scikit-learn directly. |
| Build fails on dependency resolution | Repin to a single consistent stack; the versions in `requirements.txt` are known to work together. |

## Disclaimer

Research and educational use only. The model was developed in a single centre
and has not been prospectively validated; inputs outside the observed cohort
range are extrapolations. The output must not replace clinical judgement and must
not be the sole basis for treatment-limitation decisions.
