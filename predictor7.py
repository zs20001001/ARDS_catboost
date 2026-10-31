# -*- coding: utf-8 -*-
"""
ARDS 28-Day Mortality Risk Calculator
=====================================

Streamlit front-end for the CatBoost model stored as
``catboost_theta6_model.pkl``. The file is a joblib pickle of a dictionary that
bundles the fitted estimator with its provenance::

    {"model": CatBoostClassifier, "features": [...], "target": "outcome",
     "params": {...}, "cutoff": 0.5, "train_info": {...}, "versions": {...},
     "saved_at": "..."}

A bare estimator (no metadata) is also accepted, so an alternative .pkl can be
dropped in without touching this script.

Model specification (see ``CatBoost_θ6_模型_训练与保存.py``)
----------------------------------------------------------
* CatBoostClassifier(depth = 6, learning_rate = 0.06, iterations = 200,
  l2_leaf_reg = 120, random_strength = 6, border_count = 96,
  loss_function = "Logloss", eval_metric = "AUC", random_seed = 42,
  thread_count = 1).
* Trained on the complete development cohort (n = 284, 111 deaths, 39.1%).
* Predictors, in the model's own column order:
  ``d3_a_ado2``, ``peep``, ``d3_sofa``, ``d3_lac``.
* Reported discrimination: apparent AUROC 0.954; 10-fold cross-validated AUROC
  0.903 (SD 0.085, fold range 0.687-0.970). Internal and external validation are
  reported in the accompanying manuscript and are not bundled with this app.

Repository layout expected by this script (all files in the same folder):

    predictor7.py                  <- this file (main script for Streamlit Cloud)
    catboost_theta6_model.pkl      <- trained model (joblib bundle)
    cohort_features.csv            <- development-cohort covariates, used as the
                                      SHAP background / percentile reference
    requirements.txt

Notes on reproducibility
------------------------
* The pickle stores a CatBoost object, so the runtime ``catboost`` version
  should match the one recorded in the bundle (``versions.catboost``); the app
  reads that field, warns on a mismatch, and shows both versions in the
  "Model details and limitations" panel. ``requirements.txt`` pins the version.
* The SHAP baseline is not a universal constant: it is the model output averaged
  over whatever reference population you integrate over.
  ``shap.maskers.Independent`` defaults to ``max_samples=100``, i.e. it silently
  replaces a 284-row cohort by a 100-row subsample (drawn with a fixed seed, but
  from whatever row order the CSV happens to have), which shifts E[f(X)] by
  several percentage points. Passing ``max_samples = n`` keeps the full cohort,
  so the baseline equals the cohort's mean predicted risk (38.94%) and no longer
  depends on row order.
* The prediction path takes its column order from the bundle, never from the
  form: a permuted design matrix is silently accepted by CatBoost (no error, just
  wrong probabilities), so the vector is always rebuilt in ``features`` order.
"""

from pathlib import Path

import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import catboost
import joblib
import shap
import streamlit as st

# ============================================================================
# 1. Configuration
# ============================================================================
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "catboost_theta6_model.pkl"
BACKGROUND_PATH = APP_DIR / "cohort_features.csv"

# catboost version the model was fitted with (keep in sync with requirements.txt).
# The bundle records this value; the constant is only a fallback for a bare
# estimator, which carries no metadata.
PINNED_CATBOOST = "1.2.10"

# Column order of the model itself - do not reorder without retraining.
# The .pkl bundle also carries this list; when present it takes precedence, so
# the app follows the order the fitted model actually saw.
FEATURES = ["d3_a_ado2", "peep", "d3_sofa", "d3_lac"]

LABELS = {
    "d3_a_ado2": "Day-3 A-aDO2 (mmHg)",
    "peep": "PEEP (cmH2O)",
    "d3_sofa": "Day-3 SOFA score",
    "d3_lac": "Day-3 lactate (mmol/L)",
}

CAPTIONS = {
    "d3_a_ado2": "Alveolar-arterial oxygen gradient",
    "peep": "Positive end-expiratory pressure",
    "d3_sofa": "Sequential Organ Failure Assessment, 0-24",
    "d3_lac": "Arterial blood lactate",
}

# (min, max, step, default); defaults are the development-cohort medians.
INPUT_SPEC = {
    "d3_a_ado2": (0.0, 620.0, 1.0, 190.6),
    "peep": (0, 30, 1, 6),
    "d3_sofa": (0, 24, 1, 8),
    "d3_lac": (0.1, 20.0, 0.1, 1.4),
}

DECISION_THRESHOLD = 0.50

# Stand-in used only if cohort_features.csv is missing: the app still works, but
# the SHAP baseline (E[f(X)]) is then computed over these rows instead of the
# cohort, so the contributions no longer sum to the published baseline.
FALLBACK_BACKGROUND = pd.DataFrame(
    {
        "d3_a_ado2": [100.0, 150.0, 190.0, 250.0, 320.0, 420.0, 550.0],
        "peep": [4.0, 5.0, 6.0, 6.0, 8.0, 10.0, 12.0],
        "d3_sofa": [3.0, 4.0, 8.0, 9.0, 12.0, 15.0, 18.0],
        "d3_lac": [0.8, 1.0, 1.4, 1.6, 2.4, 4.0, 7.0],
    }
)


# ============================================================================
# 2. Cached resources
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_bundle():
    """
    Load the model file.

    The published artefact is a dictionary that packages the fitted estimator
    together with its provenance. A bare estimator (no metadata) is also
    accepted, so an alternative .pkl can be dropped in without touching this
    script.
    """
    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict):
        model = bundle["model"]
        features = [str(f) for f in bundle.get("features") or FEATURES]
        meta = bundle
    else:
        model = bundle
        features = list(FEATURES)
        meta = {}
    return model, features, meta


@st.cache_data(show_spinner=False)
def load_background():
    """Development-cohort covariates: SHAP background + percentile reference."""
    if BACKGROUND_PATH.exists():
        bg = pd.read_csv(BACKGROUND_PATH)
        bg = bg[FEATURES].apply(pd.to_numeric, errors="coerce").dropna()
        if len(bg):
            return bg, True
    return FALLBACK_BACKGROUND.copy(), False


@st.cache_resource(show_spinner="Building the SHAP explainer...")
def build_explainer(_model, _background):
    """
    Interventional TreeSHAP on the probability scale.

    ``max_samples = len(background)`` disables the default 100-row subsampling,
    so E[f(X)] is the mean predicted risk over the whole development cohort
    (38.94% here) instead of the mean over an arbitrary 100-patient subset, which
    would give a different value depending on the row order of the CSV.
    """
    masker = shap.maskers.Independent(_background, max_samples=len(_background))
    return shap.TreeExplainer(_model, data=masker, model_output="probability")


def shap_decomposition(explainer, model, patient):
    """
    Return (contributions, baseline) for the death class on the probability scale.

    CatBoost returns a 2-D matrix ``(n_samples, n_features)`` for a binary
    target, but older releases and some sklearn forest wrappers return one value
    per feature per class, ``(n_samples, n_features, n_classes)``. Both layouts
    are handled, and class 1 (death) is located through ``model.classes_``
    instead of assuming a column position.
    """
    explanation = explainer(patient)
    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)

    classes = list(getattr(model, "classes_", [0, 1]))
    class_index = classes.index(1) if 1 in classes else -1

    if values.ndim == 3:
        contributions = values[0][:, class_index]
    else:
        contributions = np.ravel(values[0])

    if base.ndim == 0:
        baseline = float(base)
    elif base.size == 1:
        baseline = float(np.ravel(base)[0])
    else:
        baseline = float(np.ravel(base)[class_index])

    return np.asarray(contributions, dtype=float), baseline


# ============================================================================
# 3. Page configuration
# ============================================================================
st.set_page_config(
    page_title="ARDS 28-Day Mortality Risk Calculator",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1180px;}
      .hero {padding: 1.1rem 1.4rem; border-radius: 14px;
             background: linear-gradient(120deg, #eef4fb 0%, #f7fbf8 100%);
             border: 1px solid #dbe6f2; margin-bottom: 1.1rem;}
      .hero h1 {font-size: 1.72rem; margin: 0 0 .35rem 0; color: #10243a;}
      .hero p  {margin: 0; color: #40566d; font-size: .95rem; line-height: 1.5;}
      .risk-card {border-radius: 14px; padding: 1.1rem 1.3rem; border: 1px solid;}
      .risk-high {background: #fdf1f1; border-color: #f0c2c2;}
      .risk-low  {background: #eef8f1; border-color: #bfe3cb;}
      .risk-card .pct {font-size: 2.45rem; font-weight: 700; line-height: 1.1;}
      .risk-high .pct {color: #a4262b;}
      .risk-low  .pct {color: #1d6b3c;}
      .risk-card .cap {font-size: .9rem; color: #40566d;}
      footer, #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>ARDS 28-Day Mortality Risk Calculator</h1>
      <p>CatBoost model for invasively ventilated adults with
      moderate-to-severe ARDS. Enter Day-3 respiratory and organ-function
      variables to estimate the individual probability of 28-day mortality,
      together with a SHAP breakdown of how each variable moved that estimate.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# 4. Load artefacts and run integrity checks
# ============================================================================
if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: `{MODEL_PATH.name}`. Upload "
        "`catboost_theta6_model.pkl` to the same folder as this script "
        "(repository root) and reload the page."
    )
    st.stop()

model, MODEL_FEATURES, MODEL_META = load_bundle()
background, background_is_real = load_background()

if sorted(MODEL_FEATURES) != sorted(FEATURES):
    st.error(
        "The variables stored in the model file "
        f"({', '.join(MODEL_FEATURES)}) do not match the four inputs of this "
        f"form ({', '.join(FEATURES)}). Refusing to predict."
    )
    st.stop()

# CatBoost does not validate column names against a fitted schema in this setup,
# so a permuted frame is accepted without any error (only a user warning). The
# patient vector is therefore always built in the order recorded in the .pkl.
FEATURE_ORDER = list(MODEL_FEATURES)

# The decision threshold travels with the artefact; fall back to 0.50.
DECISION_THRESHOLD = float(MODEL_META.get("cutoff", DECISION_THRESHOLD))

_model_versions = MODEL_META.get("versions") or {}
_model_catboost = str(_model_versions.get("catboost") or PINNED_CATBOOST)
_running_version = catboost.__version__
if _running_version != _model_catboost:
    st.warning(
        f"CatBoost version mismatch: the model was fitted with "
        f"{_model_catboost}, the running environment has {_running_version}. "
        "A pickled estimator is not guaranteed to be stable across versions, so "
        "the predicted probabilities may differ slightly. Pin "
        f"`catboost=={_model_catboost}` in requirements.txt to reproduce the "
        "published estimates exactly."
    )

if not background_is_real:
    st.warning(
        "`cohort_features.csv` was not found, so the SHAP baseline is computed on "
        "a small built-in reference table instead of the full 284-patient cohort. "
        "Predicted probabilities are unaffected, but the baseline E[f(X)] and the "
        "individual contributions will differ from the published values."
    )

# ============================================================================
# 5. Inputs
# ============================================================================
for _f in FEATURES:
    st.session_state.setdefault(f"in_{_f}", INPUT_SPEC[_f][3])

st.markdown("#### Patient variables (Day 3)")
col_left, col_right = st.columns(2)


def render_input(column, name):
    lo, hi, step, _default = INPUT_SPEC[name]
    with column:
        value = st.number_input(
            LABELS[name],
            min_value=lo,
            max_value=hi,
            step=step,
            key=f"in_{name}",
            help=CAPTIONS[name],
        )
        percentile = float((background[name] <= value).mean() * 100.0)
        note = f"{CAPTIONS[name]} &middot; {percentile:.0f}th percentile of the cohort"
        if value > background[name].max() or value < background[name].min():
            note += " &middot; <span style='color:#a4262b'>outside cohort range</span>"
        st.markdown(
            f"<div style='font-size:.8rem;color:#5b7189;margin-top:-.6rem'>{note}</div>",
            unsafe_allow_html=True,
        )
    return float(value)


inputs = {}
inputs["d3_a_ado2"] = render_input(col_left, "d3_a_ado2")
inputs["d3_lac"] = render_input(col_right, "d3_lac")
inputs["d3_sofa"] = render_input(col_left, "d3_sofa")
inputs["peep"] = render_input(col_right, "peep")

st.write("")
_, button_col, _ = st.columns([1, 1.1, 1])
with button_col:
    run = st.button("Estimate risk", type="primary", width="stretch")

# ============================================================================
# 6. Prediction and SHAP
# ============================================================================
if run:
    patient = pd.DataFrame(
        [[inputs[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER
    )

    mortality_risk = float(model.predict_proba(patient)[0, 1])
    survival_risk = 1.0 - mortality_risk
    high_risk = mortality_risk >= DECISION_THRESHOLD
    risk_word = "High" if high_risk else "Low"
    card_class = "risk-high" if high_risk else "risk-low"

    st.divider()

    _, card_col, _ = st.columns([1, 1.5, 1])
    with card_col:
        st.markdown(
            f"""
            <div class="risk-card {card_class}">
              <div class="cap">Probability of 28-day mortality</div>
              <div class="pct">{mortality_risk * 100:.1f}%</div>
              <div class="cap">{risk_word} risk at the {DECISION_THRESHOLD:.2f} threshold
              &nbsp;|&nbsp; predicted survival {survival_risk * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    explainer = build_explainer(model, background)
    values, base_value = shap_decomposition(explainer, model, patient)

    explanation = shap.Explanation(
        values=np.round(values * 100.0, 1),
        base_values=base_value * 100.0,
        data=patient.iloc[0].values.astype(float),
        feature_names=[LABELS[f] for f in FEATURE_ORDER],
    )

    fig = plt.figure(figsize=(9.2, 3.4), dpi=150)
    shap.plots.waterfall(explanation, max_display=len(FEATURE_ORDER), show=False)
    fig = plt.gcf()
    fig.patch.set_facecolor("#ffffff")
    if len(fig.axes) > 1:
        fig.axes[1].remove()
    ax = fig.axes[0]
    # The waterfall starts at E[f(X)]; state it explicitly so the two ends of the
    # plot are not confused with each other.
    ax.set_xlabel(
        "Contribution to predicted 28-day mortality (percentage points)\n"
        f"starting point  E[f(X)] = {base_value * 100:.1f}%  "
        "(mean predicted risk of the cohort)",
        fontsize=10,
    )
    plt.tight_layout()

    st.pyplot(fig, clear_figure=True)
    plt.close("all")

    st.caption(
        "Contributions are interventional TreeSHAP values on the probability "
        "scale, expressed in percentage points. E[f(X)] is the mean predicted "
        "risk of the development cohort; E[f(X)] + sum of contributions is "
        "approximately equal to the model output."
    )

# ============================================================================
# 7. Model details and limitations
# ============================================================================
st.write("")
with st.expander("Model details and limitations"):
    n_ref = len(background)
    _saved = MODEL_META.get("saved_at")
    st.markdown(
        f"""
        - Algorithm: CatBoost (depth 6, learning rate 0.06, 200 boosting
          iterations, L2 regularisation 120, random strength 6, 96 border
          bins, `random_seed` = 42, single-threaded)
        - Development cohort: n = 284 (111 deaths, 39.1%); the model was fitted on
          the complete cohort
        - Discrimination: apparent AUROC 0.954; 10-fold cross-validated AUROC
          0.903 (SD 0.085, range 0.687-0.970)
        - Calibration: mean predicted risk 38.9% vs observed mortality 39.1%;
          cross-validated Brier score 0.131
        - Predictors: Day-3 A-aDO2, PEEP, SOFA score, Day-3 lactate
        - Ranked variable importance (CatBoost): PEEP 75.2%, SOFA 11.2%,
          A-aDO2 7.2%, lactate 6.5%
        - Decision threshold: {DECISION_THRESHOLD:.2f}
        - Reference population for SHAP: {'development cohort' if background_is_real else 'built-in fallback'} (n = {n_ref})
        - Model file: `{MODEL_PATH.name}` (fitted with CatBoost {_model_catboost}{'; saved ' + str(_saved) if _saved else ''})
        - Runtime: CatBoost {_running_version}
        """
    )
    st.caption(
        "This is a research tool. It is not a substitute for clinical judgement "
        "and has not been prospectively validated for routine care. Predicted "
        "probabilities are based on a single-centre development cohort and on "
        "Day-3 values only."
    )
