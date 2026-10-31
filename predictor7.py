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
    cohort_features.csv            <- development-cohort covariates, used for the
                                      percentile notes under each input (SHAP
                                      itself needs no background sample)
    requirements.txt

Notes on reproducibility
------------------------
* The pickle stores a CatBoost object, so the runtime ``catboost`` version
  should match the one recorded in the bundle (``versions.catboost``); the app
  reads that field and warns on a mismatch. ``requirements.txt`` pins the
  version.
* SHAP values are computed with CatBoost's own path-dependent TreeSHAP
  (``get_feature_importance(..., type="ShapValues")``) on the raw margin
  (log-odds) scale. This is exactly the routine that produced Figure 7B of the
  manuscript, so the numbers shown here match the figure digit for digit. The
  baseline E[f(X)] is a quantity stored inside the fitted model (-0.754) and
  therefore does not depend on any background sample, on the row order of the
  CSV, or on the ``shap`` release.
* The waterfall is drawn on the log-odds scale, but the logistic transform of
  both ends is displayed underneath it, so a clinician reads probabilities
  (32.0% at the base value, 68.1% at the patient's f(x) in the worked example)
  while the arithmetic stays additive. Note that the cohort's mean predicted
  probability (38.9%) is the average of the individual probabilities and is
  therefore higher than sigma(E[f(X)]) = 32.0%; averaging before and after the
  logistic transform are different summaries, not two different baselines.
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

# Decimals used when the observed value is printed next to each waterfall row,
# kept identical to the figure script (A-aDO2, PEEP and SOFA as integers,
# lactate to one decimal).
UNIT_DEC = {"d3_a_ado2": 0, "peep": 0, "d3_sofa": 0, "d3_lac": 1}

# Same palette as Figure 7: blue lowers the predicted risk, magenta raises it.
SHAP_BLUE = "#1E9BF0"
SHAP_RED = "#E62E7B"

# (min, max, step, default); defaults are the development-cohort medians.
INPUT_SPEC = {
    "d3_a_ado2": (0.0, 620.0, 1.0, 190.6),
    "peep": (0, 30, 1, 6),
    "d3_sofa": (0, 24, 1, 8),
    "d3_lac": (0.1, 20.0, 0.1, 1.4),
}

DECISION_THRESHOLD = 0.50

# Stand-in used only if cohort_features.csv is missing: the app still predicts
# normally and the SHAP values are unchanged, but the percentile notes are then
# computed over these rows instead of the development cohort.
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
    """Development-cohort covariates, used for the percentile reference only."""
    if BACKGROUND_PATH.exists():
        bg = pd.read_csv(BACKGROUND_PATH)
        bg = bg[FEATURES].apply(pd.to_numeric, errors="coerce").dropna()
        if len(bg):
            return bg, True
    return FALLBACK_BACKGROUND.copy(), False


def shap_decomposition(model, patient):
    """
    Return (contributions, baseline) for the death class on the log-odds scale.

    Uses CatBoost's native path-dependent TreeSHAP, i.e. the same routine as
    Figure 7B of the manuscript. Unlike ``shap.TreeExplainer`` with an
    interventional masker, this needs no background sample: the baseline is the
    expected value stored inside the booster (-0.754), so the numbers are
    identical to the figure and are stable across re-uploads, row orders and
    ``shap`` releases.

    CatBoost returns a matrix with one extra trailing column holding the
    baseline; for a binary target the remaining columns refer to the positive
    class (death), so no class indexing is required.
    """
    if hasattr(model, "get_feature_importance"):
        values = np.asarray(
            model.get_feature_importance(
                catboost.Pool(patient), type="ShapValues"
            ),
            dtype=float,
        )
        return values[0, :-1], float(values[0, -1])

    # Fallback for a bare, non-CatBoost estimator: classic path-dependent
    # TreeSHAP. The baseline is then taken from the explainer, not from a masker.
    explanation = shap.TreeExplainer(model)(patient)
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, -1]
    return np.ravel(values[0]), float(np.ravel(explanation.base_values)[0])


def logistic(z):
    """Inverse logit, vectorised and numerically safe for the values used here."""
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def plot_waterfall(ax, base, contrib, values, keys, order, f_x):
    """
    Redraw Figure 7B with the manuscript's own routine.

    This is a straight port of ``panel_waterfall`` from the figure script, so the
    app and the paper show the same bars in the same order, the same two-decimal
    labels, the same true minus sign and the same palette, instead of whatever
    number formatting the installed ``shap`` release happens to use.
    ``order`` lists the feature indices by descending absolute contribution.
    """
    k = len(order)
    rows = np.arange(k)[::-1]          # highest contribution on the top row
    cur = base
    ax.axvline(base, color="#7A7A7A", lw=0.9, ls=(0, (4, 3)), zorder=1)

    for row, j in zip(rows, order):
        contribution = contrib[j]
        nxt = cur + contribution
        left, width = min(cur, nxt), abs(contribution)
        ax.barh(
            row,
            width,
            left=left,
            height=0.52,
            color=SHAP_RED if contribution > 0 else SHAP_BLUE,
            edgecolor="none",
            zorder=3,
        )
        ax.text(
            left + width / 2,
            row + 0.30,
            f"{contribution:+.2f}".replace("-", "\u2212"),
            va="bottom",
            ha="center",
            fontsize=10.5,
            color=SHAP_RED if contribution > 0 else SHAP_BLUE,
            fontweight="bold",
            zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2),
        )
        ax.axhline(row, color="#DDDDDD", lw=0.7, ls=(0, (1, 2)), zorder=1)
        cur = nxt

    ax.set_yticks(rows)
    ax.set_yticklabels(
        [f"{values[j]:.{UNIT_DEC[keys[j]]}f} = {LABELS[keys[j]]}" for j in order],
        fontsize=10.5,
        color="#333333",
    )
    ax.set_ylim(-1.05, k - 0.35)
    ax.tick_params(axis="x", labelsize=10, length=3.5)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#333333")
    ax.set_title("Individual patient", fontsize=11.5, pad=12)

    # Both ends of the waterfall, below the axis as in the published panel.
    ax.text(
        base,
        -0.80,
        f"$E[f(X)]$ = {base:.2f}".replace("-", "\u2212"),
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.6),
    )
    ax.plot([base], [-0.55], marker="v", ms=4, color="#7A7A7A", clip_on=False)
    ax.axvline(f_x, color="#333333", lw=0.8, ls=(0, (2, 2)), zorder=2)
    ax.text(
        f_x,
        -0.80,
        f"$f(x)$ = {f_x:.2f}".replace("-", "\u2212"),
        ha="center",
        va="center",
        fontsize=10.5,
        color="#1A1A1A",
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.6),
    )
    ax.plot([f_x], [-0.55], marker="v", ms=4, color="#333333", clip_on=False)




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
        "`cohort_features.csv` was not found, so the percentile notes under each "
        "input are based on a small built-in reference table instead of the full "
        "284-patient cohort. Predicted probabilities and SHAP values are "
        "unaffected - the SHAP baseline comes from the model itself."
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

    values, base_value = shap_decomposition(model, patient)
    f_x = float(base_value + values.sum())
    base_prob = float(logistic(base_value))
    f_prob = float(logistic(f_x))

    # The cohort's mean predicted probability is the average of the individual
    # probabilities (sigma applied *after* averaging), which is deliberately not
    # the same number as sigma(E[f(X)]). It is used below only to explain why the
    # two summaries differ, never as a SHAP baseline.
    if background_is_real:
        cohort_mean_risk = float(
            model.predict_proba(background[FEATURE_ORDER])[:, 1].mean()
        )
    else:
        cohort_mean_risk = base_prob

    order = np.argsort(-np.abs(values))
    x_lo = min(base_value, f_x) - 0.9
    x_hi = max(base_value, f_x) + 0.9

    fig, ax = plt.subplots(figsize=(8.4, 4.1), dpi=160)
    fig.patch.set_facecolor("#ffffff")
    ax.set_xlim(x_lo, x_hi)
    plot_waterfall(
        ax,
        base_value,
        values,
        patient.iloc[0].values.astype(float),
        FEATURE_ORDER,
        order,
        f_x,
    )
    ax.set_xlabel(
        "Contribution to predicted 28-day mortality (log-odds)\n"
        "the same two ends on the probability scale:  "
        f"$E[f(X)]$ \u2192 {base_prob * 100:.1f}%      "
        f"$f(x)$ \u2192 {f_prob * 100:.1f}%",
        fontsize=9.5,
        labelpad=7,
    )
    plt.tight_layout()

    st.pyplot(fig, clear_figure=True)
    plt.close("all")

    st.markdown(
        f"""
        <div style="font-size:.93rem;line-height:1.85;margin-top:.1rem">
          <span style="color:#5b7189">Probability translation of the same waterfall</span><br>
          &nbsp;&nbsp;base value &nbsp;E[f(X)] = <code>{base_value:.2f}</code>
          &nbsp;&rarr;&nbsp; <b>{base_prob * 100:.1f}%</b>
          &nbsp;<span style="color:#5b7189">(cohort-average risk after the logistic transform)</span><br>
          &nbsp;&nbsp;final output &nbsp;f(x) = <code>{f_x:.2f}</code>
          &nbsp;&rarr;&nbsp; <b>{f_prob * 100:.1f}%</b>
          &nbsp;<span style="color:#5b7189">(predicted 28-day mortality for this patient)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Contributions are path-dependent TreeSHAP values on the CatBoost margin "
        "(log-odds) scale, computed with the model's own "
        "`get_feature_importance(..., type=\"ShapValues\")` and therefore "
        "identical to Figure 7B of the manuscript; they sum exactly to f(x). "
        "Applying the logistic transform to the two ends of the waterfall gives "
        f"the probabilities above ({base_prob * 100:.1f}% and {f_prob * 100:.1f}%). "
        f"The cohort's mean predicted probability is {cohort_mean_risk * 100:.1f}%, "
        "which is the average of the individual probabilities and is therefore "
        f"higher than the probability implied by the mean log-odds "
        f"({base_prob * 100:.1f}%); these are two summaries of the same cohort, "
        "not two competing baselines."
    )

# ============================================================================
# 7. Disclaimer
# ============================================================================
st.write("")
st.caption(
    "This is a research tool. It is not a substitute for clinical judgement and "
    "has not been prospectively validated for routine care. Predicted "
    "probabilities are based on a single-centre development cohort and on Day-3 "
    "values only."
)
