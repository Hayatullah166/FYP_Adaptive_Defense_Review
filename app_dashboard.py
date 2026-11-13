import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import plotly.express as px

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

RISK_MODEL_PATH = BASE_DIR / "models/risk_assessment/xgboost_risk_model.joblib"
RISK_FEATURES_PATH = BASE_DIR / "models/risk_assessment/risk_model_feature_names.joblib"
MLP_CHECKPOINT_PATH = BASE_DIR / "models/mlp/mlp_adversarial_best.pth"
SCALER_PATH = BASE_DIR / "data/processed/grouped/models/standard_scaler_grouped.joblib"

RISK_THRESHOLD = 0.01
MLP_THRESHOLD = 0.9073078036308289
SQUEEZE_DECIMALS = 2

DEVICE = torch.device("cpu")

st.set_page_config(
    page_title="Adaptive NIDS | Cybersecurity",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. PROFESSIONAL INTERFACE
# =========================================================

st.markdown("""
<style>

/* =====================================================
   1. MAIN DASHBOARD
===================================================== */

:root {
    color-scheme: dark;
}

html, body, [data-testid="stAppViewContainer"] {
    background: #101827 !important;
    color: #FFFFFF !important;
}

.stApp {
    background: #101827 !important;
    color: #FFFFFF !important;
}

[data-testid="stHeader"] {
    background: #101827 !important;
}

[data-testid="stSidebar"] {
    background: #0B1220 !important;
    border-right: 1px solid #334155;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* =====================================================
   2. GENERAL TEXT
===================================================== */

h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
    font-weight: 750 !important;
}

.stMarkdown,
.stMarkdown p,
.stMarkdown li,
.stMarkdown label {
    color: #FFFFFF !important;
}

[data-testid="stCaptionContainer"] {
    color: #FFFFFF !important;
}

.section-note {
    color: #FFFFFF !important;
    font-size: 13px;
}

/* =====================================================
   3. HERO BANNER
===================================================== */

.hero {
    background: linear-gradient(110deg, #172554, #123B50);
    border: 1px solid #3B82F6;
    border-radius: 16px;
    padding: 25px 28px;
    margin-bottom: 18px;
}

.hero-title {
    font-size: 30px;
    font-weight: 800;
    color: #FFFFFF !important;
    margin-bottom: 7px;
}

.hero-subtitle {
    color: #FFFFFF !important;
    font-size: 14px;
}

/* =====================================================
   4. STATUS PANEL
===================================================== */

.status {
    background: #12372F;
    border: 1px solid #16A34A;
    color: #FFFFFF !important;
    border-radius: 10px;
    padding: 13px 16px;
    margin: 12px 0 20px;
}

.status * {
    color: #FFFFFF !important;
}

/* =====================================================
   5. FAST / DEEP PATH CARDS
===================================================== */

.path-card {
    background: #1E293B;
    border: 1px solid #475569;
    border-radius: 14px;
    padding: 19px;
    min-height: 140px;
    transition: all 0.2s ease;
}

.path-card:hover {
    border-color: #60A5FA;
    transform: translateY(-3px);
    box-shadow: 0 8px 20px #00000040;
}

.path-card h3,
.path-card p,
.path-card span {
    color: #FFFFFF !important;
}

/* =====================================================
   6. METRIC CARDS
===================================================== */

[data-testid="stMetric"] {
    background: #1E293B !important;
    border: 1px solid #475569 !important;
    padding: 17px !important;
    border-radius: 12px !important;
    transition: all 0.2s ease-in-out;
}

[data-testid="stMetric"]:hover {
    border-color: #60A5FA !important;
    transform: translateY(-2px);
    box-shadow: 0 5px 18px #00000040;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #FFFFFF !important;
}

/* =====================================================
   7. BUTTONS
===================================================== */

.stButton > button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: 1px solid #60A5FA !important;
    border-radius: 9px !important;
    min-height: 42px !important;
    font-weight: 700 !important;
    opacity: 1 !important;
    transition: all 0.2s ease;
}

.stButton > button p,
.stButton > button span,
.stButton > button div {
    color: #FFFFFF !important;
}

.stButton > button:disabled {
    background-color: #334155 !important;
    color: #FFFFFF !important;
    border-color: #475569 !important;
    opacity: 1 !important;
}

.stButton > button:hover:not(:disabled) {
    background-color: #1D4ED8 !important;
    border-color: #93C5FD !important;
    transform: translateY(-2px);
}

/* =====================================================
   8. INPUTS AND TEXT AREAS
   White background = Black font
===================================================== */

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    border: 1px solid #000000 !important;
}

[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stTextArea"] label {
    color: #FFFFFF !important;
}

/* =====================================================
   9. SELECT BOXES
===================================================== */

[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label {
    color: #FFFFFF !important;
}

[data-baseweb="select"] > div {
    background-color: #1E293B !important;
    color: #FFFFFF !important;
    border-color: #64748B !important;
}

[data-baseweb="select"] input,
[data-baseweb="select"] span {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Dropdown popup - White background, black text */

[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}

[role="option"],
[role="option"] * {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

[role="option"]:hover,
[role="option"]:hover * {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* =====================================================
   10. SLIDERS
===================================================== */

[data-testid="stSlider"] label,
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #FFFFFF !important;
}

/* =====================================================
   11. FILE UPLOADER
===================================================== */

[data-testid="stFileUploader"] {
    background: #1E293B !important;
    border: 1px dashed #64748B !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #1E293B !important;
    border: 1px dashed #60A5FA !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: #FFFFFF !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: 1px solid #60A5FA !important;
}

[data-testid="stFileUploaderDropzone"] button * {
    color: #FFFFFF !important;
}

/* =====================================================
   12. EXPANDERS
===================================================== */

[data-testid="stExpander"] {
    background: #1E293B !important;
    border: 1px solid #475569 !important;
    border-radius: 10px !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    color: #FFFFFF !important;
}

/* =====================================================
   13. DATAFRAMES AND ALERTS
===================================================== */

[data-testid="stDataFrame"] {
    border: 1px solid #475569;
    border-radius: 10px;
}

[data-testid="stAlert"] {
    border-radius: 10px;
}

hr {
    border-color: #475569 !important;
}

/* =====================================================
   14. MATHEMATICAL CONTENT
===================================================== */

[data-testid="stMarkdownContainer"] .katex {
    color: #FFFFFF !important;
}

/* =====================================================
   15. PRINT / PDF
===================================================== */

@media print {
    * {
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
    }
}


/* =====================================================
   FINAL FONT CONTRAST FIX
   White backgrounds = BLACK text
   Dark / Blue backgrounds = WHITE text
   Background colors remain unchanged
===================================================== */

/* Dark dashboard text */
.stApp,
.stApp p,
.stApp li,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6,
.stApp label,
.stApp .stMarkdown,
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricDelta"],
.stApp [data-testid="stDataFrame"] {
    color: #FFFFFF !important;
}

/* Plotly chart text, axis labels, legends and annotations */
.js-plotly-plot .plotly text,
.js-plotly-plot .plotly .xtick text,
.js-plotly-plot .plotly .ytick text,
.js-plotly-plot .plotly .legend text,
.js-plotly-plot .plotly .gtitle,
.js-plotly-plot .plotly .annotation-text {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
}


/* BUTTON FONT CONTRAST ? PRESERVE BUTTON BACKGROUNDS */

/* Blue / dark buttons */
.stButton > button,
.stButton > button *,
.stDownloadButton > button,
.stDownloadButton > button * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* White buttons, if present */
.stButton > button[style*="background-color: white"],
.stButton > button[style*="background-color: rgb(255, 255, 255)"],
.stDownloadButton > button[style*="background-color: white"],
.stDownloadButton > button[style*="background-color: rgb(255, 255, 255)"] {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* White text-area background = black text */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* White dropdown background = black text */
[role="listbox"],
[role="option"],
[role="option"] *,
[data-baseweb="popover"] [role="option"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* Dropdown hover: black background = white text */
[role="option"]:hover,
[role="option"]:hover * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Blue and dark buttons = white text */
.stButton > button,
.stButton > button *,
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: #FFFFFF !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. MLP MODEL ARCHITECTURE
# =========================================================

class MLP(nn.Module):

    def __init__(self, input_features):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


# =========================================================
# 4. LOAD TRAINED ARTIFACTS
# =========================================================

@st.cache_resource
def load_models():

    required = [
        RISK_MODEL_PATH,
        RISK_FEATURES_PATH,
        MLP_CHECKPOINT_PATH,
        SCALER_PATH
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing model artifact: {path}"
            )

    risk_model = joblib.load(RISK_MODEL_PATH)
    risk_features = list(joblib.load(RISK_FEATURES_PATH))
    scaler = joblib.load(SCALER_PATH)

    try:
        checkpoint = torch.load(
            MLP_CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            MLP_CHECKPOINT_PATH,
            map_location=DEVICE
        )

    mlp_features = list(checkpoint["feature_names"])

    if risk_features != mlp_features:
        raise ValueError(
            "XGBoost and MLP feature order does not match."
        )

    if len(risk_features) != 68:
        raise ValueError(
            f"Expected 68 features, found {len(risk_features)}."
        )

    mlp_model = MLP(len(mlp_features)).to(DEVICE)
    mlp_model.load_state_dict(checkpoint["model_state_dict"])
    mlp_model.eval()

    return risk_model, scaler, mlp_model, risk_features


# =========================================================
# 5. ADAPTIVE E3 PIPELINE
# =========================================================

def run_pipeline(
    uploaded_df,
    risk_model,
    scaler,
    mlp_model,
    features
):

    start_total = time.perf_counter()

    df = uploaded_df.copy()

    if len(df) == 0:
        raise ValueError("Uploaded CSV is empty.")

    # Ground truth is optional.
    y_true = None

    if "Label" in df.columns:

        labels = pd.to_numeric(
            df["Label"],
            errors="coerce"
        )

        if labels.isna().any():
            raise ValueError(
                "Label column contains invalid values."
            )

        # Validate before converting to integers.
        if not labels.isin([0, 1]).all():
            raise ValueError(
                "Label must contain only exact binary values 0 and 1."
            )

        y_true = labels.to_numpy(dtype=np.int32)

    # Check required features.
    missing = [
        feature for feature in features
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing[:15])
        )

    # Preserve exact training feature order.
    X_raw_df = df[features].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if X_raw_df.isna().any().any():
        raise ValueError(
            "Feature columns contain missing or non-numeric values."
        )

    X_raw = X_raw_df.to_numpy(dtype=np.float32)

    if not np.isfinite(X_raw).all():
        raise ValueError(
            "Input contains NaN or infinite values."
        )

    # -----------------------------------------------------
    # XGBoost risk assessment
    # -----------------------------------------------------

    risk_start = time.perf_counter()

    risk_probabilities = risk_model.predict_proba(
        pd.DataFrame(X_raw, columns=features)
    )[:, 1]

    risk_seconds = time.perf_counter() - risk_start

    predicted_class = (
        risk_probabilities >= 0.5
    ).astype(np.int32)

    # Uncertainty = 1 - maximum class probability.
    uncertainty = np.minimum(
        risk_probabilities,
        1.0 - risk_probabilities
    )

    # -----------------------------------------------------
    # Adaptive Fast / Deep routing
    # -----------------------------------------------------

    fast_mask = (
        (predicted_class == 0) &
        (uncertainty < RISK_THRESHOLD)
    )

    deep_mask = ~fast_mask

    fast_indices = np.flatnonzero(fast_mask)
    deep_indices = np.flatnonzero(deep_mask)

    final_predictions = predicted_class.copy()
    final_probabilities = risk_probabilities.copy()

    # -----------------------------------------------------
    # Feature scaling
    # -----------------------------------------------------

    scale_start = time.perf_counter()

    X_scaled = scaler.transform(X_raw).astype(np.float32)

    scaling_seconds = time.perf_counter() - scale_start

    if not np.isfinite(X_scaled).all():
        raise ValueError(
            "Scaled features contain invalid values."
        )

    # -----------------------------------------------------
    # Deep Path: Feature squeezing + adversarial MLP
    # -----------------------------------------------------

    deep_seconds = 0.0
    deep_probabilities = np.array([], dtype=np.float32)

    if len(deep_indices) > 0:

        X_deep = X_scaled[deep_indices]

        X_squeezed = np.round(
            X_deep,
            decimals=SQUEEZE_DECIMALS
        ).astype(np.float32)

        deep_tensor = torch.tensor(
            X_squeezed,
            dtype=torch.float32
        )

        deep_start = time.perf_counter()

        with torch.no_grad():

            logits = mlp_model(deep_tensor)

            deep_probabilities = torch.sigmoid(
                logits
            ).cpu().numpy().reshape(-1)

        deep_seconds = time.perf_counter() - deep_start

        deep_predictions = (
            deep_probabilities >= MLP_THRESHOLD
        ).astype(np.int32)

        final_predictions[deep_indices] = deep_predictions
        final_probabilities[deep_indices] = deep_probabilities

    total_seconds = time.perf_counter() - start_total

    # -----------------------------------------------------
    # Output results
    # -----------------------------------------------------

    results = df.copy()

    results["XGBoost_Attack_Probability"] = risk_probabilities
    results["Uncertainty_Risk"] = uncertainty

    results["Route"] = np.where(
        fast_mask,
        "Fast",
        "Deep"
    )

    results["MLP_Attack_Probability"] = np.nan
    results["Final_Attack_Probability"] = final_probabilities
    results["Prediction"] = final_predictions

    results["Prediction_Label"] = np.where(
        final_predictions == 1,
        "Attack",
        "Benign"
    )

    if len(deep_indices) > 0:

        results.iloc[
            deep_indices,
            results.columns.get_loc("MLP_Attack_Probability")
        ] = deep_probabilities

    timing = {
        "total_seconds": total_seconds,
        "risk_seconds": risk_seconds,
        "scaling_seconds": scaling_seconds,
        "deep_seconds": deep_seconds,
        "records_per_second": (
            len(df) / total_seconds
            if total_seconds > 0 else 0
        )
    }

    return results, y_true, timing


# =========================================================
# 6. METRICS
# =========================================================

def calculate_metrics(y_true, y_pred):

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_true, y_pred, zero_division=0
        )
    }


# =========================================================
# 7. HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">🛡️ Adaptive Risk-Aware NIDS</div>
    <div class="hero-subtitle">
        Machine Learning-Based Network Intrusion Detection
        with Adaptive Fast / Deep Defense
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="status">
    ● OFFLINE DATASET REPLAY &nbsp; | &nbsp;
    Experimental flow-record evaluation &nbsp; | &nbsp;
    Not live packet capture
</div>
""", unsafe_allow_html=True)


# =========================================================
# 8. SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🛡️ System Panel")

    st.caption("ADAPTIVE DEFENSE CONFIGURATION")

    st.metric("Input Features", "68")
    st.metric("Risk Threshold", f"{RISK_THRESHOLD:.3f}")
    st.metric("MLP Threshold", f"{MLP_THRESHOLD:.4f}")
    st.metric("Feature Squeezing", f"{SQUEEZE_DECIMALS} decimals")

    st.divider()

    st.markdown("**Risk Engine**")
    st.write("XGBoost")

    st.markdown("**Deep Classifier**")
    st.write("Adversarial MLP")

    st.markdown("**Execution Device**")
    st.write("CPU")

    st.divider()

    st.caption(
        "FYP Prototype | CICIDS-2018 Flow Records"
    )


# =========================================================
# 9. LOAD MODELS
# =========================================================

try:

    with st.spinner(
        "Loading trained models and preprocessing artifacts..."
    ):

        risk_model, scaler, mlp_model, FEATURES = load_models()

    st.success(
        "All trained models and preprocessing artifacts loaded."
    )

except Exception as error:

    st.error(f"Model loading failed: {error}")
    st.stop()


# =========================================================
# 10. DATASET UPLOAD
# =========================================================

st.header("01 | Dataset Input")

upload_col, info_col = st.columns([1.5, 1])

with upload_col:

    uploaded_file = st.file_uploader(
        "Upload network-flow CSV dataset",
        type=["csv"]
    )

with info_col:

    st.markdown("""
    <div class="path-card">
        <h3>Dataset Requirements</h3>
        <p>• 68 trained network-flow features</p>
        <p>• Label column is optional</p>
        <p>• Label format: 0 = Benign, 1 = Attack</p>
        <p>• Missing and infinite feature values are rejected</p>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file is None:

    st.info(
        "Upload your prepared validation dataset to begin."
    )

    st.code(
        "data/processed/grouped/validation_lgbm_grouped.csv"
    )

    st.stop()


try:

    uploaded_df = pd.read_csv(uploaded_file)

    if uploaded_df.empty:
        st.error("The uploaded CSV has no records.")
        st.stop()

    st.subheader("Dataset Preview")

    st.dataframe(
        uploaded_df.head(8),
        width="stretch",
        hide_index=True
    )

    d1, d2, d3 = st.columns(3)

    d1.metric("Uploaded Records", f"{len(uploaded_df):,}")
    d2.metric("CSV Columns", f"{len(uploaded_df.columns):,}")

    d3.metric(
        "Required Features",
        f"{sum(f in uploaded_df.columns for f in FEATURES)}/68"
    )

except Exception as error:

    st.error(f"CSV reading failed: {error}")
    st.stop()


# =========================================================
# 11. RUN INFERENCE
# =========================================================

if st.button(
    "🚀 RUN ADAPTIVE DEFENSE",
    type="primary",
    width="stretch"
):

    try:

        with st.spinner(
            "Assessing risk, routing records and executing Deep Defense..."
        ):

            results, y_true, timing = run_pipeline(
                uploaded_df,
                risk_model,
                scaler,
                mlp_model,
                FEATURES
            )

        st.session_state["results"] = results
        st.session_state["labels"] = y_true
        st.session_state["timing"] = timing
        st.session_state["source_name"] = uploaded_file.name

        # Reset replay controls for the new dataset.
        st.session_state["replay_position"] = 0
        st.session_state["replay_slider"] = 0
        st.session_state["replay_filter"] = "Deep Path Only"

        st.success("Adaptive defense evaluation completed!")

    except Exception as error:

        st.error(f"Inference failed: {error}")
        st.stop()


# =========================================================
# 12. DISPLAY RESULTS
# =========================================================

if "results" not in st.session_state:
    st.stop()

results = st.session_state["results"]
y_true = st.session_state["labels"]
timing = st.session_state["timing"]

total = len(results)

if total == 0:
    st.error("No records were processed.")
    st.stop()

fast_count = int((results["Route"] == "Fast").sum())
deep_count = int((results["Route"] == "Deep").sum())

fast_share = fast_count / total * 100
deep_share = deep_count / total * 100

predicted_attacks = int(results["Prediction"].sum())
predicted_benign = total - predicted_attacks


# =========================================================
# 13. ROUTING OVERVIEW
# =========================================================

st.divider()
st.header("02 | Adaptive Routing Overview")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Records", f"{total:,}")
k2.metric("Fast Path", f"{fast_count:,}", f"{fast_share:.2f}%")
k3.metric("Deep Path", f"{deep_count:,}", f"{deep_share:.2f}%")
k4.metric("Predicted Attacks", f"{predicted_attacks:,}")

st.markdown("#### Two-Path Defense Architecture")

path1, arrow, path2 = st.columns([5, 1, 5])

with path1:

    st.markdown(f"""
    <div class="path-card">
        <h3>⚡ FAST PATH</h3>
        <h2>{fast_count:,} Records</h2>
        <p>High-confidence benign records.</p>
        <p>XGBoost decision is retained. Deep MLP is bypassed.</p>
    </div>
    """, unsafe_allow_html=True)

with arrow:

    st.markdown(
        "<div style='text-align:center; padding-top:45px; "
        "font-size:30px; color:#FFFFFF;'>⇄</div>",
        unsafe_allow_html=True
    )

with path2:

    st.markdown(f"""
    <div class="path-card">
        <h3>🛡️ DEEP DEFENSE PATH</h3>
        <h2>{deep_count:,} Records</h2>
        <p>Suspicious or uncertain records.</p>
        <p>Feature squeezing followed by adversarially trained MLP.</p>
    </div>
    """, unsafe_allow_html=True)


chart1, chart2 = st.columns(2)

with chart1:

    st.subheader("Fast / Deep Distribution")

    route_counts = pd.DataFrame({
        "Route": ["Fast Path", "Deep Path"],
        "Records": [fast_count, deep_count]
    })

    fig = px.pie(
        route_counts,
        names="Route",
        values="Records",
        hole=0.55,
        color="Route",
        color_discrete_map={
            "Fast Path": "#22c55e",
            "Deep Path": "#f59e0b"
        }
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20, l=10, r=10)
    )

    st.plotly_chart(fig, width="stretch")

with chart2:

    st.subheader("Final Classification")

    prediction_counts = pd.DataFrame({
        "Prediction": ["Benign", "Attack"],
        "Records": [predicted_benign, predicted_attacks]
    })

    fig = px.bar(
        prediction_counts,
        x="Prediction",
        y="Records",
        color="Prediction",
        text="Records",
        color_discrete_map={
            "Benign": "#38bdf8",
            "Attack": "#ef4444"
        }
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False
    )

    # Highlight the Attack bar without changing its actual value.
    fig.add_scatter(
        x=["Attack"],
        y=[predicted_attacks],
        mode="markers",
        marker=dict(
            symbol="diamond",
            size=14,
            color="#ef4444",
            line=dict(color="#FFFFFF", width=2)
        ),
        name="Attack Count Marker",
        showlegend=False,
        hovertemplate="Attack Records: %{y:,}<extra></extra>"
    )

    st.plotly_chart(fig, width="stretch")


# =========================================================

# =========================================================
# 13A. ROUTING THRESHOLD SWEEP — GATE-ONLY ANALYSIS
# =========================================================

st.divider()
st.header("02A | Routing Threshold Sensitivity Analysis")

st.caption(
    "This analysis evaluates how different uncertainty thresholds "
    "change Fast/Deep routing on the currently uploaded dataset. "
    "It does not rerun the MLP or modify the predictions already generated."
)

required_sweep_columns = [
    "XGBoost_Attack_Probability",
    "Uncertainty_Risk"
]

if all(column in results.columns for column in required_sweep_columns):

    sweep_probability = results[
        "XGBoost_Attack_Probability"
    ].to_numpy(dtype=float)

    sweep_uncertainty = results[
        "Uncertainty_Risk"
    ].to_numpy(dtype=float)

    sweep_xgb_prediction = (
        sweep_probability >= 0.5
    ).astype(int)

    threshold_values = sorted(set([
        0.0001,
        0.001,
        0.005,
        0.01,
        0.02,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.49,
        float(RISK_THRESHOLD)
    ]))

    sweep_rows = []

    labels_available = (
        y_true is not None
        and len(y_true) == len(results)
    )

    if labels_available:
        sweep_labels = np.asarray(y_true).astype(int)
        total_malicious = int((sweep_labels == 1).sum())
        total_benign = int((sweep_labels == 0).sum())

    for threshold in threshold_values:

        sweep_fast_mask = (
            (sweep_xgb_prediction == 0)
            & (sweep_uncertainty < threshold)
        )

        sweep_deep_mask = ~sweep_fast_mask

        fast_records = int(sweep_fast_mask.sum())
        deep_records = int(sweep_deep_mask.sum())

        row = {
            "Threshold": threshold,
            "Fast Records": fast_records,
            "Deep Records": deep_records,
            "Fast Path (%)": (
                fast_records / len(results) * 100
            ),
            "Deep Path (%)": (
                deep_records / len(results) * 100
            )
        }

        if labels_available:

            malicious_fast = int(
                (
                    sweep_fast_mask
                    & (sweep_labels == 1)
                ).sum()
            )

            benign_deep = int(
                (
                    sweep_deep_mask
                    & (sweep_labels == 0)
                ).sum()
            )

            attack_detection_routing = (
                (
                    total_malicious - malicious_fast
                ) / total_malicious * 100
                if total_malicious > 0
                else 0.0
            )

            row["Malicious in Fast"] = malicious_fast
            row["Benign in Deep"] = benign_deep
            row["Malicious Routed Deep (%)"] = (
                attack_detection_routing
            )

        sweep_rows.append(row)

    threshold_sweep_df = pd.DataFrame(sweep_rows)

    current_row = threshold_sweep_df[
        np.isclose(
            threshold_sweep_df["Threshold"],
            RISK_THRESHOLD
        )
    ]

    if not current_row.empty:

        current_fast = int(
            current_row.iloc[0]["Fast Records"]
        )

        current_deep = int(
            current_row.iloc[0]["Deep Records"]
        )

        current_fast_percentage = float(
            current_row.iloc[0]["Fast Path (%)"]
        )

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Current Risk Threshold",
            f"{RISK_THRESHOLD:.4f}"
        )

        m2.metric(
            "Fast Records at Current Threshold",
            f"{current_fast:,}"
        )

        m3.metric(
            "Fast Path Share",
            f"{current_fast_percentage:.2f}%"
        )

    threshold_chart = px.line(
        threshold_sweep_df,
        x="Threshold",
        y=["Fast Path (%)", "Deep Path (%)"],
        markers=True,
        title="Fast and Deep Routing Share Across Thresholds"
    )

    threshold_chart.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Uncertainty Threshold",
        yaxis_title="Records (%)",
        legend_title="Routing Path",
        hovermode="x unified"
    )

    threshold_chart.add_vline(
        x=RISK_THRESHOLD,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="Current Threshold",
        annotation_position="top"
    )

    st.plotly_chart(
        threshold_chart,
        width="stretch"
    )

    if labels_available:

        leakage_chart = px.line(
            threshold_sweep_df,
            x="Threshold",
            y="Malicious Routed Deep (%)",
            markers=True,
            title="Malicious Records Routed to Deep Path"
        )

        leakage_chart.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Uncertainty Threshold",
            yaxis_title="Malicious Records Routed Deep (%)",
            yaxis=dict(range=[0, 105])
        )

        leakage_chart.add_vline(
            x=RISK_THRESHOLD,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text="Current Threshold"
        )

        st.plotly_chart(
            leakage_chart,
            width="stretch"
        )

        st.caption(
            "Malicious Routed Deep (%) measures the proportion "
            "of ground-truth malicious records sent to Deep Path. "
            "It is a routing metric, not final classification recall."
        )

    else:

        st.info(
            "Ground-truth Label is unavailable. "
            "Malicious routing analysis cannot be calculated. "
            "Fast/Deep distribution is still displayed."
        )

    st.subheader("Threshold Sweep Results")

    display_sweep_df = threshold_sweep_df.copy()

    display_sweep_df["Threshold"] = (
        display_sweep_df["Threshold"].map(
            lambda value: f"{value:.4f}"
        )
    )

    percentage_columns = [
        "Fast Path (%)",
        "Deep Path (%)",
        "Malicious Routed Deep (%)"
    ]

    for column in percentage_columns:

        if column in display_sweep_df.columns:

            display_sweep_df[column] = (
                display_sweep_df[column].map(
                    lambda value: (
                        f"{value:.2f}%"
                        if pd.notna(value)
                        else "Not reported"
                    )
                )
            )

    st.dataframe(
        display_sweep_df,
        width="stretch",
        hide_index=True
    )

else:

    st.warning(
        "Threshold sweep unavailable. Required XGBoost "
        "probability or uncertainty columns are missing."
    )

# 14. CLASSIFICATION EVALUATION
# =========================================================

st.divider()
st.header("03 | Classification Evaluation")

if y_true is not None:

    overall_metrics = calculate_metrics(
        y_true,
        results["Prediction"].to_numpy()
    )

    metric_cols = st.columns(4)

    for col, (name, value) in zip(
        metric_cols,
        overall_metrics.items()
    ):

        col.metric(
            name,
            f"{value * 100:.2f}%"
        )

    st.caption(
        "Metrics are calculated against the uploaded Label column. "
        "These results describe this uploaded dataset, not independent test performance."
    )

    st.subheader("Per-Route Performance")

    route_metrics = []

    for route in ["Fast", "Deep"]:

        mask = results["Route"].to_numpy() == route

        if mask.sum() == 0:
            continue

        route_result = calculate_metrics(
            y_true[mask],
            results["Prediction"].to_numpy()[mask]
        )

        route_result["Route"] = route
        route_result["Records"] = int(mask.sum())

        route_metrics.append(route_result)

    st.dataframe(
        pd.DataFrame(route_metrics),
        width="stretch",
        hide_index=True
    )

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_true,
        results["Prediction"].to_numpy(),
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    cm_col, detail_col = st.columns([1.4, 1])

    with cm_col:

        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted Benign", "Predicted Attack"],
            y=["Actual Benign", "Actual Attack"],
            color_continuous_scale="Blues",
            aspect="auto"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, width="stretch")

    with detail_col:

        st.metric("True Negatives", f"{tn:,}")
        st.metric("False Positives", f"{fp:,}")
        st.metric("False Negatives", f"{fn:,}")
        st.metric("True Positives", f"{tp:,}")

else:

    st.warning(
        "No Label column was provided. Classification metrics "
        "cannot be calculated, but routing and predictions are available."
    )


# =========================================================
# 15. PROCESSING EFFICIENCY
# =========================================================

st.divider()
st.header("04 | Processing Efficiency")

e1, e2, e3, e4 = st.columns(4)

e1.metric(
    "Total Processing",
    f"{timing['total_seconds']:.4f} s"
)

e2.metric(
    "XGBoost Inference",
    f"{timing['risk_seconds']:.4f} s"
)

e3.metric(
    "Deep MLP Inference",
    f"{timing['deep_seconds']:.4f} s"
)

e4.metric(
    "Batch Throughput (k records/s)",
    f"{timing['records_per_second'] / 1000:,.2f}"
)

st.caption(
    "These measurements describe this dashboard batch run. "
    "They are not repeated benchmark results or single-record latency measurements."
)


# =========================================================
# 16. INTERACTIVE RECORD PROCESSING REPLAY
# =========================================================

st.divider()
st.header("05 | Network Flow Processing Replay")

st.write(
    "Follow one uploaded network-flow record through "
    "the complete Adaptive Defense pipeline."
)

route_filter = st.selectbox(
    "Select records to inspect",
    ["All Records", "Fast Path Only", "Deep Path Only"],
    key="replay_filter"
)

if route_filter == "Deep Path Only":

    available_indices = np.flatnonzero(
        results["Route"].to_numpy() == "Deep"
    )

elif route_filter == "Fast Path Only":

    available_indices = np.flatnonzero(
        results["Route"].to_numpy() == "Fast"
    )

else:

    available_indices = np.arange(total)


if len(available_indices) == 0:

    st.info("No records available in this route.")

else:

    current_position = st.session_state.get(
        "replay_position", 0
    )

    current_position = min(
        max(current_position, 0),
        len(available_indices) - 1
    )

    # Keep slider and navigation position synchronized.
    st.session_state["replay_position"] = current_position
    st.session_state["replay_slider"] = current_position

    nav1, nav2, nav3 = st.columns([1, 4, 1])

    with nav1:

        if st.button(
            "⬅ Previous",
            disabled=current_position == 0,
            width="stretch"
        ):

            new_position = max(0, current_position - 1)

            st.session_state["replay_position"] = new_position
            st.session_state["replay_slider"] = new_position

            st.rerun()

    with nav3:

        if st.button(
            "Next ➡",
            disabled=current_position >= len(available_indices) - 1,
            width="stretch"
        ):

            new_position = min(
                len(available_indices) - 1,
                current_position + 1
            )

            st.session_state["replay_position"] = new_position
            st.session_state["replay_slider"] = new_position

            st.rerun()

    with nav2:

        selected_position = st.slider(
            "Record position",
            min_value=0,
            max_value=len(available_indices) - 1,
            key="replay_slider"
        )

        st.session_state["replay_position"] = selected_position

    selected_idx = int(available_indices[selected_position])

    record = results.iloc[selected_idx]

    # -----------------------------------------------------
    # STEP 1: INPUT FLOW RECORD
    # -----------------------------------------------------

    st.divider()
    st.subheader("STEP 1 — Incoming Network Flow")

    st.metric(
        "Record Number",
        f"{selected_idx + 1:,} / {total:,}"
    )

    st.write(
        "This record contains 68 network-flow features. "
        "These are the exact feature columns used by our trained models."
    )

    raw_values = uploaded_df.iloc[selected_idx][FEATURES]

    raw_feature_df = pd.DataFrame({
        "Feature": FEATURES,
        "Raw Input Value": raw_values.to_numpy()
    })

    st.caption("Actual input features received from the uploaded CSV.")

    with st.expander(
        "Inspect all 68 input features",
        expanded=True
    ):

        st.dataframe(
            raw_feature_df,
            width="stretch",
            height=400,
            hide_index=True
        )

    # -----------------------------------------------------
    # STEP 2: XGBOOST RISK ENGINE
    # -----------------------------------------------------

    st.divider()
    st.subheader("STEP 2 — XGBoost Risk Assessment")

    attack_probability = float(
        record["XGBoost_Attack_Probability"]
    )

    benign_probability = 1.0 - attack_probability

    predicted_class = int(attack_probability >= 0.5)

    uncertainty = float(record["Uncertainty_Risk"])

    st.write("The XGBoost model evaluates the 68 raw features.")

    x1, x2, x3 = st.columns(3)

    x1.metric(
        "Benign Probability",
        f"{benign_probability:.6f}"
    )

    x2.metric(
        "Attack Probability",
        f"{attack_probability:.6f}"
    )

    x3.metric(
        "Uncertainty / Risk",
        f"{uncertainty:.6f}"
    )

    st.markdown("**Risk calculation used by the system:**")

    st.latex(r"R(x)=1-\max_c P(y=c\mid x)")

    st.latex(r"R(x)=\min(P(Attack),P(Benign))")

    st.code(
        f"""
Attack Probability = {attack_probability:.8f}
Benign Probability = {benign_probability:.8f}

Risk R(x) = min(
    {attack_probability:.8f},
    {benign_probability:.8f}
)

Risk Threshold = {RISK_THRESHOLD}

Predicted Class = {predicted_class}
        """.strip(),
        language="text"
    )

    # -----------------------------------------------------
    # STEP 3: ROUTING DECISION
    # -----------------------------------------------------

    st.divider()
    st.subheader("STEP 3 — Adaptive Routing Decision")

    st.markdown("**Fast Path condition:**")

    st.latex(r"\hat y=0 \quad \text{AND} \quad R(x)<\tau")

    st.write(
        f"Predicted benign: {predicted_class == 0}"
    )

    st.write(
        f"Risk below threshold: "
        f"{uncertainty:.6f} < {RISK_THRESHOLD}"
    )

    if record["Route"] == "Fast":

        st.success(
            "FAST PATH SELECTED — "
            "The record is predicted benign with sufficiently "
            "low uncertainty. Deep MLP is bypassed."
        )

        st.info(
            "Final prediction is retained from the XGBoost Risk Engine."
        )

    else:

        st.warning(
            "DEEP DEFENSE PATH SELECTED — "
            "The record is suspicious, predicted as an attack, "
            "or its uncertainty is above the threshold."
        )

    # -----------------------------------------------------
    # STEP 4: DEEP DEFENSE
    # -----------------------------------------------------

    st.divider()
    st.subheader("STEP 4 — Deep Defense Processing")

    if record["Route"] == "Deep":

        st.write("The Deep Path applies the following operations:")

        st.markdown(
            "**A. Feature Scaling** — raw features are transformed "
            "using the saved training StandardScaler."
        )

        selected_raw = uploaded_df.iloc[
            selected_idx
        ][FEATURES].to_numpy(dtype=np.float32).reshape(1, -1)

        selected_scaled = scaler.transform(
            selected_raw
        ).astype(np.float32)

        st.markdown(
            "**B. Feature Squeezing** — scaled values are rounded "
            f"to {SQUEEZE_DECIMALS} decimal places."
        )

        selected_squeezed = np.round(
            selected_scaled,
            decimals=SQUEEZE_DECIMALS
        ).astype(np.float32)

        transform_df = pd.DataFrame({
            "Feature": FEATURES,
            "Raw Value": selected_raw.reshape(-1),
            "Scaled Value": selected_scaled.reshape(-1),
            "Squeezed Value": selected_squeezed.reshape(-1)
        })

        with st.expander(
            "Inspect scaling and feature squeezing for all 68 features"
        ):

            st.dataframe(
                transform_df,
                width="stretch",
                height=400,
                hide_index=True
            )

        st.markdown(
            "**C. Adversarially Trained MLP Classifier**"
        )

        mlp_probability = float(
            record["MLP_Attack_Probability"]
        )

        st.metric(
            "MLP Attack Probability",
            f"{mlp_probability:.6f}"
        )

        st.code(
            f"""
MLP Threshold = {MLP_THRESHOLD:.8f}
MLP Attack Probability = {mlp_probability:.8f}

Final MLP Decision =
{"ATTACK" if mlp_probability >= MLP_THRESHOLD else "BENIGN"}
            """.strip(),
            language="text"
        )

        st.success(
            "Deep Defense processing was executed for this record."
        )

    else:

        st.info(
            "Deep Defense was not executed because this record "
            "qualified for the Fast Path."
        )

    # -----------------------------------------------------
    # STEP 5: FINAL DECISION
    # -----------------------------------------------------

    st.divider()
    st.subheader("STEP 5 — Final Classification")

    final_prediction = record["Prediction_Label"]

    if final_prediction == "Attack":

        st.error(f"FINAL RESULT: {final_prediction}")

    else:

        st.success(f"FINAL RESULT: {final_prediction}")

    f1_col, f2_col, f3_col = st.columns(3)

    f1_col.metric("Selected Route", record["Route"])
    f2_col.metric("Final Prediction", final_prediction)

    f3_col.metric(
        "Final Attack Probability",
        f"{record['Final_Attack_Probability']:.6f}"
    )

    if y_true is not None:

        actual_label = (
            "Attack" if y_true[selected_idx] == 1
            else "Benign"
        )

        st.metric("Actual Dataset Label", actual_label)

        st.caption(
            "Actual label is ground truth from the uploaded dataset; "
            "it is not used by the routing decision."
        )

    st.progress(
        (selected_position + 1) / len(available_indices)
    )

    st.caption(
        f"Offline replay position: "
        f"{selected_position + 1} of {len(available_indices)} "
        f"selected records."
    )


# =========================================================
# 17. EXPERIMENTAL RESULTS — E1 vs E2 vs E3
# =========================================================

st.divider()
st.header("06 | Experimental Results — E1, E2 and E3")

st.caption(
    "Experimental findings from baseline detection, adversarial "
    "training, feature squeezing and adaptive risk-aware defense."
)

# ---------------------------------------------------------
# A. CLEAN PERFORMANCE SUMMARY
# ---------------------------------------------------------

st.subheader("A. Clean Classification Performance")

clean_results = pd.DataFrame([
    {
        "Experiment": "E1 — Baseline MLP",
        "Dataset": "Validation",
        "Accuracy (%)": 99.9391,
        "Precision (%)": 71.05,
        "Recall (%)": 100.00,
        "F1 (%)": 83.08,
        "PR-AUC": 0.877838
    },
    {
        "Experiment": "E2 — Adversarial MLP",
        "Dataset": "Validation",
        "Accuracy (%)": None,
        "Precision (%)": None,
        "Recall (%)": None,
        "F1 (%)": 76.60,
        "PR-AUC": 0.814894
    },
    {
        "Experiment": "E3 — Adaptive Defense",
        "Dataset": "Independent Test",
        "Accuracy (%)": 99.9917,
        "Precision (%)": 94.74,
        "Recall (%)": 100.00,
        "F1 (%)": 97.30,
        "PR-AUC": None
    }
])

display_clean_results = (
    clean_results
    .where(pd.notna(clean_results), "Not reported")
    .astype(str)
)

st.dataframe(
    display_clean_results,
    width="stretch",
    hide_index=True
)

st.warning(
    """
    **IMPORTANT — FAIR COMPARISON LIMITATION**

    - **E1 — Baseline MLP:** Evaluated on the validation split.
    - **E2 — Adversarial MLP:** Evaluated on the validation split.
    - **E3 — Adaptive Defense:** Evaluated on the independent test split.

    E1/E2 and E3 results must not be interpreted as a direct
    same-dataset or same-split performance comparison.

    E1 and E2 provide a validation-based comparison of baseline
    and adversarial training. E3 reports the adaptive defense
    system's independent-test evaluation.

    E3 PR-AUC is marked **Not reported** because the available
    PR-AUC value belongs to standalone XGBoost and is not an
    independently measured end-to-end E3 PR-AUC.
    """
)

# ---------------------------------------------------------
# B. E1 vs E2 ADVERSARIAL ROBUSTNESS
# ---------------------------------------------------------

st.subheader("B. E1 vs E2 — Adversarial Attack Success Rate")

st.caption(
    "Validation results using each model's validation-F1-tuned "
    "threshold. Lower Attack Success Rate (ASR) indicates fewer "
    "successful evasions."
)

robustness_data = pd.DataFrame([
    ["FGSM", 0.01, "E1 Baseline MLP", 31.4815],
    ["FGSM", 0.01, "E2 Adversarial MLP", 27.7778],
    ["FGSM", 0.02, "E1 Baseline MLP", 44.4444],
    ["FGSM", 0.02, "E2 Adversarial MLP", 44.4444],
    ["FGSM", 0.05, "E1 Baseline MLP", 50.0000],
    ["FGSM", 0.05, "E2 Adversarial MLP", 50.0000],
    ["PGD", 0.01, "E1 Baseline MLP", 25.9259],
    ["PGD", 0.01, "E2 Adversarial MLP", 24.0741],
    ["PGD", 0.02, "E1 Baseline MLP", 44.4444],
    ["PGD", 0.02, "E2 Adversarial MLP", 38.8889],
    ["PGD", 0.05, "E1 Baseline MLP", 50.0000],
    ["PGD", 0.05, "E2 Adversarial MLP", 48.1481],
], columns=["Attack", "Epsilon", "Model", "ASR (%)"])

fig_robustness = px.bar(
    robustness_data,
    x="Epsilon",
    y="ASR (%)",
    color="Model",
    barmode="group",
    facet_col="Attack",
    title="Baseline vs Adversarial MLP — ASR"
)

fig_robustness.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis_title="Attack Success Rate (%)"
)

st.plotly_chart(fig_robustness, width="stretch")

st.dataframe(
    robustness_data,
    width="stretch",
    hide_index=True
)

st.caption(
    "Finding: adversarial training reduced ASR in some settings, "
    "but not consistently across all attack strengths."
)

# ---------------------------------------------------------
# C. FEATURE SQUEEZING ABLATION
# ---------------------------------------------------------

st.subheader("C. Feature Squeezing Ablation")

st.caption(
    "Baseline MLP validation experiment. Results use the "
    "validation-tuned threshold. ASR is calculated over 54 "
    "malicious validation records."
)

squeezing_data = pd.DataFrame([
    ["Clean", 0.00, "No squeezing", 0.00],
    ["Clean", 0.00, "1 decimal", 20.37],
    ["Clean", 0.00, "2 decimals", 0.00],

    ["FGSM", 0.01, "No squeezing", 24.07],
    ["FGSM", 0.01, "1 decimal", 31.48],
    ["FGSM", 0.01, "2 decimals", 24.07],

    ["PGD", 0.01, "No squeezing", 25.93],
    ["PGD", 0.01, "1 decimal", 35.19],
    ["PGD", 0.01, "2 decimals", 24.07],

    ["FGSM", 0.02, "No squeezing", 44.44],
    ["FGSM", 0.02, "1 decimal", 46.30],
    ["FGSM", 0.02, "2 decimals", 44.44],

    ["PGD", 0.02, "No squeezing", 44.44],
    ["PGD", 0.02, "1 decimal", 48.15],
    ["PGD", 0.02, "2 decimals", 44.44],

    ["FGSM", 0.05, "No squeezing", 50.00],
    ["FGSM", 0.05, "1 decimal", 50.00],
    ["FGSM", 0.05, "2 decimals", 50.00],

    ["PGD", 0.05, "No squeezing", 50.00],
    ["PGD", 0.05, "1 decimal", 50.00],
    ["PGD", 0.05, "2 decimals", 50.00],
], columns=["Attack", "Epsilon", "Condition", "ASR (%)"])

fig_squeezing = px.bar(
    squeezing_data[squeezing_data["Attack"] != "Clean"],
    x="Epsilon",
    y="ASR (%)",
    color="Condition",
    barmode="group",
    facet_col="Attack",
    title="Feature Squeezing — Adversarial ASR"
)

fig_squeezing.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig_squeezing, width="stretch")

st.dataframe(
    squeezing_data,
    width="stretch",
    hide_index=True
)

st.info(
    "Finding: 2-decimal squeezing did not consistently reduce "
    "adversarial evasion. It is evaluated as a component of the "
    "Deep Path, not claimed as a standalone reliable defense."
)

# ---------------------------------------------------------
# D. E3 INDEPENDENT TEST RESULTS
# ---------------------------------------------------------

st.subheader("D. E3 — Independent Test Evaluation")

e3_col1, e3_col2, e3_col3, e3_col4 = st.columns(4)

e3_col1.metric("Test Accuracy", "99.9917%")
e3_col2.metric("Test Precision", "94.7368%")
e3_col3.metric("Test Recall", "100%")
e3_col4.metric("Test F1", "97.2973%")

e3_confusion = pd.DataFrame([
    ["True Negative", 36058],
    ["False Positive", 3],
    ["False Negative", 0],
    ["True Positive", 54]
], columns=["Confusion Matrix Component", "Records"])

st.dataframe(
    e3_confusion,
    width="stretch",
    hide_index=True
)

st.caption(
    "Independent test: 36,115 records, including 36,061 benign "
    "records and 54 malicious records. No test-set training or "
    "threshold tuning was performed in the evaluation script."
)

# ---------------------------------------------------------
# E. E3 ADAPTIVE ROUTING
# ---------------------------------------------------------

st.subheader("E. E3 — Adaptive Fast / Deep Routing")

routing_data = pd.DataFrame([
    ["Fast Path", 36055],
    ["Deep Path", 60]
], columns=["Path", "Records"])

route_col1, route_col2 = st.columns(2)

route_col1.metric("Fast Path", "36,055", "99.8339%")
route_col2.metric("Deep Path", "60", "0.1661%")

fig_routing = px.pie(
    routing_data,
    names="Path",
    values="Records",
    hole=0.55,
    title="Independent Test Routing Distribution"
)

fig_routing.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig_routing, width="stretch")

st.caption(
    "All 54 malicious test records were routed to Deep Path. "
    "Six benign records were also routed to Deep Path."
)

# ---------------------------------------------------------
# F. E3 CONSTRAINED ADVERSARIAL ROUTING
# ---------------------------------------------------------

st.subheader("F. E3 — Constrained Adversarial Routing")

st.caption(
    "Validation-only tabular feature-space simulation. "
    "These are not packet-level attack results."
)

e3_attack_data = pd.DataFrame([
    ["FGSM", 0.01, 62.9630],
    ["FGSM", 0.02, 61.1111],
    ["FGSM", 0.05, 53.7037],
    ["PGD", 0.01, 57.4074],
    ["PGD", 0.02, 53.7037],
    ["PGD", 0.05, 51.8519],
], columns=["Attack", "Epsilon", "ASR (%)"])

fig_e3_attack = px.bar(
    e3_attack_data,
    x="Epsilon",
    y="ASR (%)",
    color="Attack",
    barmode="group",
    title="E3 Constrained Adversarial Routing — ASR"
)

fig_e3_attack.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig_e3_attack, width="stretch")

st.dataframe(
    e3_attack_data,
    width="stretch",
    hide_index=True
)

st.warning(
    "The constrained simulation enforces binary, categorical and "
    "integer feature domains and an epsilon budget. Relational "
    "packet constraints are not implemented."
)

# ---------------------------------------------------------
st.warning(
    """
    **ROUTING THRESHOLD DIFFERENCE**

    - **Adversarial simulation threshold:** 0.0001
    - **Dashboard live routing threshold:** 0.01

    The adversarial routing results above were generated using
    threshold 0.0001. The live dashboard uses 0.01.

    Therefore, these adversarial simulation results should not
    be interpreted as measurements of live dashboard routing
    performance at threshold 0.01.
    """
)
# G. E3 LATENCY AND EFFICIENCY
# ---------------------------------------------------------

st.subheader("G. E3 — Latency and Efficiency")

lat_col1, lat_col2, lat_col3 = st.columns(3)

lat_col1.metric(
    "Pipeline Mean Batch Latency",
    "58.78 ms"
)

lat_col2.metric(
    "Latency (ms/record)",
    "0.001627"
)

lat_col3.metric(
    "Throughput",
    "614,473 records/sec"
)

st.caption(
    "CPU validation-batch benchmark with 30 measured runs. "
    "These are batch-amortized measurements, not live packet "
    "latency or single-record request timings."
)

st.caption(
    "Memory usage after benchmark: approximately 391.77 MB."
)

st.subheader("E2 vs E3 — Efficiency Reporting")

efficiency_comparison = pd.DataFrame([
    {
        "Metric": "Mean Batch Latency",
        "E2 — Adversarial MLP": "Not reported",
        "E3 — Adaptive Defense": "58.78 ms"
    },
    {
        "Metric": "Amortized Latency",
        "E2 — Adversarial MLP": "Not reported",
        "E3 — Adaptive Defense": "0.001627 ms / record"
    },
    {
        "Metric": "Throughput",
        "E2 — Adversarial MLP": "Not reported",
        "E3 — Adaptive Defense": "614,473 records/sec"
    },
    {
        "Metric": "Memory Usage",
        "E2 — Adversarial MLP": "Not reported",
        "E3 — Adaptive Defense": "391.77 MB"
    },
    {
        "Metric": "CPU Utilization",
        "E2 — Adversarial MLP": "Not reported",
        "E3 — Adaptive Defense": "Not separately reported"
    }
])

st.dataframe(
    efficiency_comparison,
    width="stretch",
    hide_index=True
)

st.caption(
    "E2 efficiency measurements were not available in the "
    "reported experiments. Missing values are explicitly shown "
    "as Not reported; no estimates have been introduced."
)
st.divider()
st.caption(
    "E1/E2: grouped validation experiments | "
    "E3: independent test and separate validation-only "
    "adversarial/efficiency experiments."
)

# =========================================================
# 17. GLOBAL FEATURE IMPORTANCE
# =========================================================

st.divider()
st.header("07 | Risk Engine Feature Importance")

st.caption(
    "Global XGBoost feature importance. This explains the model "
    "overall, not the individual record selected above."
)

try:

    importance = risk_model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importance
    }).sort_values(
        "Importance",
        ascending=False
    ).head(15)

    fig = px.bar(
        importance_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 15 Global Features",
        color="Importance",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="",
        xaxis_title="Importance"
    )

    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        importance_df,
        width="stretch",
        hide_index=True
    )

except Exception as error:

    st.warning(
        f"Feature importance unavailable: {error}"
    )


# =========================================================
# 18. EXPORT RESULTS
# =========================================================

st.divider()
st.header("08 | Export Evaluation Results")

csv_data = results.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇ Download Prediction Results CSV",
    data=csv_data,
    file_name="adaptive_nids_predictions.csv",
    mime="text/csv",
    width="stretch"
)

st.success(
    "Evaluation completed for the uploaded offline dataset."
)

st.caption(
    "Adaptive Risk-Aware NIDS | FYP Experimental Prototype"
)