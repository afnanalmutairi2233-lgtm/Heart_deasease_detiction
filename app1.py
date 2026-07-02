import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Heart Deasease Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# COLOR PALETTE (matches the reference dashboard)
# ============================================================
BG        = "#0e1017"
CARD_BG   = "#1a1d2b"
CARD_BG2  = "#181b28"
BORDER    = "#2a2e42"
BLUE      = "#4a9fd8"
ORANGE    = "#f2a541"
TEAL      = "#2dd4bf"
RED       = "#e2725b"
PURPLE    = "#7c6ce0"
TEXT      = "#f5f6fa"
MUTED     = "#9399ad"
GREEN     = "#3ecf8e"

PALETTE = [BLUE, ORANGE, TEAL, RED, PURPLE, GREEN]

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT, family="Trebuchet MS, sans-serif", size=13),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=MUTED),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=MUTED),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=50, b=40),
        colorway=PALETTE,
    )
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(f"""
<style>
    .stApp {{
        background-color: {BG};
        color: {TEXT};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {CARD_BG2};
        border-right: 1px solid {BORDER};
    }}
    h1, h2, h3, h4 {{
        color: {TEXT} !important;
        font-weight: 800 !important;
    }}
    p, li, span, label, .stMarkdown {{
        color: {TEXT};
    }}
    .subtle {{ color: {MUTED}; font-size: 0.92rem; }}

    .kpi-card {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 10px;
    }}
    .kpi-value {{
        font-size: 2rem;
        font-weight: 800;
        color: {TEXT};
        margin: 0;
    }}
    .kpi-label {{
        color: {MUTED};
        font-size: 0.85rem;
        margin: 0;
    }}
    .section-card {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 16px;
    }}
    .pill {{
        display:inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right:6px;
    }}
    .pill-blue   {{ background: rgba(74,159,216,0.18); color:{BLUE}; }}
    .pill-orange {{ background: rgba(242,165,65,0.18); color:{ORANGE}; }}
    .pill-teal   {{ background: rgba(45,212,191,0.18); color:{TEAL}; }}
    .pill-red    {{ background: rgba(226,114,91,0.18); color:{RED}; }}
    .pill-purple {{ background: rgba(124,108,224,0.18); color:{PURPLE}; }}
    .pill-green  {{ background: rgba(62,207,142,0.18); color:{GREEN}; }}

    .result-box {{
        border-radius: 18px;
        padding: 28px;
        text-align:center;
        border: 1px solid {BORDER};
    }}
    div[data-testid="stMetricValue"] {{ color: {TEXT}; }}
    div[data-testid="stMetricLabel"] {{ color: {MUTED}; }}

    .stButton>button {{
        background: linear-gradient(135deg, {BLUE}, {PURPLE});
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.4rem;
        font-weight: 700;
    }}
    .stDownloadButton>button {{
        background-color: {CARD_BG};
        color: {TEAL};
        border: 1px solid {TEAL};
        border-radius: 10px;
        font-weight: 700;
    }}
    hr {{ border-color: {BORDER}; }}

    thead tr th {{ background-color: {CARD_BG2} !important; color: {TEXT} !important; }}
    tbody tr td {{ color: {TEXT} !important; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# FEATURE DEFINITIONS
# ============================================================
BINARY_FEATURES = [
    "Chest_Pain", "Shortness_of_Breath", "Fatigue", "Palpitations", "Dizziness",
    "Swelling", "Pain_Arms_Jaw_Back", "Cold_Sweats_Nausea", "High_BP",
    "High_Cholesterol", "Diabetes", "Smoking", "Obesity", "Sedentary_Lifestyle",
    "Family_History", "Chronic_Stress",
]
ALL_FEATURES = BINARY_FEATURES + ["Gender", "Age"]
TARGET = "Heart_Risk"
ALL_COLUMNS = ALL_FEATURES + [TARGET]

COLUMN_DESCRIPTIONS = {
    "Chest_Pain": "Patient reports chest pain (1 = yes, 0 = no)",
    "Shortness_of_Breath": "Patient reports shortness of breath (1 = yes, 0 = no)",
    "Fatigue": "Patient reports unusual fatigue (1 = yes, 0 = no)",
    "Palpitations": "Patient reports heart palpitations (1 = yes, 0 = no)",
    "Dizziness": "Patient reports dizziness (1 = yes, 0 = no)",
    "Swelling": "Patient reports swelling, e.g. legs/ankles (1 = yes, 0 = no)",
    "Pain_Arms_Jaw_Back": "Pain radiating to arms, jaw or back (1 = yes, 0 = no)",
    "Cold_Sweats_Nausea": "Cold sweats and/or nausea (1 = yes, 0 = no)",
    "High_BP": "Diagnosed with high blood pressure (1 = yes, 0 = no)",
    "High_Cholesterol": "Diagnosed with high cholesterol (1 = yes, 0 = no)",
    "Diabetes": "Diagnosed with diabetes (1 = yes, 0 = no)",
    "Smoking": "Current or past smoker (1 = yes, 0 = no)",
    "Obesity": "Diagnosed with obesity (1 = yes, 0 = no)",
    "Sedentary_Lifestyle": "Mostly inactive lifestyle (1 = yes, 0 = no)",
    "Family_History": "Family history of heart disease (1 = yes, 0 = no)",
    "Chronic_Stress": "Reports chronic stress (1 = yes, 0 = no)",
    "Gender": "Patient gender (0 = Female, 1 = Male)",
    "Age": "Patient age in years",
    "Heart_Risk": "Target: heart disease risk (0 = Low Risk, 1 = High Risk)",
}

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(show_spinner=False)
def load_default_or_none():
    candidates = ["project_dataset_heart_diesese.csv", "heart_dataset.csv"]
    for c in candidates:
        if os.path.exists(c):
            for sep in [";", ","]:
                try:
                    df_try = pd.read_csv(c, sep=sep)
                    if set(ALL_COLUMNS).issubset(df_try.columns):
                        return df_try[ALL_COLUMNS].copy()
                except Exception:
                    pass
    return None

@st.cache_data(show_spinner=False)
def generate_synthetic(n=4000, seed=42):
    rng = np.random.default_rng(seed)
    data = {f: rng.integers(0, 2, n) for f in BINARY_FEATURES}
    data["Gender"] = rng.integers(0, 2, n)
    data["Age"] = rng.integers(18, 91, n)
    df_s = pd.DataFrame(data)
    risk_score = (
        df_s[BINARY_FEATURES].sum(axis=1) * 1.0
        + (df_s["Age"] - 18) / 72 * 4
        + rng.normal(0, 1.4, n)
    )
    threshold = np.percentile(risk_score, 55)
    df_s[TARGET] = (risk_score > threshold).astype(int)
    return df_s[ALL_COLUMNS]

# ============================================================
# CLEANING + PREPROCESSING (mirrors the notebook)
# ============================================================
@st.cache_data(show_spinner=False)
def clean_data(df_raw):
    stats = {}
    stats["n_missing"] = int(df_raw.isnull().sum().sum())
    stats["n_before"] = len(df_raw)
    stats["n_duplicates"] = int(df_raw.duplicated().sum())
    df_c = df_raw.drop_duplicates().reset_index(drop=True)
    stats["n_after"] = len(df_c)

    Q1, Q3 = df_c["Age"].quantile(0.25), df_c["Age"].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    stats["n_outliers"] = int(((df_c["Age"] < lower) | (df_c["Age"] > upper)).sum())
    return df_c, stats

# ============================================================
# MODEL TRAINING (cached resource)
# ============================================================
@st.cache_resource(show_spinner=False)
def train_models(df_clean):
    X = df_clean[ALL_FEATURES]
    y = df_clean[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)

    svm = SVC(random_state=42, probability=True)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)

    def metrics(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-score": f1_score(y_true, y_pred),
        }

    results = {
        "Random Forest": {
            "model": rf, "y_pred": y_pred_rf, "metrics": metrics(y_test, y_pred_rf),
            "cm": confusion_matrix(y_test, y_pred_rf),
        },
        "SVM": {
            "model": svm, "y_pred": y_pred_svm, "metrics": metrics(y_test, y_pred_svm),
            "cm": confusion_matrix(y_test, y_pred_svm),
        },
    }
    return {
        "scaler": scaler, "results": results,
        "X_train": X_train, "X_train_scaled": X_train_scaled,
        "X_test": X_test, "y_test": y_test,
        "feature_names": ALL_FEATURES,
    }

# ============================================================
# SIDEBAR - DATA SOURCE + NAV + FILTERS
# ============================================================
st.sidebar.markdown(f"<h2 style='color:{TEXT};margin-bottom:0;'> Heart Deasease in AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown(f"<p class='subtle'>Analytics and Prediction </p>", unsafe_allow_html=True)
st.sidebar.image("heart_ai.png", width=250)
#st.sidebar.markdown("---")

data_source_note = ""
raw_df = load_default_or_none()

if raw_df is None:
    raw_df = generate_synthetic()
    data_source_note = ("No dataset file found — showing DEMO data with the same structure. "
                         "Upload `project_dataset_heart_diesese.csv` (or use the uploader) to see your real results.")
elif not data_source_note:
    data_source_note = "Using bundled project dataset."

df_clean, clean_stats = clean_data(raw_df)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    [" Data Overview", " EDA", " Cleaning & Preprocessing", " Model Results", " Interactive Prediction"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"<p class='subtle'><b>Filters</b> (apply to Overview & EDA)</p>", unsafe_allow_html=True)

gender_filter = st.sidebar.multiselect(
    "Gender", options=[0, 1], default=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male",
)
age_min, age_max = int(df_clean["Age"].min()), int(df_clean["Age"].max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
risk_filter = st.sidebar.multiselect(
    "Heart Risk", options=[0, 1], default=[0, 1],
    format_func=lambda x: "Low Risk" if x == 0 else "High Risk",
)
if st.sidebar.button("↺ Reset filters"):
    st.rerun()

mask = (
    df_clean["Gender"].isin(gender_filter if gender_filter else [0, 1])
    & df_clean["Age"].between(age_range[0], age_range[1])
    & df_clean["Heart_Risk"].isin(risk_filter if risk_filter else [0, 1])
)
fdf = df_clean[mask].copy()

st.sidebar.markdown("---")
st.sidebar.caption(data_source_note)

# ============================================================
# HEADER
# ============================================================
st.markdown(f"<h1 style='margin-bottom:0;'> Predicting heart disease</h1>", unsafe_allow_html=True)
#st.markdown(
   # f"<p class='subtle'>An end-to-end walkthrough: data → EDA → cleaning → modeling → live prediction</p>",
   # unsafe_allow_html=True,
#)
st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# PAGE 1 — DATA OVERVIEW
# ============================================================
if page == " Data Overview":
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("Patients (filtered)", f"{len(fdf):,}", "pill-blue"),
        ("Features", f"{len(ALL_FEATURES)}", "pill-teal"),
        ("High Risk %", f"{(fdf[TARGET].mean()*100 if len(fdf) else 0):.1f}%", "pill-red"),
        ("Average Age", f"{(fdf['Age'].mean() if len(fdf) else 0):.1f}", "pill-orange"),
    ]
    for col, (label, value, pill) in zip([c1, c2, c3, c4], kpis):
        col.markdown(f"""
        <div class="kpi-card">
            <span class="pill {pill}">●</span>
            <p class="kpi-value">{value}</p>
            <p class="kpi-label">{label}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
    <h4> Project Description</h4>
    <p class="subtle">
    This project describes the prediction of heart disease risk, which can lead to sudden cardiac arrest. 
    A cardiac arrest is a medical emergency caused by a severe electrical disturbance in the heart that prevents it from pumping blood effectively.
    The prediction is based on several factors, including family history, age, unhealthy habits such as smoking, cholesterol levels, and respiratory rate. 
    Predicting heart disease risk can allow for further testing of high-risk patients, review of their medications, and more precise monitoring of their condition.
    It can also help identify individuals who might benefit from preventative measures or pacemakers/cardiac pacemakers in the future. Two models were trained,
    and the results were analyzed to achieve a reasonably accurate predictive score.
    </p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([2, 1])
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("####  Dataset Sample (filtered)")
        st.dataframe(fdf.head(15), use_container_width=True, height=380)
        csv_bytes = fdf.to_csv(index=False).encode("utf-8")
        st.download_button(" Download filtered data (CSV)", csv_bytes, "filtered_heart_data.csv", "text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("####  Column Guide")
        for col in ALL_COLUMNS:
            st.markdown(f"<p style='margin-bottom:4px;'><b style='color:{TEAL};'>{col}</b> "
                         f"<span class='subtle'>— {COLUMN_DESCRIPTIONS.get(col,'')}</span></p>",
                         unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 2 — EDA
# ============================================================
elif page == " EDA":
    if len(fdf) == 0:
        st.warning("No rows match the current filters — adjust the sidebar filters.")
    else:
        r1c1, r1c2 = st.columns(2)

        with r1c1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### Age Distribution")
            fig = px.histogram(fdf, x="Age", nbins=25, color_discrete_sequence=[BLUE])
            fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(), bargap=0.05, height=320)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"<p class='subtle'>Distribution of patient ages in the dataset.</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r1c2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### Family History vs. Heart Risk")
            grp = fdf.groupby(["Family_History", TARGET]).size().reset_index(name="Count")
            grp["Family_History"] = grp["Family_History"].map({0: "No history", 1: "Has history"})
            grp[TARGET] = grp[TARGET].map({0: "Low Risk", 1: "High Risk"})
            fig = px.bar(grp, x="Family_History", y="Count", color=TARGET, barmode="group",
                         color_discrete_map={"Low Risk": BLUE, "High Risk": ORANGE})
            fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(), height=320)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"<p class='subtle'>Relationship between family history and the likelihood of developing heart disease.</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        r2c1, r2c2 = st.columns(2)

        with r2c1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### Chest Pain & Shortness of Breath → Risk")
            symptoms = fdf[(fdf["Chest_Pain"] == 1) & (fdf["Shortness_of_Breath"] == 1)]
            if len(symptoms) > 0:
                risk_counts = symptoms[TARGET].value_counts().reindex([0, 1], fill_value=0)
                fig = go.Figure(go.Pie(
                    labels=["Low Risk", "High Risk"], values=risk_counts.values,
                    hole=0.62, marker=dict(colors=[BLUE, ORANGE]),
                    textinfo="percent", textfont=dict(color=TEXT, size=14),
                ))
                fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(), height=320,
                                   showlegend=True,
                                   annotations=[dict(text=f"{len(symptoms)}", x=0.5, y=0.5,
                                                      font_size=26, showarrow=False, font_color=TEXT)])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No patients with both symptoms in current filter.")
            st.markdown(f"<p class='subtle'>Among patients reporting both chest pain and shortness of breath, "
                         f"this shows the split between low and high risk outcomes.</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r2c2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### Correlation Heatmap")
            corr = fdf[ALL_FEATURES + [TARGET]].corr()
            custom_scale = [[0, RED], [0.5, CARD_BG], [1, TEAL]]
            fig = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.columns,
                colorscale=custom_scale, zmid=0,
                colorbar=dict(tickfont=dict(color=MUTED)),
            ))
            base_layout = PLOTLY_TEMPLATE["layout"].to_plotly_json()
            base_layout["xaxis"] = {**base_layout.get("xaxis", {}), "tickfont": dict(size=8)}
            base_layout["yaxis"] = {**base_layout.get("yaxis", {}), "tickfont": dict(size=8)}
            fig.update_layout(**base_layout, height=320)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"<p class='subtle'>Correlation between all features and Heart_Risk. "
                         f"Teal = positive correlation, red = negative.</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Risk Factor Prevalence")
        prevalence = fdf[BINARY_FEATURES].mean().sort_values(ascending=False) * 100
        fig = px.bar(x=prevalence.values, y=prevalence.index, orientation="h",
                     color=prevalence.values, color_continuous_scale=[BLUE, TEAL, ORANGE])
        fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(), height=420,
                           coloraxis_showscale=False, xaxis_title="% of patients", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<p class='subtle'>Share of patients (in the filtered data) reporting each symptom or risk factor.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 3 — CLEANING & PREPROCESSING
# ============================================================
elif page == " Cleaning & Preprocessing":
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("Missing values", clean_stats["n_missing"], "pill-teal"),
        ("Duplicates removed", clean_stats["n_duplicates"], "pill-red"),
        ("Rows before → after", f"{clean_stats['n_before']:,} → {clean_stats['n_after']:,}", "pill-blue"),
        ("Age outliers (IQR)", clean_stats["n_outliers"], "pill-orange"),
    ]
    for col, (label, value, pill) in zip([c1, c2, c3, c4], kpis):
        col.markdown(f"""
        <div class="kpi-card">
            <span class="pill {pill}">●</span>
            <p class="kpi-value">{value}</p>
            <p class="kpi-label">{label}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="section-card">
    <h4>What was done: </h4>
    <ul class="subtle">
      <li><b>Missing values:</b> checked with <code>isnull().sum()</code> — {clean_stats['n_missing']} missing values found, no imputation required.</li>
      <li><b>Duplicates:</b> {clean_stats['n_duplicates']:,} duplicate rows found and removed with <code>drop_duplicates()</code>.</li>
      <li><b>Outliers:</b> IQR method applied to <code>Age</code> — {clean_stats['n_outliers']} outliers detected (none removed, values were within a reasonable clinical range).</li>
      <li><b>Encoding:</b> all symptom/risk-factor columns and Gender are already numerically encoded (0/1), so no additional categorical encoding was needed.</li>
      <li><b>Scaling:</b> all features were standardized with <code>StandardScaler</code> (mean 0, std 1) before modeling, fit on the training set only to avoid data leakage.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Before vs. After Scaling")
    bcol, acol = st.columns(2)
    sample_cols = ["Age", "Chest_Pain", "High_Cholesterol"]
    X = df_clean[ALL_FEATURES]
    scaler_preview = StandardScaler().fit(X)
    scaled_preview = pd.DataFrame(scaler_preview.transform(X), columns=ALL_FEATURES)
    with bcol:
        st.markdown(f"<p class='subtle'><b style='color:{ORANGE};'>Before (raw)</b></p>", unsafe_allow_html=True)
        st.dataframe(X[sample_cols].head(6), use_container_width=True)
    with acol:
        st.markdown(f"<p class='subtle'><b style='color:{TEAL};'>After (StandardScaler)</b></p>", unsafe_allow_html=True)
        st.dataframe(scaled_preview[sample_cols].head(6).round(3), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 4 — MODEL RESULTS
# ============================================================
elif page == " Model Results":
    with st.spinner("Training models..."):
        bundle = train_models(df_clean)
    results = bundle["results"]

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("####  Model Comparison")
    metric_rows = []
    for name, r in results.items():
        row = {"Model": name}
        row.update({k: round(v, 4) for k, v in r["metrics"].items()})
        metric_rows.append(row)
    metrics_df = pd.DataFrame(metric_rows)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    long_df = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(long_df, x="Metric", y="Score", color="Model", barmode="group",
                 color_discrete_map={"Random Forest": TEAL, "SVM": ORANGE})
    fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(), height=380, yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    best_model = metrics_df.loc[metrics_df["F1-score"].idxmax(), "Model"]
    st.markdown(f"<p class='subtle'> <b style='color:{GREEN};'>{best_model}</b> achieved the best overall F1-score on the held-out test set.</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    model_choice = c1.selectbox("Select model for evaluation plot", list(results.keys()))

    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        cm = results[model_choice]["cm"]
        colA, colB = st.columns(2)
        with colA:
            st.markdown(f"#### Confusion Matrix — {model_choice}")
            fig = go.Figure(go.Heatmap(
                z=cm, x=["Pred: Low", "Pred: High"], y=["True: Low", "True: High"],
                colorscale=[[0, CARD_BG], [1, BLUE]],
                text=cm, texttemplate="%{text}", textfont=dict(size=18, color=TEXT),
                showscale=False,
            ))
            fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(), height=340)
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            if model_choice == "Random Forest":
                st.markdown("#### Feature Importance")
                importances = pd.Series(
                    results[model_choice]["model"].feature_importances_, index=bundle["feature_names"]
                ).sort_values(ascending=True)
                fig = px.bar(x=importances.values, y=importances.index, orientation="h",
                             color=importances.values, color_continuous_scale=[BLUE, TEAL, ORANGE])
                fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(), height=340,
                                   coloraxis_showscale=False, xaxis_title="Importance", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("#### ROC Curve")
                y_score = results[model_choice]["model"].predict_proba(
                    bundle["scaler"].transform(bundle["X_test"])
                )[:, 1]
                fpr, tpr, _ = roc_curve(bundle["y_test"], y_score)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=ORANGE, width=3), name="ROC"))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dash"), name="Random"))
                fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(), height=340,
                                   xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 5 — INTERACTIVE PREDICTION
# ============================================================
elif page == " Interactive Prediction":
    with st.spinner("Preparing model..."):
        bundle = train_models(df_clean)
    results = bundle["results"]

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Enter patient information: ")

    top1, top2 = st.columns(2)
    model_name = top1.selectbox("Model to use for prediction", list(results.keys()), index=0)
    age = top2.slider("Age", 18, 90, 45)
    gender_label = top1.selectbox("Gender", ["Female", "Male"])
    gender = 0 if gender_label == "Female" else 1

    st.markdown(f"<p class='subtle' style='margin-top:10px;'><b>Symptoms & Risk Factors</b></p>", unsafe_allow_html=True)
    input_vals = {}
    cols = st.columns(4)
    for i, feat in enumerate(BINARY_FEATURES):
        with cols[i % 4]:
            label = feat.replace("_", " ")
            input_vals[feat] = 1 if st.checkbox(label, value=False, key=f"chk_{feat}") else 0

    predict_clicked = st.button(" Predict Heart Risk")
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        row = {**input_vals, "Gender": gender, "Age": age}
        X_input = pd.DataFrame([row])[ALL_FEATURES]
        X_scaled = bundle["scaler"].transform(X_input)
        model = results[model_name]["model"]
        pred = model.predict(X_scaled)[0]
        try:
            proba = model.predict_proba(X_scaled)[0][1]
        except Exception:
            proba = float(pred)

        is_high = pred == 1
        color = RED if is_high else GREEN
        label = "HIGH RISK" if is_high else "LOW RISK"
        emoji = "⚠️" if is_high else "✅"

        r1, r2 = st.columns([1, 1])
        with r1:
            st.markdown(f"""
            <div class="result-box" style="background: rgba({'226,114,91' if is_high else '62,207,142'},0.12); border-color:{color};">
                <h2 style="color:{color}; margin-bottom:4px;">{emoji} {label}</h2>
                <p class="subtle">Predicted by {model_name}</p>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"color": TEXT}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": MUTED},
                    "bar": {"color": color},
                    "bgcolor": CARD_BG,
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50], "color": "rgba(62,207,142,0.15)"},
                        {"range": [50, 100], "color": "rgba(226,114,91,0.15)"},
                    ],
                },
                title={"text": "High-risk probability", "font": {"color": MUTED, "size": 14}},
            ))
            fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(), height=260)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Set the values above and click **Predict Heart Risk** to get a live prediction.")

st.markdown("<br><hr><p class='subtle' style='text-align:center;'>Heart Risk Analytics App — built with Streamlit</p>", unsafe_allow_html=True)
