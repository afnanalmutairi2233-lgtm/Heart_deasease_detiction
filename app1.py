"""Streamlit dashboard for descriptive analysis of the bundled heart-label data."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from heart_analysis import BINARY_FEATURES, FEATURES, TARGET, data_quality_report, load_dataset, train_models


st.set_page_config(page_title="Heart Dataset Analysis", page_icon="📊", layout="wide")
st.title("Heart Dataset: Descriptive Analysis")
st.caption("Exploratory analysis and an illustrative classifier evaluation. This is not a medical or diagnostic tool.")
st.warning(
    "The dataset does not document its source or how the Heart_Risk label was created. "
    "Model scores measure agreement with these supplied labels only; they do not establish clinical accuracy."
)


@st.cache_data(show_spinner=False)
def get_data():
    return load_dataset()


try:
    df = get_data()
    quality = data_quality_report(df)
except (OSError, ValueError) as exc:
    st.error(f"Unable to load or validate the bundled dataset: {exc}")
    st.stop()

st.sidebar.header("Filters")
gender_options = sorted(df["Gender"].unique().tolist())
gender_filter = st.sidebar.multiselect(
    "Gender code", gender_options, default=gender_options,
    help="0 and 1 are codes in the supplied file; code meanings are not documented here.",
)
age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("Age", age_min, age_max, (age_min, age_max))
risk_options = sorted(df[TARGET].unique().tolist())
risk_filter = st.sidebar.multiselect("Heart_Risk label", risk_options, default=risk_options)
filtered = df[
    df["Gender"].isin(gender_filter)
    & df["Age"].between(age_range[0], age_range[1])
    & df[TARGET].isin(risk_filter)
].copy()

tab_overview, tab_eda, tab_quality, tab_models = st.tabs(
    ["Overview", "Exploratory Analysis", "Data Quality", "Model Evaluation"]
)

with tab_overview:
    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filtered records", f"{len(filtered):,}")
    c2.metric("Input features", len(FEATURES))
    c3.metric("Exact duplicate rows", f"{quality['duplicate_rows']:,}")
    c4.metric("Missing cells", f"{quality['missing_cells']:,}")
    st.markdown(
        "The dashboard describes the recorded fields and the supplied binary `Heart_Risk` label. "
        "The target's definition and dataset provenance are not documented in the project files."
    )
    if filtered.empty:
        st.info("No records match the selected filters. Change one or more filters to view data.")
    else:
        st.dataframe(filtered.head(20), width="stretch", hide_index=True)
        st.download_button(
            "Download filtered records", filtered.to_csv(index=False).encode("utf-8"),
            "filtered_heart_records.csv", "text/csv",
        )

with tab_eda:
    st.subheader("Label and feature distributions")
    if filtered.empty:
        st.info("No records match the selected filters. Change one or more filters to view charts.")
    else:
        counts = filtered[TARGET].value_counts().sort_index().rename_axis("Heart_Risk label").reset_index(name="Records")
        st.plotly_chart(px.bar(counts, x="Heart_Risk label", y="Records", title="Supplied label counts"), width="stretch")
        rate = filtered.groupby("Age", as_index=False)[TARGET].mean()
        st.plotly_chart(px.line(rate, x="Age", y=TARGET, title="Mean supplied label by age"), width="stretch")
        prevalence = filtered[BINARY_FEATURES].mean().sort_values().rename("Share with value 1").reset_index()
        prevalence.columns = ["Recorded field", "Share with value 1"]
        st.plotly_chart(px.bar(prevalence, x="Share with value 1", y="Recorded field", orientation="h",
                               title="Recorded binary-field prevalence"), width="stretch")
        st.caption("These are descriptive associations in this file. They do not establish causation or medical meaning.")

with tab_quality:
    st.subheader("Data quality report")
    st.json(quality)
    st.write("Rows with missing required values or invalid field codes cause validation to fail; they are not silently dropped.")
    st.write(
        "Exact duplicate rows remain in the source data and descriptive views. "
        "Model evaluation removes exact copies in memory before splitting to reduce train/test duplication."
    )

with tab_models:
    st.subheader("Illustrative label-classification results")
    st.caption(
        "A stratified random holdout is used. Preprocessing is fitted on training data only. "
        "The supplied target's provenance is unknown, so these metrics are not evidence of medical prediction."
    )
    if st.button("Evaluate models", type="primary"):
        with st.spinner("Evaluating models on a stratified holdout…"):
            try:
                bundle = train_models(df)
            except ValueError as exc:
                st.error(str(exc))
            else:
                metrics = pd.DataFrame(
                    [{
                        "Model": name,
                        "Accuracy": result["metrics"]["Accuracy"],
                        "Precision (label 1)": result["metrics"]["Precision"],
                        "Recall (label 1)": result["metrics"]["Recall"],
                        "F1-score (label 1)": result["metrics"]["F1-score"],
                    } for name, result in bundle["results"].items()]
                )
                st.caption(
                    f"Unique rows available: {bundle['model_rows']:,}; "
                    f"stratified evaluation sample: {bundle['sample_rows']:,}; "
                    f"exact duplicates excluded from evaluation: {bundle['rows_removed_as_exact_duplicates_for_evaluation']:,}."
                )
                st.dataframe(metrics.round(4), width="stretch", hide_index=True)
                selected = st.selectbox("Confusion matrix model", list(bundle["results"]))
                selected_result = bundle["results"][selected]
                st.markdown(f"**Per-class metrics — {selected}**")
                class_metrics = pd.DataFrame.from_dict(selected_result["class_metrics"], orient="index")
                class_metrics.index.name = "Heart_Risk label"
                st.dataframe(class_metrics.round(4), width="stretch")
                cm = selected_result["confusion_matrix"]
                fig = go.Figure(go.Heatmap(z=cm, x=["Predicted 0", "Predicted 1"],
                                           y=["Actual 0", "Actual 1"], text=cm,
                                           texttemplate="%{text}", colorscale="Blues"))
                fig.update_layout(title=f"{selected}: held-out label confusion matrix")
                st.plotly_chart(fig, width="stretch")
                importances = bundle["results"]["Random Forest"]["feature_importances"]
                importance_df = importances.rename("Importance").rename_axis("Recorded feature").reset_index()
                st.plotly_chart(px.bar(importance_df.head(10), x="Importance", y="Recorded feature",
                                       orientation="h", title="Random Forest feature importance (top 10)"),
                                width="stretch")
    else:
        st.info("Select the button to run the evaluation. The reported scores will describe agreement with the supplied labels only.")
