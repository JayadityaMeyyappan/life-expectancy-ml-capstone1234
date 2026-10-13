from __future__ import annotations

from pathlib import Path
import re
import io

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from preprocessing_utils import IQRClipper  # importable class used by the saved model

ROOT = Path(__file__).resolve().parent
DATA_PATHS = [
    ROOT / "data" / "Life Expectancy Data.csv",
    ROOT / "data" / "Life expectancy.csv",
    ROOT / "Life Expectancy Data.csv",
    ROOT / "Life expectancy.csv",
]
RESULTS = ROOT / "results"
MODEL_PATH = ROOT / "models" / "life_expectancy_model.joblib"

st.set_page_config(page_title="LifeLens | Life Expectancy ML", page_icon="🌍",
                   layout="wide", initial_sidebar_state="expanded")


def norm(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


ALIASES = {
    "country": "Country",
    "year": "Year",
    "status": "Status",
    "lifeexpectancy": "Life_expectancy",
    "adultmortality": "Adult_Mortality",
    "infantdeaths": "Infant_deaths",
    "alcohol": "Alcohol",
    "percentageexpenditure": "Percentage_expenditure",
    "hepatitisb": "Hepatitis_B",
    "measles": "Measles",
    "bmi": "BMI",
    "underfivedeaths": "Under_five_deaths",
    "polio": "Polio",
    "totalexpenditure": "Total_expenditure",
    "diphtheria": "Diphtheria",
    "hivaids": "HIV_AIDS",
    "gdp": "GDP",
    "population": "Population",
    "thinness119years": "Thinness_1_19_years",
    "thinness59years": "Thinness_5_9_years",
    "incomecompositionofresources": "Income_composition_of_resources",
    "schooling": "Schooling",
    "mortalityburden": "Mortality_Burden",
    "immunizationscore": "Immunization_Score",
}

INPUT_SPECS = [
    ("Year", "Year", "General"),
    ("Life_expectancy", "Life expectancy", "General"),
    ("Adult_Mortality", "Adult mortality", "General"),
    ("Infant_deaths", "Infant deaths", "General"),
    ("Alcohol", "Alcohol consumption", "General"),
    ("Percentage_expenditure", "Percentage expenditure", "General"),
    ("Hepatitis_B", "Hepatitis B coverage", "Health"),
    ("Measles", "Measles cases", "Health"),
    ("BMI", "BMI", "Health"),
    ("Under_five_deaths", "Under-five deaths", "Health"),
    ("Polio", "Polio coverage", "Health"),
    ("Total_expenditure", "Total expenditure", "Health"),
    ("Diphtheria", "Diphtheria coverage", "Health"),
    ("HIV_AIDS", "HIV/AIDS indicator", "Health"),
    ("GDP", "GDP", "Economy & education"),
    ("Population", "Population", "Economy & education"),
    ("Thinness_1_19_years", "Thinness ages 1–19", "Economy & education"),
    ("Thinness_5_9_years", "Thinness ages 5–9", "Economy & education"),
    ("Income_composition_of_resources", "Income composition of resources", "Economy & education"),
    ("Schooling", "Schooling", "Economy & education"),
]

DEFAULTS = {
    "Year": 2015, "Life_expectancy": 70.0, "Adult_Mortality": 150.0,
    "Infant_deaths": 30.0, "Alcohol": 4.0, "Percentage_expenditure": 100.0,
    "Hepatitis_B": 85.0, "Measles": 100.0, "BMI": 25.0, "Under_five_deaths": 40.0,
    "Polio": 85.0, "Total_expenditure": 5.0, "Diphtheria": 85.0, "HIV_AIDS": 0.5,
    "GDP": 3000.0, "Population": 1_000_000.0, "Thinness_1_19_years": 5.0,
    "Thinness_5_9_years": 5.0, "Income_composition_of_resources": 0.6, "Schooling": 12.0,
}


def normalize_data(frame):
    df = frame.copy()
    df.rename(columns={c: ALIASES.get(norm(c), str(c).strip()) for c in df.columns}, inplace=True)
    for col in df.columns:
        if col not in {"Country", "Status"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Country" in df:
        df["Country"] = df["Country"].astype("string").str.strip()
    if "Status" in df:
        df["Status"] = df["Status"].astype("string").str.strip()
    if {"Adult_Mortality", "Infant_deaths"}.issubset(df.columns):
        df["Mortality_Burden"] = (df["Adult_Mortality"] + df["Infant_deaths"]) / 2
    if {"Hepatitis_B", "Polio", "Diphtheria"}.issubset(df.columns):
        df["Immunization_Score"] = df[["Hepatitis_B", "Polio", "Diphtheria"]].mean(axis=1)
    return df


@st.cache_data(show_spinner=False)
def read_csv_bytes(content):
    return normalize_data(pd.read_csv(io.BytesIO(content)))


def load_data(uploaded):
    if uploaded is not None:
        try:
            return read_csv_bytes(uploaded.getvalue()), uploaded.name
        except Exception as exc:
            st.sidebar.error(f"Could not read CSV: {exc}")
            return None, None
    for path in DATA_PATHS:
        if path.exists():
            try:
                return normalize_data(pd.read_csv(path)), str(path.relative_to(ROOT))
            except Exception as exc:
                st.sidebar.error(f"Could not read {path.name}: {exc}")
                return None, None
    return None, None


def read_export(name):
    path = RESULTS / name
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def country_row(df, country, year=None):
    if df is None or "Country" not in df:
        return None
    sub = df[df["Country"].astype(str).str.strip().str.casefold() == str(country).strip().casefold()]
    if year is not None and "Year" in sub:
        matched = sub[pd.to_numeric(sub["Year"], errors="coerce") == int(year)]
        if not matched.empty:
            sub = matched
    if sub.empty:
        return None
    if "Year" in sub:
        sub = sub.sort_values("Year", ascending=False)
    return sub.iloc[0]


def values_from_row(row):
    values = DEFAULTS.copy()
    if row is not None:
        for key in values:
            if key in row.index and pd.notna(row[key]):
                try:
                    values[key] = int(row[key]) if key == "Year" else float(row[key])
                except (TypeError, ValueError):
                    pass
    values["Mortality_Burden"] = (values["Adult_Mortality"] + values["Infant_deaths"]) / 2
    values["Immunization_Score"] = (
        values["Hepatitis_B"] + values["Polio"] + values["Diphtheria"]
    ) / 3
    return values


def model_input_frame(values, feature_columns):
    by_name = {norm(k): v for k, v in values.items()}
    missing, ordered = [], {}
    for feature in feature_columns:
        key = norm(feature)
        if key not in by_name:
            missing.append(feature)
        else:
            ordered[feature] = by_name[key]
    if missing:
        raise ValueError("Missing features: " + ", ".join(missing))
    return pd.DataFrame([ordered], columns=list(feature_columns))


def hero(kicker, title, description):
    st.markdown(
        f"""<div class="hero"><div class="kicker">{kicker}</div>
        <div class="hero-title">{title}</div><div class="hero-text">{description}</div></div>""",
        unsafe_allow_html=True,
    )


def kpi(label, value, note):
    st.markdown(
        f"""<div class="kpi"><div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>""",
        unsafe_allow_html=True,
    )


def chart_style(fig, height=390):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#CBD5E1"),
        title_font=dict(size=18, color="#F8FAFC"),
        margin=dict(l=20, r=20, t=65, b=35), height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#26364D")
    fig.update_yaxes(gridcolor="#26364D")
    return fig


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:Inter,sans-serif;}
.stApp {background:radial-gradient(circle at 90% 0%,rgba(37,99,235,.18),transparent 32%),#0A1020;color:#E5EDF8;}
[data-testid="stSidebar"] {background:#0E182A;border-right:1px solid #26364D;}
[data-testid="stSidebar"] h1 {color:#67E8F9;font-weight:800;}
h1,h2,h3 {color:#F8FAFC!important;font-weight:750!important;}
p,label,li {color:#CBD5E1;}
.hero {padding:29px 32px;border-radius:20px;background:linear-gradient(115deg,rgba(37,99,235,.30),rgba(8,145,178,.13));border:1px solid rgba(103,232,249,.22);margin:4px 0 24px;}
.kicker {color:#67E8F9;font-size:11px;font-weight:800;letter-spacing:2px;text-transform:uppercase;}
.hero-title {font-size:35px;font-weight:800;color:#fff;margin-top:9px;line-height:1.2;}
.hero-text {color:#B8C6DA;margin-top:12px;line-height:1.7;max-width:900px;}
.kpi {padding:19px;border-radius:16px;min-height:122px;background:linear-gradient(145deg,#121F34,#101A2B);border:1px solid #263853;}
.kpi-label {color:#9FB0C7;font-size:12px;font-weight:600;}
.kpi-value {color:#F8FAFC;font-size:27px;font-weight:800;margin-top:10px;}
.kpi-note {color:#67E8F9;font-size:11px;margin-top:6px;}
div[data-testid="stForm"] {background:#101B2E;padding:20px;border:1px solid #263853;border-radius:16px;}
.stButton>button,div[data-testid="stFormSubmitButton"] button {background:linear-gradient(90deg,#2563EB,#0891B2);color:#fff;border:0;border-radius:10px;min-height:43px;font-weight:700;}
div[data-baseweb="input"] input,div[data-baseweb="select"]>div {background:#152238;color:#F8FAFC;}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("# 🌍 LifeLens")
    st.caption("LIFE EXPECTANCY ML ANALYTICS")
    uploaded_csv = st.file_uploader(
        "WHO dataset CSV", type=["csv"],
        help="Recommended: put the CSV at data/Life Expectancy Data.csv in the project folder."
    )

data, data_source = load_data(uploaded_csv)
try:
    model_bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
except Exception as exc:
    model_bundle = None
    st.sidebar.error(f"Model bundle could not be loaded: {exc}")

st.sidebar.divider()
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Predict Status", "Classification", "Clustering",
     "Regression", "Dataset Explorer", "About Project"]
)
st.sidebar.divider()
if data is not None:
    st.sidebar.success(f"Dataset: {data_source}")
else:
    st.sidebar.warning("Dataset CSV not loaded")

if model_bundle is not None:
    st.sidebar.success("Prediction model loaded")
else:
    st.sidebar.info("Model export required")


# Exported result tables are used when present.
classification_df = read_export("classification_all10.csv")
classification_complete = classification_df is not None and len(classification_df) >= 10
if classification_df is None:
    # These are the five figures the user shared, not fabricated extra models.
    classification_df = pd.DataFrame({
        "Model": ["Logistic Regression", "K-Nearest Neighbors", "Gaussian Naive Bayes",
                  "Decision Tree Classifier", "Support Vector Machine"],
        "Accuracy": [0.9303, 0.9456, 0.8929, 0.9320, 0.9456],
        "F1_weighted": [0.9293, 0.9454, 0.8939, 0.9320, 0.9454],
        "ROC_AUC": [0.9683, 0.9661, 0.9467, 0.9336, 0.9728],
    })

clustering_df = read_export("clustering_metrics.csv")
if clustering_df is None:
    clustering_df = pd.DataFrame({
        "Algorithm": ["Agglomerative (complete)", "Agglomerative (average)", "K-Means", "Agglomerative (ward)"],
        "Number of clusters": [3, 3, 3, 3],
        "Silhouette Score (higher is better)": [0.7157, 0.7110, 0.2412, 0.2219],
        "Davies-Bouldin Index (lower is better)": [0.6618, 0.6783, 1.4297, 1.4506],
        "Calinski-Harabasz Index (higher is better)": [245.73, 248.97, 638.96, 584.36],
    })

regression_df = read_export("regression_comparison.csv")
regression_predictions = read_export("regression_predictions.csv")
elbow_df = read_export("elbow_curve.csv")
cluster_pca_df = read_export("clustering_pca.csv")
cluster_tsne_df = read_export("clustering_tsne.csv")


# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    hero("MACHINE LEARNING CAPSTONE", "Life Expectancy Analytics",
         "Explore country-level health, education and economic indicators using classification, regression and clustering.")

    total_records = f"{len(data):,}" if data is not None else "2,938"
    countries = f"{data['Country'].nunique():,}" if data is not None and "Country" in data else "—"
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Dataset records", total_records, "Loaded / reference dataset")
    with c2: kpi("Unique countries", countries, "From the CSV" if data is not None else "Load CSV to calculate")
    with c3: kpi("Classifiers", "10" if classification_complete else "5 / 10", "All-ten table needed" if not classification_complete else "Consolidated table loaded")
    with c4: kpi("Cluster configurations", str(len(clustering_df)), "K-Means + linkage variants")

    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Classification performance")
        metric = "Accuracy" if "Accuracy" in classification_df else classification_df.columns[1]
        plot_df = classification_df[["Model", metric]].copy()
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
        plot_df = plot_df.dropna().sort_values(metric)
        fig = px.bar(plot_df, x=metric, y="Model", orientation="h", color=metric,
                     color_continuous_scale=["#2563EB", "#22D3EE"],
                     title=f"Classifier comparison — {metric}")
        st.plotly_chart(chart_style(fig, max(340, 38 * len(plot_df))), use_container_width=True)
    with right:
        st.subheader("Class distribution")
        if data is not None and "Status" in data:
            counts = data["Status"].dropna().value_counts().rename_axis("Status").reset_index(name="Records")
        else:
            counts = pd.DataFrame({"Status": ["Developing", "Developed"], "Records": [2426, 512]})
        fig = px.pie(counts, names="Status", values="Records", hole=0.62,
                     title="Developing vs Developed",
                     color_discrete_sequence=["#3B82F6", "#22D3EE"])
        fig.update_traces(textposition="inside", textinfo="label+percent")
        st.plotly_chart(chart_style(fig, 340), use_container_width=True)

    st.subheader("Project pipeline")
    cols = st.columns(4)
    for col, title, desc in [
        (cols[0], "01 · Prepare", "Audit and engineer dataset features"),
        (cols[1], "02 · Train", "Fit classifiers and regressors"),
        (cols[2], "03 · Evaluate", "Compare metrics and cluster plots"),
        (cols[3], "04 · Predict", "Enter indicators and classify status"),
    ]:
        with col:
            st.markdown(f"**{title}**")
            st.caption(desc)

    if not classification_complete:
        st.warning("Only the five classification results shared so far are shown. Export the notebook's all-ten table to display all ten.")


# ============================================================
# PREDICTION — country match + editable fields
# ============================================================
elif page == "Predict Status":
    hero("INTERACTIVE MODEL", "Country Status Prediction",
         "Search for a country, load one of its recorded years to prefill the form, edit indicators, and submit to the trained model.")

    country, year, selected = None, None, None
    if data is not None and "Country" in data.columns:
        options = sorted(data["Country"].dropna().astype(str).unique().tolist())
        default_index = next((i for i, name in enumerate(options) if name.casefold() == "india"), 0)
        a, b = st.columns([2, 1])
        with a:
            country = st.selectbox("Search country in CSV", options, index=default_index,
                                   key="profile_country",
                                   help="This is a searchable list from your dataset. Searching 'india' will find 'India'.")
        subset = data[data["Country"].astype(str).str.strip().str.casefold() == country.strip().casefold()]
        years = sorted(pd.to_numeric(subset["Year"], errors="coerce").dropna().astype(int).unique().tolist(),
                       reverse=True) if "Year" in subset else []
        with b:
            year = st.selectbox("Year", years, index=0, key="profile_year") if years else None
        selected = country_row(data, country, year)
        action, tip = st.columns([1, 2])
        with action:
            load_record = st.button("↻ Load selected record", use_container_width=True)
        with tip:
            st.caption("Pick a country/year, then load its actual row into the inputs. You can edit every value before predicting.")
        if load_record and selected is not None:
            for key, value in values_from_row(selected).items():
                st.session_state[f"input_{key}"] = value
            st.session_state["last_loaded_record"] = f"{country} · {year}"
            st.rerun()
        if st.session_state.get("last_loaded_record"):
            st.success(f"Input values last loaded from: {st.session_state['last_loaded_record']}")
        st.caption("Country itself is not a model feature; it is used only to look up a dataset record.")
    else:
        st.info("For searchable country matching, add the CSV to data/Life Expectancy Data.csv or upload it from the sidebar.")

    initial_values = values_from_row(selected)
    for key, value in initial_values.items():
        st.session_state.setdefault(f"input_{key}", value)

    st.markdown("### Enter or edit country indicators")
    st.caption("Life expectancy is included because the current notebook's classifier uses it as one of its input features.")
    values = {}

    with st.form("prediction_form"):
        for group, heading in [
            ("General", "General indicators"),
            ("Health", "Health indicators"),
            ("Economy & education", "Economic and education indicators"),
        ]:
            st.markdown(f"#### {heading}")
            group_fields = [(key, label) for key, label, grp in INPUT_SPECS if grp == group]
            for start in range(0, len(group_fields), 3):
                cols = st.columns(3)
                for col, (key, label) in zip(cols, group_fields[start:start + 3]):
                    with col:
                        state_key = f"input_{key}"
                        if key == "Year":
                            values[key] = st.number_input(label, min_value=2000, max_value=2030, step=1, key=state_key)
                        elif key == "Life_expectancy":
                            values[key] = st.number_input(label, min_value=0.0, max_value=100.0, step=0.5, key=state_key)
                        elif key in {"Hepatitis_B", "Polio", "Diphtheria"}:
                            values[key] = st.number_input(label, min_value=0.0, max_value=100.0, step=1.0, key=state_key)
                        elif key == "Income_composition_of_resources":
                            values[key] = st.number_input(label, min_value=0.0, max_value=1.0, step=0.01, format="%.4f", key=state_key)
                        else:
                            values[key] = st.number_input(label, min_value=0.0, step=1.0,
                                                          format="%.2f" if key in {"Alcohol", "Total_expenditure"} else "%.2f",
                                                          key=state_key)

        values["Mortality_Burden"] = (values["Adult_Mortality"] + values["Infant_deaths"]) / 2
        values["Immunization_Score"] = (values["Hepatitis_B"] + values["Polio"] + values["Diphtheria"]) / 3
        m1, m2 = st.columns(2)
        with m1: st.metric("Mortality Burden · engineered", f"{values['Mortality_Burden']:.2f}")
        with m2: st.metric("Immunization Score · engineered", f"{values['Immunization_Score']:.2f}")
        submitted = st.form_submit_button("🔮 Predict country status", use_container_width=True)

    if submitted:
        if model_bundle is None:
            st.error("The input form works, but the trained model is not exported yet. Run the notebook export cell to create models/life_expectancy_model.joblib.")
        else:
            try:
                X_input = model_input_frame(values, model_bundle["feature_columns"])
                processed = model_bundle["preprocessor"].transform(X_input)
                reduced = model_bundle["pca"].transform(processed)
                model = model_bundle["model"]
                prediction = int(model.predict(reduced)[0])
                label = {0: "Developing", 1: "Developed"}.get(prediction, str(prediction))
                st.divider()
                st.subheader("Prediction result")
                if prediction == 1:
                    st.success(f"Predicted status: **{label}**")
                else:
                    st.info(f"Predicted status: **{label}**")
                st.caption("This is a machine-learning output, not an official country classification.")
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(reduced)[0]
                    class_names = [{0: "Developing", 1: "Developed"}.get(int(c), str(c))
                                   for c in model.classes_]
                    prob_df = pd.DataFrame({"Status": class_names, "Probability": probabilities})
                    fig = px.bar(prob_df, x="Status", y="Probability", text=prob_df["Probability"].map(lambda x: f"{x:.1%}"),
                                 color="Status", title="Model output probabilities",
                                 color_discrete_sequence=["#3B82F6", "#22D3EE"])
                    fig.update_yaxes(tickformat=".0%")
                    st.plotly_chart(chart_style(fig, 300), use_container_width=True)
            except Exception as exc:
                st.error("Prediction failed. Confirm the exported preprocessor, PCA, feature list and classifier match the notebook.")
                st.exception(exc)


# ============================================================
# CLASSIFICATION — actual all-ten export or clearly marked fallback
# ============================================================
elif page == "Classification":
    hero("MODEL EVALUATION", "Classification Laboratory",
         "Compare all ten required classifiers using Accuracy, weighted Precision, weighted Recall, weighted F1 and ROC-AUC.")

    if not classification_complete:
        st.warning("This is currently the five-model fallback from the results you shared. Export classification_all10_df from your notebook to show all ten.")

    metric_pairs = [
        ("Accuracy", "Accuracy"),
        ("Precision (weighted)", "Precision_weighted"),
        ("Recall (weighted)", "Recall_weighted"),
        ("F1-score (weighted)", "F1_weighted"),
        ("ROC-AUC", "ROC_AUC"),
    ]
    available = [(label, col) for label, col in metric_pairs if col in classification_df.columns]
    if available:
        chosen = st.selectbox("Metric", [x[0] for x in available])
        col_name = next(col for label, col in available if label == chosen)
        plot_df = classification_df[["Model", col_name]].copy()
        plot_df[col_name] = pd.to_numeric(plot_df[col_name], errors="coerce")
        plot_df = plot_df.dropna().sort_values(col_name)
        fig = px.bar(plot_df, x=col_name, y="Model", orientation="h", color=col_name,
                     color_continuous_scale=["#2563EB", "#22D3EE"], title=f"Comparison — {chosen}")
        if plot_df[col_name].between(0, 1).all():
            fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(chart_style(fig, max(360, len(plot_df) * 38)), use_container_width=True)

    st.subheader("Consolidated model table")
    st.dataframe(classification_df, use_container_width=True, hide_index=True)


# ============================================================
# CLUSTERING — metrics, elbow, PCA and t-SNE when exported
# ============================================================
elif page == "Clustering":
    hero("UNSUPERVISED LEARNING", "Clustering Studio",
         "Compare K-Means and hierarchical clustering using internal metrics and visual cluster projections.")

    silhouette_col = next((c for c in clustering_df if "Silhouette" in c), None)
    db_col = next((c for c in clustering_df if "Davies" in c), None)
    if silhouette_col and db_col:
        a, b = st.columns(2)
        with a:
            kpi("Best Silhouette", f"{pd.to_numeric(clustering_df[silhouette_col], errors='coerce').max():.4f}", "Higher is generally better")
        with b:
            kpi("Lowest Davies–Bouldin", f"{pd.to_numeric(clustering_df[db_col], errors='coerce').min():.4f}", "Lower is generally better")

    st.subheader("Clustering metrics")
    st.dataframe(clustering_df, use_container_width=True, hide_index=True)

    if elbow_df is not None and {"k", "inertia"}.issubset(elbow_df.columns):
        fig = px.line(elbow_df, x="k", y="inertia", markers=True, title="K-Means elbow curve")
        st.plotly_chart(chart_style(fig, 340), use_container_width=True)
    else:
        st.info("Export elbow_k_values and elbow_inertia from the notebook to show the actual elbow curve.")

    if cluster_pca_df is not None and {"Algorithm", "PC1", "PC2", "Cluster"}.issubset(cluster_pca_df.columns):
        choices = sorted(cluster_pca_df["Algorithm"].astype(str).unique())
        which = st.selectbox("PCA cluster configuration", choices)
        subset = cluster_pca_df[cluster_pca_df["Algorithm"].astype(str) == which].copy()
        subset["Cluster"] = subset["Cluster"].astype(str)
        fig = px.scatter(subset, x="PC1", y="PC2", color="Cluster", opacity=0.72,
                         title=f"PCA 2D cluster view — {which}",
                         color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(chart_style(fig, 480), use_container_width=True)
    else:
        st.info("Export X_cluster_pca_2d and cluster_label_sets to results/clustering_pca.csv to show PCA scatter plots.")

    if cluster_tsne_df is not None and {"TSNE1", "TSNE2", "Cluster"}.issubset(cluster_tsne_df.columns):
        subset = cluster_tsne_df.copy()
        subset["Cluster"] = subset["Cluster"].astype(str)
        fig = px.scatter(subset, x="TSNE1", y="TSNE2", color="Cluster", opacity=0.72,
                         title="t-SNE visualization — K-Means",
                         color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(chart_style(fig, 450), use_container_width=True)


# ============================================================
# REGRESSION — use notebook predictions; otherwise show real dataset EDA
# ============================================================
elif page == "Regression":
    hero("REGRESSION ANALYSIS", "Life Expectancy Regression",
         "Review regression metrics and inspect actual-versus-predicted and residual plots from the notebook.")

    if regression_df is not None and {"Model", "R2", "RMSE", "MAE"}.issubset(regression_df.columns):
        st.subheader("Regression model comparison")
        metric = st.selectbox("Metric", ["R2", "RMSE", "MAE"])
        plot_df = regression_df[["Model", metric]].copy()
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
        plot_df = plot_df.dropna().sort_values(metric, ascending=(metric != "R2"))
        fig = px.bar(plot_df, x=metric, y="Model", orientation="h", color=metric,
                     color_continuous_scale=["#2563EB", "#22D3EE"], title=f"Regression algorithms — {metric}")
        st.plotly_chart(chart_style(fig, max(350, 34 * len(plot_df))), use_container_width=True)
        st.dataframe(regression_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Regression model metrics are not exported yet. Run the notebook export cell to create results/regression_comparison.csv.")
        if data is not None and "Life_expectancy" in data:
            fig = px.histogram(data, x="Life_expectancy", nbins=30, title="Life expectancy distribution in the dataset",
                               color_discrete_sequence=["#22D3EE"])
            st.plotly_chart(chart_style(fig, 340), use_container_width=True)

    if regression_predictions is not None and {"Actual_Life_expectancy", "Predicted_Life_expectancy"}.issubset(regression_predictions.columns):
        st.subheader("Predicted vs actual")
        pred = regression_predictions
        fig = px.scatter(pred, x="Actual_Life_expectancy", y="Predicted_Life_expectancy",
                         opacity=0.6, title="Held-out test set: predicted vs actual")
        low = min(pred["Actual_Life_expectancy"].min(), pred["Predicted_Life_expectancy"].min())
        high = max(pred["Actual_Life_expectancy"].max(), pred["Predicted_Life_expectancy"].max())
        fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="Ideal prediction"))
        st.plotly_chart(chart_style(fig, 410), use_container_width=True)
        if "Residual" in pred:
            fig = px.scatter(pred, x="Predicted_Life_expectancy", y="Residual", opacity=0.6,
                             title="Residuals vs predicted life expectancy")
            fig.add_hline(y=0, line_dash="dash", line_color="#22D3EE")
            st.plotly_chart(chart_style(fig, 380), use_container_width=True)
    elif data is not None and {"Schooling", "Life_expectancy"}.issubset(data.columns):
        st.subheader("Dataset relationship (exploratory, not model predictions)")
        fig = px.scatter(data.dropna(subset=["Schooling", "Life_expectancy"]), x="Schooling", y="Life_expectancy",
                         color="Status" if "Status" in data else None, opacity=0.55,
                         title="Schooling vs life expectancy",
                         color_discrete_sequence=["#3B82F6", "#22D3EE"])
        st.plotly_chart(chart_style(fig, 380), use_container_width=True)


# ============================================================
# DATASET EXPLORER — genuine CSV-backed plots
# ============================================================
elif page == "Dataset Explorer":
    hero("DATA EXPLORATION", "Explore the WHO Dataset",
         "The dataset is used directly for real summary statistics, missing-value counts and feature visualizations.")
    if data is None:
        st.warning("Add the real dataset at data/Life Expectancy Data.csv or upload it from the sidebar.")
    else:
        numeric = data.select_dtypes(include=np.number).columns.tolist()
        missing_total = int(data.isna().sum().sum())
        country_count = data["Country"].nunique() if "Country" in data else "—"
        a, b, c, d = st.columns(4)
        with a: kpi("Records", f"{len(data):,}", "Rows")
        with b: kpi("Columns", str(len(data.columns)), "Normalized field names")
        with c: kpi("Missing cells", f"{missing_total:,}", "All columns")
        with d: kpi("Countries", f"{country_count:,}" if isinstance(country_count, (int, np.integer)) else str(country_count), "Unique names")

        st.subheader("Dataset preview")
        st.dataframe(data.head(25), use_container_width=True, hide_index=True)

        st.subheader("Feature distribution")
        if numeric:
            feature = st.selectbox("Select a numeric feature", numeric)
            vals = data[feature].dropna()
            fig = px.histogram(data, x=feature, nbins=35, marginal="box", title=f"Distribution of {feature}",
                               color_discrete_sequence=["#22D3EE"])
            st.plotly_chart(chart_style(fig, 410), use_container_width=True)
            st.dataframe(vals.describe().to_frame("Value").round(3), use_container_width=True)

        st.subheader("Missing values by feature")
        miss = data.isna().sum().sort_values(ascending=False).rename_axis("Feature").reset_index(name="Missing values")
        miss = miss[miss["Missing values"] > 0]
        if miss.empty:
            st.success("No missing values found in this CSV.")
        else:
            fig = px.bar(miss, x="Missing values", y="Feature", orientation="h", title="Missing-value counts",
                         color="Missing values", color_continuous_scale=["#2563EB", "#22D3EE"])
            st.plotly_chart(chart_style(fig, max(320, len(miss) * 27)), use_container_width=True)

        if len(numeric) >= 2:
            st.subheader("Feature relationship")
            c1, c2 = st.columns(2)
            with c1: x_col = st.selectbox("X-axis", numeric, index=0, key="explorer_x")
            with c2: y_col = st.selectbox("Y-axis", numeric, index=min(1, len(numeric)-1), key="explorer_y")
            plot_data = data.dropna(subset=[x_col, y_col])
            fig = px.scatter(plot_data, x=x_col, y=y_col, color="Status" if "Status" in data else None,
                             hover_data=["Country", "Year"] if {"Country", "Year"}.issubset(data.columns) else None,
                             opacity=0.55, title=f"{x_col} vs {y_col}",
                             color_discrete_sequence=["#3B82F6", "#22D3EE"])
            st.plotly_chart(chart_style(fig, 420), use_container_width=True)

        if len(numeric) >= 3:
            with st.expander("Show correlation heatmap"):
                fig = px.imshow(data[numeric].corr(), zmin=-1, zmax=1, aspect="auto",
                                color_continuous_scale="RdBu_r", title="Numeric feature correlations")
                st.plotly_chart(chart_style(fig, 600), use_container_width=True)


# ============================================================
# ABOUT
# ============================================================
elif page == "About Project":
    hero("PROJECT DOCUMENTATION", "About LifeLens",
         "A machine-learning capstone for life expectancy classification, regression and clustering.")
    st.subheader("Objective")
    st.write("Explore country-level health, education and economic indicators, compare machine-learning models, inspect cluster structure and demonstrate a trained classification model through an interactive form.")
    a, b, c = st.columns(3)
    with a:
        st.markdown("### 01 · Prepare")
        st.caption("Data audit, imputation, outlier handling and engineered features.")
    with b:
        st.markdown("### 02 · Train")
        st.caption("Regression, classification, PCA and hyperparameter tuning.")
    with c:
        st.markdown("### 03 · Evaluate")
        st.caption("Accuracy/F1/ROC-AUC, R²/RMSE/MAE, and cluster metrics.")
    st.markdown("""
    **Technologies:** Python, Pandas, NumPy, scikit-learn, Plotly and Streamlit.

    **Classification target:** Developing = 0, Developed = 1.

    For submission, include the dataset description, environment setup, results summary,
    run instructions, meaningful Git history and acknowledgement of AI-assisted scaffolding
    as required by the capstone guidelines.
    """)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#718198;font-size:12px'>LIFELENS · LIFE EXPECTANCY MACHINE LEARNING PROJECT</p>", unsafe_allow_html=True)
