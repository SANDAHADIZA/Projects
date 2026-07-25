import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AfriRisk AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent

# ---------------------------------------------------------
# LOAD REAL MODEL + DATA (trained in Africa_Recession_Prediction.ipynb)
# ---------------------------------------------------------

@st.cache_resource
def load_model_artifacts():
    model = joblib.load(BASE_DIR / "models" / "recession_model.joblib")
    scaler = joblib.load(BASE_DIR / "models" / "scaler.joblib")
    feature_cols = joblib.load(BASE_DIR / "models" / "feature_columns.joblib")
    return model, scaler, feature_cols

@st.cache_data
def load_training_data():
    return pd.read_csv(BASE_DIR / "africa_recession.csv")

@st.cache_data
def load_model_comparison():
    return pd.read_csv(BASE_DIR / "model_comparison_results.csv")

model, scaler, FEATURE_COLS = load_model_artifacts()
df = load_training_data()
comparison_df = load_model_comparison()

# ---------------------------------------------------------
# LOAD COUNTRY-IDENTIFIED MODEL (trained in Country_Recession_Prediction.ipynb
# on real World Bank WDI data - has actual country names, unlike the model above)
# ---------------------------------------------------------

@st.cache_resource
def load_country_model_artifacts():
    m = joblib.load(BASE_DIR / "country_models" / "country_recession_model.joblib")
    s = joblib.load(BASE_DIR / "country_models" / "country_scaler.joblib")
    imp = joblib.load(BASE_DIR / "country_models" / "country_imputer.joblib")
    cols = joblib.load(BASE_DIR / "country_models" / "country_feature_columns.joblib")
    return m, s, imp, cols

@st.cache_data
def load_country_data():
    risk_full = pd.read_csv(BASE_DIR / "country_year_risk_full.csv")
    latest_risk = pd.read_csv(BASE_DIR / "latest_country_risk.csv")
    comparison = pd.read_csv(BASE_DIR / "country_model_comparison_results.csv")
    return risk_full, latest_risk, comparison

country_model, country_scaler, country_imputer, COUNTRY_FEATURE_COLS = load_country_model_artifacts()
country_risk_full, latest_country_risk, country_comparison_df = load_country_data()
country_best_row = country_comparison_df.sort_values("F1", ascending=False).iloc[0]

MEDIANS = df[FEATURE_COLS].median()

# Top features by importance — used to build a manageable input form.
# (Fitting a form to all 49 raw indicators is not usable for a demo; the
#  remaining features are held at their dataset median.)
importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
TOP_FEATURES = importances.head(10).index.tolist()

FEATURE_LABELS = {
    "rdana": "Real domestic absorption (consumption + investment)",
    "cwtfp": "Welfare-relevant total factor productivity",
    "cn": "Capital stock (current PPPs)",
    "energy": "Bank of Canada energy commodity price index",
    "ck": "Capital services level (current PPPs)",
    "excl_energy_change": "YoY change, commodity index excl. energy",
    "rkna": "Capital services (constant national prices)",
    "csh_r": "Share of residual trade / GDP discrepancy",
    "pl_n": "Price level of capital stock",
    "fish_change": "YoY change, fish commodity price index",
}


def predict_from_inputs(user_values: dict) -> tuple[float, int]:
    """Build a full 49-feature row (user inputs + medians for the rest), scale, predict."""
    row = MEDIANS.copy()
    for k, v in user_values.items():
        row[k] = v
    X = pd.DataFrame([row])[FEATURE_COLS]
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[0, 1]
    pred = model.predict(X_scaled)[0]
    return proba, pred


def risk_tier(prob):
    if prob >= 0.6:
        return "High", "🔴"
    elif prob >= 0.3:
        return "Moderate", "🟡"
    return "Low", "🟢"


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("🌍 AfriRisk AI")
st.sidebar.markdown("**Economic Intelligence & Early Warning Platform**")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🗺️ Country Explorer",
        "📊 Data Dashboard",
        "🔮 Recession Predictor",
        "⚖️ Scenario Comparison",
        "⚠️ Batch Early Warning",
        "ℹ️ About",
    ],
)

st.sidebar.markdown("---")
best_row = comparison_df.sort_values("F1", ascending=False).iloc[0]
st.sidebar.success(
    f"✅ Live model: **{best_row['Model']}**\n\n"
    f"F1: {best_row['F1']:.2f} · ROC-AUC: {best_row['ROC_AUC']:.2f}\n\n"
    "Trained on africa_recession.csv (486 obs., 49 features, SMOTE-balanced)."
)
st.sidebar.info(
    "⚠️ The main model's training data has no country/year identifiers, so its "
    "predictions are indicator-based, not tied to a named country. For real named "
    "countries, see **🗺️ Country Explorer** — powered by a second model trained "
    "on real World Bank data."
)

# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

if page == "🏠 Home":
    st.title("🌍 AfriRisk AI")
    st.subheader("Know the Economic Risk Before It Becomes a Crisis.")
    st.markdown(
        "AfriRisk AI is an economic intelligence platform that estimates recession "
        "risk from macroeconomic indicators using a trained machine-learning model — "
        "not sample numbers."
    )
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observations Trained On", f"{len(df)}")
    col2.metric("Recession Cases in Data", f"{int(df['growthbucket'].sum())}")
    col3.metric("Best Model", best_row["Model"])
    col4.metric("Held-out F1 Score", f"{best_row['F1']:.2f}")

    st.markdown("---")
    st.header("🎯 The Problem")
    st.write(
        "Economic information is scattered across sources, and recessions are rare, "
        "imbalanced events — a model can look 90%+ accurate while still missing every "
        "real recession. That accuracy paradox is exactly what this project addresses."
    )
    st.header("💡 Our Solution")
    st.write(
        "We combine macroeconomic indicators with SMOTE class-balancing and a "
        "comparison of Logistic Regression, Random Forest, and XGBoost, then deploy "
        "the strongest model (by F1 and ROC-AUC, not raw accuracy) here."
    )

    st.markdown("---")
    st.header("📈 Model Comparison (held-out test set)")
    st.dataframe(
        comparison_df.style.format({c: "{:.3f}" for c in comparison_df.columns if c != "Model"}),
        use_container_width=True, hide_index=True,
    )
    fig = px.bar(comparison_df, x="Model", y="F1", text="F1", color="Model",
                 color_discrete_sequence=["#2E86AB", "#6C757D", "#D4A94A"])
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# COUNTRY EXPLORER (real countries, real World Bank data)
# ---------------------------------------------------------

elif page == "🗺️ Country Explorer":
    st.title("🗺️ Country Explorer")
    st.write(
        "Real, named African countries — data pulled from the World Bank's Indicators "
        "API (2000–2017). This model is separate from the anonymized-data model on other "
        "pages, and its held-out performance is meaningfully weaker (details below), "
        "because predicting one country's GDP direction a year out is genuinely hard."
    )
    st.warning(
        f"⚠️ Recession label here = negative real GDP growth (a simple proxy, not an "
        f"official designation). Held-out F1: {country_best_row['F1']:.2f} · "
        f"ROC-AUC: {country_best_row['ROC_AUC']:.2f} · Model: {country_best_row['Model']}. "
        "Read this as a directional signal, not a verdict."
    )
    st.markdown("---")

    tab1, tab2 = st.tabs(["Latest Risk by Country", "Country History"])

    with tab1:
        st.subheader(f"Fitted Recession Probability — most recent year available per country")
        c1, c2 = st.columns([2, 1])
        with c2:
            top_n = st.slider("Show top N highest-risk countries", 5, 54, 20)
        display_df = latest_country_risk.head(top_n).copy()
        display_df["fitted_recession_probability"] = (display_df["fitted_recession_probability"] * 100).round(1)
        display_df = display_df.rename(columns={
            "fitted_recession_probability": "Recession Probability (%)",
            "country": "Country", "year": "Year", "iso3": "ISO3"
        })
        fig = px.bar(
            display_df, x="Recession Probability (%)", y="Country", orientation="h",
            color="Recession Probability (%)", color_continuous_scale="RdYlGn_r",
            hover_data=["Year"],
        )
        fig.update_layout(height=max(400, top_n * 22), yaxis={"categoryorder": "total ascending"},
                           coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tab2:
        countries = sorted(country_risk_full["country"].unique())
        selected_country = st.selectbox("Select a country", countries, index=countries.index("Nigeria") if "Nigeria" in countries else 0)
        hist = country_risk_full[country_risk_full["country"] == selected_country].sort_values("year")

        c1, c2, c3 = st.columns(3)
        latest_row = hist.iloc[-1] if len(hist) else None
        if latest_row is not None:
            prob = latest_row["fitted_recession_probability"]
            tier, emoji = risk_tier(prob)
            c1.metric("Latest Year Available", int(latest_row["year"]))
            c2.metric("Recession Probability", f"{prob*100:.1f}%")
            c3.metric("Risk Level", f"{emoji} {tier}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["year"], y=hist["fitted_recession_probability"] * 100,
            mode="lines+markers", line=dict(color="#D4A94A", width=3),
            name="Recession probability",
        ))
        actual_recessions = hist[hist["recession"] == 1]
        fig.add_trace(go.Scatter(
            x=actual_recessions["year"], y=[105] * len(actual_recessions),
            mode="markers", marker=dict(color="#EF4444", size=10, symbol="triangle-down"),
            name="Actual recession year",
        ))
        fig.update_layout(
            title=f"{selected_country} — fitted recession probability over time",
            yaxis_title="Probability (%)", yaxis_range=[0, 110], height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Red triangles mark years where real GDP growth was actually negative "
            "(the label the model was trained to predict, one year ahead)."
        )

# ---------------------------------------------------------
# DATA DASHBOARD
# ---------------------------------------------------------

elif page == "📊 Data Dashboard":
    st.title("📊 Data Dashboard")
    st.write("Exploratory view of the training data behind the model.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Observations", len(df))
    col2.metric("Recession Rate", f"{df['growthbucket'].mean()*100:.1f}%")
    col3.metric("Features", len(FEATURE_COLS))

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Class Balance")
        counts = df["growthbucket"].value_counts().rename({0: "No Recession", 1: "Recession"})
        fig = px.pie(values=counts.values, names=counts.index,
                     color_discrete_sequence=["#2E86AB", "#D4A94A"])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Top Features Correlated with Recession")
        corr = df.corr(numeric_only=True)["growthbucket"].drop("growthbucket")
        top_corr = pd.concat([corr.sort_values().head(6), corr.sort_values().tail(6)])
        fig = px.bar(x=top_corr.values, y=top_corr.index, orientation="h",
                     color=top_corr.values, color_continuous_scale="RdYlGn")
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Correlation", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Importance (XGBoost)")
    imp_df = importances.head(15).sort_values()
    fig = px.bar(x=imp_df.values, y=imp_df.index, orientation="h",
                 color_discrete_sequence=["#D4A94A"])
    fig.update_layout(xaxis_title="Importance", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View raw training data"):
        st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------
# RECESSION PREDICTOR
# ---------------------------------------------------------

elif page == "🔮 Recession Predictor":
    st.title("🔮 AI Recession Predictor")
    st.write(
        "Adjust the indicators below (the 10 most influential features in the trained "
        "model) to see how recession probability responds. Remaining indicators are "
        "held at their dataset median."
    )
    st.markdown("---")

    inputs = {}
    cols = st.columns(2)
    for i, feat in enumerate(TOP_FEATURES):
        col = cols[i % 2]
        lo, hi, med = float(df[feat].min()), float(df[feat].max()), float(MEDIANS[feat])
        with col:
            inputs[feat] = st.slider(
                FEATURE_LABELS.get(feat, feat), min_value=lo, max_value=hi, value=med,
                help=f"Raw variable: `{feat}` · dataset range [{lo:.2f}, {hi:.2f}]",
            )

    st.markdown("---")
    if st.button("Run Prediction", type="primary"):
        proba, pred = predict_from_inputs(inputs)
        tier, emoji = risk_tier(proba)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predicted Recession Probability", f"{proba*100:.1f}%")
            st.metric("Risk Level", f"{emoji} {tier}")
        with c2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=proba * 100,
                title={"text": "Recession Risk (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#D4A94A"},
                    "steps": [
                        {"range": [0, 30], "color": "#1f8f4e33"},
                        {"range": [30, 60], "color": "#d97a0633"},
                        {"range": [60, 100], "color": "#dc262633"},
                    ],
                    "threshold": {"line": {"width": 4, "color": "#dc2626"}, "value": proba * 100},
                },
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# SCENARIO COMPARISON
# ---------------------------------------------------------

elif page == "⚖️ Scenario Comparison":
    st.title("⚖️ Scenario Comparison")
    st.write(
        "Compare two indicator scenarios side by side — useful for stress-testing "
        "'what if domestic activity drops' type questions, since named-country data "
        "isn't available in this dataset."
    )
    st.markdown("---")

    scenario_cols = st.columns(2)
    scenario_inputs = [{}, {}]
    for s_idx, col in enumerate(scenario_cols):
        with col:
            st.subheader(f"Scenario {'A' if s_idx == 0 else 'B'}")
            for feat in TOP_FEATURES[:6]:
                lo, hi, med = float(df[feat].min()), float(df[feat].max()), float(MEDIANS[feat])
                scenario_inputs[s_idx][feat] = st.slider(
                    FEATURE_LABELS.get(feat, feat), min_value=lo, max_value=hi, value=med,
                    key=f"{feat}_{s_idx}",
                )

    if st.button("Compare Scenarios", type="primary"):
        proba_a, _ = predict_from_inputs(scenario_inputs[0])
        proba_b, _ = predict_from_inputs(scenario_inputs[1])
        tier_a, emoji_a = risk_tier(proba_a)
        tier_b, emoji_b = risk_tier(proba_b)

        c1, c2 = st.columns(2)
        c1.metric("Scenario A — Recession Probability", f"{proba_a*100:.1f}%", f"{emoji_a} {tier_a}")
        c2.metric("Scenario B — Recession Probability", f"{proba_b*100:.1f}%", f"{emoji_b} {tier_b}")

        fig = px.bar(x=["Scenario A", "Scenario B"], y=[proba_a * 100, proba_b * 100],
                     text=[f"{proba_a*100:.1f}%", f"{proba_b*100:.1f}%"],
                     color=["Scenario A", "Scenario B"],
                     color_discrete_sequence=["#2E86AB", "#D4A94A"])
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, yaxis_range=[0, 100], yaxis_title="Recession Probability (%)")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# BATCH EARLY WARNING
# ---------------------------------------------------------

elif page == "⚠️ Batch Early Warning":
    st.title("⚠️ Batch Early Warning System")
    st.write(
        "Upload a CSV with one row per country (or country-year) containing the same "
        "49 indicator columns as the training data. Each row gets a real prediction "
        "from the trained model."
    )
    st.caption(f"Required columns: {', '.join(FEATURE_COLS)}")
    st.markdown("---")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        missing = [c for c in FEATURE_COLS if c not in batch_df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            X_scaled = scaler.transform(batch_df[FEATURE_COLS])
            batch_df["recession_probability"] = model.predict_proba(X_scaled)[:, 1]
            batch_df["predicted_recession"] = model.predict(X_scaled)
            batch_df["risk_level"] = batch_df["recession_probability"].apply(lambda p: risk_tier(p)[0])

            id_col = next((c for c in batch_df.columns if c.lower() in ("country", "name", "id")), None)
            display_cols = ([id_col] if id_col else []) + ["recession_probability", "risk_level"]

            st.subheader("Predictions")
            st.dataframe(
                batch_df[display_cols].sort_values("recession_probability", ascending=False),
                use_container_width=True, hide_index=True,
            )

            st.markdown("---")
            st.subheader("Alerts")
            for _, row in batch_df.sort_values("recession_probability", ascending=False).iterrows():
                label = row[id_col] if id_col else f"Row {row.name}"
                p = row["recession_probability"]
                if p >= 0.6:
                    st.error(f"🔴 HIGH RISK — {label}: {p*100:.1f}% recession probability")
                elif p >= 0.3:
                    st.warning(f"🟡 MODERATE RISK — {label}: {p*100:.1f}% recession probability")
                else:
                    st.success(f"🟢 LOW RISK — {label}: {p*100:.1f}% recession probability")

            st.download_button(
                "Download predictions as CSV",
                batch_df.to_csv(index=False).encode("utf-8"),
                "recession_predictions.csv", "text/csv",
            )
    else:
        st.info("No file uploaded yet. Once you have a country-identified dataset, upload it here.")

# ---------------------------------------------------------
# ABOUT
# ---------------------------------------------------------

elif page == "ℹ️ About":
    st.title("ℹ️ About AfriRisk AI")
    st.subheader("Our Mission")
    st.write("To make economic risk intelligence more accessible and actionable for organizations operating across African markets.")

    st.subheader("🧠 Methodology — two models, two purposes")
    st.markdown("**Model 1 — Anonymized indicator model** (`Africa_Recession_Prediction.ipynb`)")
    st.write(
        "- Data: 486 country-year observations, 49 macroeconomic features (Penn World "
        "Table–style variables + Bank of Canada commodity indices)\n"
        "- Imbalance handled with SMOTE, applied to the training fold only\n"
        f"- Compared Logistic Regression, Random Forest, and XGBoost — "
        f"**{best_row['Model']}** won on F1 ({best_row['F1']:.2f}) and ROC-AUC ({best_row['ROC_AUC']:.2f})\n"
        "- No country/year identifiers in the source data (confirmed limitation of the "
        "public Kaggle release) — powers the Recession Predictor / Scenario Comparison / "
        "Batch Early Warning pages"
    )
    st.markdown("**Model 2 — Country-identified model** (`Country_Recession_Prediction.ipynb`)")
    st.write(
        "- Data: real World Bank WDI indicators, 54 African countries, 2000–2017 "
        "(`fetch_external_data.py` / `Fetch_External_Data.ipynb`)\n"
        "- Recession label: negative real GDP growth (a proxy, not an official designation)\n"
        "- Features lagged one year to avoid leaking same-year GDP growth\n"
        f"- **{country_best_row['Model']}** won on F1 ({country_best_row['F1']:.2f}), "
        f"ROC-AUC ({country_best_row['ROC_AUC']:.2f}) — meaningfully weaker than Model 1, "
        "honestly, because this is a harder real-world prediction task\n"
        "- Powers the Country Explorer page"
    )

    st.subheader("⚠️ Known Limitations")
    st.warning(
        "Model 1's training data has no country or year identifiers, so it can't "
        "attribute a prediction to a named country — that's what Model 2 (Country "
        "Explorer) is for. Model 2, in turn, is built on a simplified recession "
        "label and has real-world predictive limits — treat its output as a "
        "directional signal, not a verdict."
    )

    st.subheader("🎯 Target Users")
    for u in ["Investors and investment firms", "Banks and financial institutions",
              "Businesses expanding into African markets", "Government and policymakers",
              "NGOs and development organizations"]:
        st.write(f"• {u}")

    st.subheader("🚀 Future Development")
    st.write(
        "- Source a country/year-indexed version of the data to enable live per-country tracking\n"
        "- Merge external series (World Bank WDI, IMF WEO, UNCTAD uncertainty index) by country-year\n"
        "- Explainable AI (SHAP) for per-prediction driver breakdown\n"
        "- Automated alerting and API access"
    )
