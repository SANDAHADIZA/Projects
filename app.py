import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    RandomForestRegressor, AdaBoostRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    recall_score, f1_score, precision_score, accuracy_score,
    average_precision_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)
from scipy.stats import pearsonr

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# ===========================================================================
# 🏷️ Friendly Business Labels Dictionary & Helper Function
# ===========================================================================
VARIABLE_LABELS = {
    "rwtfpna": "Welfare TFP (National Prices)",
    "rtfpna": "Total Factor Productivity (TFP)",
    "pl_n": "Price Level of Capital Stock",
    "pl_i": "Price Level of Capital Formation",
    "pl_g": "Price Level of Govt Consumption",
    "pl_con": "Price Level of Household Consumption",
    "pl_gdpo": "Price Level of CGDPo",
    "csh_i": "Investment Share of GDP",
    "csh_c": "Consumption Share of GDP",
    "csh_g": "Govt Expenditure Share of GDP",
    "csh_x": "Merchandise Export Share",
    "csh_m": "Merchandise Import Share",
    "xr": "Exchange Rate (National Currency / USD)",
    "delta": "Capital Stock Depreciation Rate",
    "irr": "Real Internal Rate of Return",
    "rdana": "Real Domestic Absorption (Consumption + Investment)",
    "rconna": "Real Consumption (National Prices)",
    "emp_to_pop_ratio": "Employment-to-Population Ratio",
    "labsh": "Labor Compensation Share of GDP",
    "total_change": "Total Commodity Index YoY Change",
    "energy_change": "Energy Commodity Index YoY Change",
    "metals_minerals_change": "Metals & Minerals Index YoY Change",
    "agriculture_change": "Agriculture Commodity Index YoY Change",
    "growthbucket": "Recession Event Indicator (Target)"
}

def get_label(col_name):
    """Returns human-readable industry label if available."""
    if col_name in VARIABLE_LABELS:
        return f"{VARIABLE_LABELS[col_name]} ({col_name})"
    return col_name

# ===========================================================================

st.set_page_config(page_title="Sovereign Risk Early Warning System", page_icon="🏦", layout="wide")

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
defaults = {
    "trained": False, "results": {}, "best_model_name": None,
    "scaler": None, "feature_cols": [], "feature_stats": {},
    "metrics_df": None, "cleaning_log": [], "encoders": {},
    "insight_text": None, "raw_df": None, "target_col": None,
    "target_type": "Binary classification",
    "smote_before": None, "smote_after": None, "smote_applied": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

CLASSIFICATION_MODEL_BUILDERS = {
    "Logistic Regression": lambda: (
        LogisticRegression(max_iter=1000, random_state=42),
        {"C": [0.01, 0.1, 1, 10, 100]},
    ),
    "Random Forest": lambda: (
        RandomForestClassifier(random_state=42),
        {"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10, 20], "min_samples_leaf": [1, 2, 4]},
    ),
    "SVM": lambda: (
        SVC(probability=True, random_state=42),
        {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
    ),
    "AdaBoost": lambda: (
        AdaBoostClassifier(random_state=42),
        {"n_estimators": [50, 100, 200], "learning_rate": [0.25, 0.5, 1.0]},
    ),
}
if XGB_AVAILABLE:
    CLASSIFICATION_MODEL_BUILDERS["XGBoost"] = lambda: (
        XGBClassifier(eval_metric="logloss", random_state=42),
        {"n_estimators": [100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1, 0.2]},
    )

REGRESSION_MODEL_BUILDERS = {
    "Linear Regression": lambda: (
        LinearRegression(),
        {"fit_intercept": [True, False]},
    ),
    "Random Forest": lambda: (
        RandomForestRegressor(random_state=42),
        {"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10, 20], "min_samples_leaf": [1, 2, 4]},
    ),
    "SVM": lambda: (
        SVR(),
        {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
    ),
    "AdaBoost": lambda: (
        AdaBoostRegressor(random_state=42),
        {"n_estimators": [50, 100, 200], "learning_rate": [0.25, 0.5, 1.0]},
    ),
}
if XGB_AVAILABLE:
    REGRESSION_MODEL_BUILDERS["XGBoost"] = lambda: (
        XGBRegressor(random_state=42),
        {"n_estimators": [100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1, 0.2]},
    )

# ---------------------------------------------------------------------------
# App Title & Industry Value Proposition Banner
# ---------------------------------------------------------------------------
st.title("🏦 Sovereign Recession Risk & Early Warning System (SR-EWS)")
st.caption(
    "**Enterprise Solution for African Markets:** Quantitative macro-risk forecasting, credit loss decision-support, "
    "and interactive policy stress-testing for financial institutions, multilateral lenders, and investors."
)

# ---------------------------------------------------------------------------
# Data Auto-Cleaning Function
# ---------------------------------------------------------------------------
def auto_clean(df, target_col):
    log = []
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c != target_col]

    for c in numeric_cols:
        if c == target_col:
            continue
        n_missing = df[c].isna().sum()
        if n_missing > 0:
            df[c] = df[c].fillna(df[c].median())
            log.append(f"Imputed {n_missing} missing values in '{c}' using median distribution.")
            
    for c in categorical_cols:
        n_missing = df[c].isna().sum()
        if n_missing > 0:
            mode_val = df[c].mode().iloc[0] if not df[c].mode().empty else "Unknown"
            df[c] = df[c].fillna(mode_val)
            log.append(f"Imputed {n_missing} missing values in '{c}' with mode ('{mode_val}').")

    for c in numeric_cols:
        if c == target_col:
            continue
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[c] < lower) | (df[c] > upper)).sum()
        if n_out > 0:
            df[c] = df[c].clip(lower, upper)
            log.append(f"Capped {n_out} extreme outliers in '{c}' within 1.5x IQR bounds.")

    encoded_frames = [df[numeric_cols]]
    dropped = []
    for c in categorical_cols:
        n_unique = df[c].nunique()
        if n_unique <= 15:
            dummies = pd.get_dummies(df[c], prefix=c, drop_first=True)
            encoded_frames.append(dummies)
            log.append(f"One-hot encoded categorical indicator '{c}' ({n_unique} categories).")
        else:
            dropped.append(c)
    if dropped:
        log.append(f"Excluded high-cardinality non-predictive columns (>15 categories): {', '.join(dropped)}.")

    df_target = df[target_col] if target_col in df.columns else None
    df_clean = pd.concat(encoded_frames, axis=1)
    if target_col not in df_clean.columns and df_target is not None:
        df_clean[target_col] = df_target.values

    return df_clean, log


def n_iter_for_search(n_train_rows, grid_size):
    if n_train_rows < 300:
        return None
    return min(grid_size, max(6, n_train_rows // 100))


def get_cv(y, is_classification=True):
    if is_classification:
        n_minority = y.value_counts().min()
        folds = max(2, min(5, n_minority))
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    folds = max(2, min(5, len(y) // 20))
    return KFold(n_splits=folds, shuffle=True, random_state=42)


def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def get_feature_importance(model, feature_cols):
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        vals = np.abs(coef.flatten())
    else:
        return None
    if len(vals) != len(feature_cols):
        return None
    return pd.Series(vals, index=feature_cols).sort_values(ascending=False)


def render_eda(df, target_col, target_type="Binary classification"):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    c1, c2 = st.columns(2)
    with c1:
        if target_type == "Binary classification":
            st.write("**Historical Macroeconomic Event Balance**")
            if target_col in df.columns:
                counts = df[target_col].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.bar(["Baseline / Growth (0)", "Recession Event (1)"], counts.values, color=["#27ae60", "#c0392b"])
                ax.set_ylabel("Historical Observations")
                st.pyplot(fig)
                imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else float("inf")
                st.caption(f"Historical Event Imbalance Ratio: **{imbalance_ratio:.1f}:1**")
        else:
            st.write("**Target Growth Distribution**")
            if target_col in df.columns:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.hist(df[target_col].dropna(), bins=30, color="#c0392b")
                ax.set_xlabel(target_col)
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

    with c2:
        st.write("**Data Quality & Missing Values**")
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            st.info("Dataset quality verified: No missing values detected.")
        else:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.barh([get_label(c) for c in missing.index], missing.values, color="#e67e22")
            ax.set_xlabel("Missing Data Count")
            ax.invert_yaxis()
            st.pyplot(fig)

    if len(numeric_cols) >= 2:
        st.write("**Macroeconomic Inter-Variable Correlation Heatmap**")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(min(0.5 * len(numeric_cols) + 2, 10), min(0.5 * len(numeric_cols) + 2, 10)))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_xticklabels([get_label(c) for c in numeric_cols], rotation=90, fontsize=6)
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_yticklabels([get_label(c) for c in numeric_cols], fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)


# ---------------------------------------------------------------------------
# Sidebar: System Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1️⃣ Economic Dataset Input")
    train_file = st.file_uploader("Upload Macroeconomic Dataset (CSV)", type=["csv"], key="train_upload")

    if train_file is not None:
        raw_df = pd.read_csv(train_file)
        st.success(f"Dataset Ingested: {raw_df.shape[0]} rows, {raw_df.shape[1]} variables")

        default_target = "growthbucket" if "growthbucket" in raw_df.columns else raw_df.columns[-1]
        target_col = st.selectbox("Target Outcome Variable", options=list(raw_df.columns),
                                   index=list(raw_df.columns).index(default_target),
                                   format_func=get_label)

        guessed_binary = raw_df[target_col].nunique(dropna=True) <= 2
        target_type = st.radio(
            "Predictive Task Framing",
            options=["Binary classification", "Continuous (regression)"],
            index=0 if guessed_binary else 1,
            help="Select 'Binary' for early warning indicators (1 = Recession, 0 = Growth). "
                 "Select 'Continuous' for numerical macro forecasts (e.g., GDP Growth Rate).",
        )
        is_classification = target_type == "Binary classification"

        st.session_state.raw_df = raw_df
        st.session_state.target_col = target_col
        st.session_state.target_type = target_type

        st.subheader("2️⃣ Model Architecture Suite")
        available_models = CLASSIFICATION_MODEL_BUILDERS if is_classification else REGRESSION_MODEL_BUILDERS
        model_choices = st.multiselect(
            "Select Algorithms for Comparison", options=list(available_models.keys()),
            default=list(available_models.keys()),
        )

        st.subheader("3️⃣ Validation & Sampling Strategy")
        test_size = st.slider("Holdout Validation Set (%)", 10, 40, 20, 5) / 100

        if is_classification:
            use_smote = st.checkbox("Apply SMOTE (Synthetic Minority Oversampling)", value=True, disabled=not SMOTE_AVAILABLE)
            if not SMOTE_AVAILABLE:
                st.warning("imbalanced-learn not installed — SMOTE unavailable.")
        else:
            use_smote = False

        train_clicked = st.button("🚀 Train & Calibrate System", type="primary", use_container_width=True)

        if train_clicked:
            if not model_choices:
                st.error("Please select at least one algorithm to proceed.")
            else:
                with st.spinner("Executing Data Preprocessing Pipeline..."):
                    df_clean, cleaning_log = auto_clean(raw_df, target_col)
                    st.session_state.cleaning_log = cleaning_log

                feature_cols = [c for c in df_clean.columns if c != target_col]
                df_clean = df_clean.dropna(subset=[target_col])
                X = df_clean[feature_cols]
                y = df_clean[target_col].astype(int) if is_classification else df_clean[target_col].astype(float)

                split_kwargs = {"test_size": test_size, "random_state": 42}
                if is_classification:
                    split_kwargs["stratify"] = y
                X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                smote_before = y_train.value_counts().sort_index() if is_classification else None
                smote_applied = False
                if is_classification and use_smote and SMOTE_AVAILABLE:
                    sm = SMOTE(random_state=42)
                    X_train_s, y_train = sm.fit_resample(X_train_s, y_train)
                    smote_applied = True
                smote_after = pd.Series(y_train).value_counts().sort_index() if is_classification else None

                st.session_state.smote_before = smote_before
                st.session_state.smote_after = smote_after
                st.session_state.smote_applied = smote_applied

                cv = get_cv(y_train, is_classification=is_classification)
                model_builders = CLASSIFICATION_MODEL_BUILDERS if is_classification else REGRESSION_MODEL_BUILDERS
                search_scoring = "recall" if is_classification else "neg_root_mean_squared_error"
                results = {}
                rows = []
                progress = st.progress(0.0, text="Calibrating model suite...")

                for i, name in enumerate(model_choices):
                    base_model, param_grid = model_builders[name]()
                    grid_size = int(np.prod([len(v) for v in param_grid.values()]))
                    budget = n_iter_for_search(X_train_s.shape[0], grid_size)

                    if budget is None:
                        search = GridSearchCV(base_model, param_grid, scoring=search_scoring, cv=cv, n_jobs=-1)
                        search_type = "Exhaustive Grid Search"
                    else:
                        search = RandomizedSearchCV(base_model, param_grid, n_iter=budget,
                                                     scoring=search_scoring, cv=cv, n_jobs=-1, random_state=42)
                        search_type = f"Budgeted Random Search ({budget} combinations)"

                    search.fit(X_train_s, y_train)
                    best_est = search.best_estimator_
                    preds = best_est.predict(X_test_s)

                    if is_classification:
                        probs = best_est.predict_proba(X_test_s)[:, 1]
                        metrics = {
                            "Model": name,
                            "Accuracy": accuracy_score(y_test, preds),
                            "Precision": precision_score(y_test, preds, zero_division=0),
                            "Recall (Early Warning Capture)": recall_score(y_test, preds),
                            "F1-score": f1_score(y_test, preds),
                            "AUC-PR": average_precision_score(y_test, probs),
                        }
                        results[name] = {
                            "model": best_est, "best_params": search.best_params_,
                            "search_type": search_type, "metrics": metrics,
                            "confusion_matrix": confusion_matrix(y_test, preds),
                            "y_test": y_test, "probs": probs, "preds": preds,
                        }
                    else:
                        mse = mean_squared_error(y_test, preds)
                        corr = pearsonr(y_test, preds)[0] if len(y_test) > 1 else np.nan
                        metrics = {
                            "Model": name,
                            "MSE": mse,
                            "RMSE": mse ** 0.5,
                            "MAE": mean_absolute_error(y_test, preds),
                            "MAPE (%)": safe_mape(y_test, preds),
                            "R2": r2_score(y_test, preds),
                            "Correlation (r)": corr,
                        }
                        results[name] = {
                            "model": best_est, "best_params": search.best_params_,
                            "search_type": search_type, "metrics": metrics,
                            "y_test": y_test, "preds": preds,
                        }
                    rows.append(metrics)
                    progress.progress((i + 1) / len(model_choices), text=f"Optimized {name}")

                if is_classification:
                    metrics_df = pd.DataFrame(rows).sort_values("AUC-PR", ascending=False).reset_index(drop=True)
                else:
                    metrics_df = pd.DataFrame(rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
                best_model_name = metrics_df.iloc[0]["Model"]

                st.session_state.results = results
                st.session_state.metrics_df = metrics_df
                st.session_state.best_model_name = best_model_name
                st.session_state.scaler = scaler
                st.session_state.feature_cols = feature_cols
                st.session_state.feature_stats = {
                    c: {"min": float(X[c].min()), "max": float(X[c].max()), "mean": float(X[c].mean())}
                    for c in feature_cols
                }
                st.session_state.trained = True
                rank_metric = "AUC-PR" if is_classification else "RMSE"
                st.success(f"System Ready! Highest Performing Model: **{best_model_name}**")

# ---------------------------------------------------------------------------
# Main Application Content
# ---------------------------------------------------------------------------
if st.session_state.get("raw_df") is not None:
    st.header("📊 Exploratory Macroeconomic Diagnostics")
    with st.expander("Expand Macro Diagnostics Panel", expanded=not st.session_state.trained):
        render_eda(st.session_state.raw_df, st.session_state.target_col, st.session_state.target_type)

if not st.session_state.trained:
    st.info("👈 Upload your country macroeconomic dataset in the sidebar and click **Train & Calibrate System** to initialize predictions.")
    st.stop()

is_classification = st.session_state.target_type == "Binary classification"

# ---------------------------------------------------------------------------
# Model Evaluation & Industry Risk Framing
# ---------------------------------------------------------------------------
st.header("📈 Model Validation & Early Warning Calibration")
metrics_df = st.session_state.metrics_df
best_name = st.session_state.best_model_name

col1, col2 = st.columns([2, 1])
with col1:
    def highlight_best(row):
        return ["background-color: #d4edda" if row["Model"] == best_name else "" for _ in row]
    st.dataframe(metrics_df.style.apply(highlight_best, axis=1), use_container_width=True, hide_index=True)
    st.success(f"🏆 Selected Model Architecture: **{best_name}**")
with col2:
    fig, ax = plt.subplots(figsize=(4, 3))
    rank_metric = "AUC-PR" if is_classification else "RMSE"
    ax.barh(metrics_df["Model"], metrics_df[rank_metric], color="#2980b9")
    ax.set_xlabel(rank_metric)
    ax.invert_yaxis()
    st.pyplot(fig)

best_result = st.session_state.results[best_name]

# ---------------------------------------------------------------------------
# Drivers of Vulnerability (Feature Importance with Friendly Labels)
# ---------------------------------------------------------------------------
st.subheader(f"🔍 Primary Macroeconomic Vulnerability Drivers ({best_name})")
importance = get_feature_importance(best_result["model"], st.session_state.feature_cols)
if importance is not None:
    top_importance = importance.head(10).sort_values(ascending=True)
    fig3, ax3 = plt.subplots(figsize=(7, 3.5))
    top_labels = [get_label(col) for col in top_importance.index]
    ax3.barh(top_labels, top_importance.values, color="#c0392b")
    ax3.set_xlabel("Relative Macro Vulnerability Weight")
    st.pyplot(fig3)
else:
    st.info("Feature importance score non-linear for selected model kernel.")

# ---------------------------------------------------------------------------
# Decision Support System & Prediction Hub
# ---------------------------------------------------------------------------
st.header("🎯 Institutional Risk Assessment & Decision Hub")
model = st.session_state.results[best_name]["model"]
scaler = st.session_state.scaler
feature_cols = st.session_state.feature_cols
stats = st.session_state.feature_stats

tab1, tab2, tab3 = st.tabs(["⚡ Single-Country Assessment", "🔮 Macro Stress-Testing (Scenario Analysis)", "📄 Portfolio Ingestion (Batch)"])

# ---------------------------------------------------------------------------
# Tab 1: Single Country Assessment
# ---------------------------------------------------------------------------
with tab1:
    st.write("Input current country macroeconomic indicators to evaluate recession risk and access financial action triggers:")
    input_vals = {}
    n_cols = 2
    cols = st.columns(n_cols)
    for i, feat in enumerate(feature_cols):
        s = stats[feat]
        with cols[i % n_cols]:
            input_vals[feat] = st.number_input(
                label=get_label(feat),
                value=round(s["mean"], 4),
                format="%.4f",
                help=f"Raw dataset variable: {feat}"
            )

    if st.button("Generate Sovereign Risk Report", type="primary"):
        X_new = pd.DataFrame([input_vals])[feature_cols]
        X_new_s = scaler.transform(X_new)
        
        if is_classification:
            prob = model.predict_proba(X_new_s)[0, 1]
            
            st.markdown("---")
            st.subheader("📋 Executive Decision Support Matrix")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Recession Probability", f"{prob:.1%}")
            
            if prob >= 0.60:
                m2.metric("Sovereign Risk Category", "HIGH RISK (Tier 3)", delta="Critical Warning", delta_color="inverse")
                m3.metric("IFRS 9 Provisioning Stage", "Stage 2 / Stage 3", delta="Increase Reserves", delta_color="inverse")
                st.error("⚠️ **CRITICAL RECESSION RISK DETECTED**")
                
                st.markdown("### 🏛️ Recommended Institutional Mitigations:")
                st.markdown("""
                * **Banking & Lending:** Increase IFRS 9 Stage 2 loan-loss provisions by 15-25%. Tighten credit standards on corporate and retail loan originations in this market.
                * **Institutional Investors:** Hedge local foreign-exchange (FX) exposure. Reduce portfolio exposure in sovereign bonds and shift toward defensive assets.
                * **Multinational Operations:** Reduce working capital balances, scale back short-term inventory imports, and secure FX liquidity buffers.
                """)
            elif prob >= 0.35:
                m2.metric("Sovereign Risk Category", "MODERATE RISK (Tier 2)", delta="Watchlist", delta_color="off")
                m3.metric("IFRS 9 Provisioning Stage", "Stage 2 Watchlist", delta="Monitor Exposure", delta_color="off")
                st.warning("⚡ **MODERATE RECESSION VULNERABILITY**")
                
                st.markdown("### 🏛️ Recommended Institutional Mitigations:")
                st.markdown("""
                * **Banking & Lending:** Place corporate obligors on close monitoring. Audit trade finance credit lines.
                * **Institutional Investors:** Avoid expanding unhedged debt investments; mandate shorter bond durations.
                * **Multinational Operations:** Maintain current inventory but prepare contingency sourcing plans for potential demand contraction.
                """)
            else:
                m2.metric("Sovereign Risk Category", "LOW RISK (Tier 1)", delta="Stable Baseline", delta_color="normal")
                m3.metric("IFRS 9 Provisioning Stage", "Stage 1", delta="Standard Provisioning", delta_color="normal")
                st.success("✅ **STABLE MACROECONOMIC BASELINE**")
                
                st.markdown("### 🏛️ Strategic Outlook:")
                st.markdown("""
                * **Market Expansion:** Macroeconomic fundamentals remain supportive for capital deployment and strategic credit growth.
                """)
        else:
            val = model.predict(X_new_s)[0]
            st.success(f"Forecasted Target ({get_label(st.session_state.target_col)}): **{val:,.4f}**")

# ---------------------------------------------------------------------------
# Tab 2: Macro Stress Testing
# ---------------------------------------------------------------------------
with tab2:
    st.write("Simulate adverse macroeconomic shocks (e.g., inflation spikes, terms of trade decline) to evaluate sovereign risk resilience:")
    
    st.subheader("⚙️ Stress Test Shock Simulation Controls")
    stress_vals = {}
    
    s_cols = st.columns(2)
    for i, feat in enumerate(feature_cols):
        s = stats[feat]
        with s_cols[i % 2]:
            stress_vals[feat] = st.slider(
                label=f"Simulate: {get_label(feat)}",
                min_value=float(s["min"]),
                max_value=float(s["max"]),
                value=float(s["mean"]),
                help=f"Raw variable: {feat}"
            )
            
    X_stress = pd.DataFrame([stress_vals])[feature_cols]
    X_stress_s = scaler.transform(X_stress)
    
    if is_classification:
        stress_prob = model.predict_proba(X_stress_s)[0, 1]
        st.markdown("---")
        st.subheader("🧪 Stress Test Simulation Output")
        
        c_a, c_b = st.columns(2)
        c_a.metric("Simulated Recession Probability", f"{stress_prob:.1%}")
        
        if stress_prob >= 0.5:
            c_b.error("STRESS TEST RESULT: UNSTABLE — Macroeconomic shocks trigger critical recession threshold.")
        else:
            c_b.success("STRESS TEST RESULT: RESILIENT — Sovereign baseline absorbs simulated shock parameters.")

# ---------------------------------------------------------------------------
# Tab 3: Batch Ingestion
# ---------------------------------------------------------------------------
with tab3:
    st.write("Upload multi-country macroeconomic CSV datasets for automated risk portfolio batch classification:")
    batch_file = st.file_uploader("Upload Multi-Country CSV", type=["csv"], key="batch_upload")
    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        missing = [c for c in feature_cols if c not in batch_df.columns]
        if missing:
            st.error(f"Missing required indicator columns in file: {', '.join(missing)}")
        else:
            X_batch = batch_df[feature_cols]
            X_batch_s = scaler.transform(X_batch)
            out_df = batch_df.copy()
            
            if is_classification:
                probs = model.predict_proba(X_batch_s)[:, 1]
                out_df["Recession_Probability"] = np.round(probs, 4)
                out_df["Risk_Rating"] = np.where(probs >= 0.6, "HIGH RISK", np.where(probs >= 0.35, "MODERATE RISK", "LOW RISK"))
                out_df["IFRS9_Action"] = np.where(probs >= 0.6, "Stage 2/3 - Increase Reserves", "Stage 1 - Standard")
            else:
                preds = model.predict(X_batch_s)
                out_df[f"Predicted_{st.session_state.target_col}"] = np.round(preds, 4)
                
            st.dataframe(out_df, use_container_width=True)
            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Risk Assessment Brief (CSV)", data=csv_bytes, file_name="sovereign_risk_assessment.csv", mime="text/csv")
