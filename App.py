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

st.set_page_config(page_title="African Recession Predictor", page_icon="📉", layout="wide")

# ---------------------------------------------------------------------------
# Session state
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

st.title("📉 African Country Recession Predictor")
st.caption(
    "End-to-end, no-code pipeline: upload data → auto-clean → pick models → "
    "train & tune → compare → get an automated business recommendation."
)

# ---------------------------------------------------------------------------
# Helper: auto data cleaning
# ---------------------------------------------------------------------------
def auto_clean(df, target_col):
    log = []
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c != target_col]

    # Missing values
    for c in numeric_cols:
        if c == target_col:
            continue
        n_missing = df[c].isna().sum()
        if n_missing > 0:
            df[c] = df[c].fillna(df[c].median())
            log.append(f"Filled {n_missing} missing values in '{c}' with the median.")
    for c in categorical_cols:
        n_missing = df[c].isna().sum()
        if n_missing > 0:
            mode_val = df[c].mode().iloc[0] if not df[c].mode().empty else "Unknown"
            df[c] = df[c].fillna(mode_val)
            log.append(f"Filled {n_missing} missing values in '{c}' with the most common value ('{mode_val}').")

    # Outlier capping (IQR method) on numeric features
    for c in numeric_cols:
        if c == target_col:
            continue
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[c] < lower) | (df[c] > upper)).sum()
        if n_out > 0:
            df[c] = df[c].clip(lower, upper)
            log.append(f"Capped {n_out} outlier values in '{c}' to the IQR bounds.")

    # Categorical encoding (one-hot for low cardinality, drop very high cardinality)
    encoded_frames = [df[numeric_cols]]
    dropped = []
    for c in categorical_cols:
        n_unique = df[c].nunique()
        if n_unique <= 15:
            dummies = pd.get_dummies(df[c], prefix=c, drop_first=True)
            encoded_frames.append(dummies)
            log.append(f"One-hot encoded '{c}' ({n_unique} categories).")
        else:
            dropped.append(c)
    if dropped:
        log.append(f"Dropped high-cardinality columns (>15 unique values): {', '.join(dropped)}.")

    df_target = df[target_col] if target_col in df.columns else None
    df_clean = pd.concat(encoded_frames, axis=1)
    if target_col not in df_clean.columns and df_target is not None:
        df_clean[target_col] = df_target.values

    return df_clean, log


def n_iter_for_search(n_train_rows, grid_size):
    """Decide search intensity automatically based on dataset size."""
    if n_train_rows < 300:
        return None  # small data -> exhaustive GridSearchCV
    return min(grid_size, max(6, n_train_rows // 100))  # RandomizedSearchCV budget


def get_cv(y, is_classification=True):
    if is_classification:
        n_minority = y.value_counts().min()
        folds = max(2, min(5, n_minority))
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    folds = max(2, min(5, len(y) // 20))
    return KFold(n_splits=folds, shuffle=True, random_state=42)


def safe_mape(y_true, y_pred):
    """Mean absolute percentage error, ignoring rows where the actual value is 0
    (otherwise those rows would divide by zero and blow up the metric)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def get_feature_importance(model, feature_cols):
    """Best-effort feature importance/coefficients. Returns None if the model
    type doesn't expose either (e.g. SVM/SVR with a non-linear kernel)."""
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


def _describe_level(value, thresholds, labels):
    """Pick a qualitative label ('strong'/'moderate'/'weak' etc.) for a metric
    value given ascending thresholds, e.g. thresholds=[0.3, 0.6], labels=['weak','moderate','strong']."""
    for t, label in zip(thresholds, labels):
        if value < t:
            return label
    return labels[-1]


def build_classification_insight(metrics_df, best_name, cm, target_col, cleaning_log):
    """Plain-language summary generated locally from the metrics -- no API call."""
    row = metrics_df[metrics_df["Model"] == best_name].iloc[0]
    acc, prec, rec, f1, aucpr = row["Accuracy"], row["Precision"], row["Recall"], row["F1-score"], row["AUC-PR"]
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    total = tn + fp + fn + tp

    perf_word = _describe_level(aucpr, [0.3, 0.6], ["limited", "moderate", "strong"])
    lines = []
    lines.append(
        f"**Summary:** The best-performing model, **{best_name}**, correctly classified "
        f"{acc:.0%} of the {total} test cases overall, with {perf_word} performance "
        f"(AUC-PR of {aucpr:.2f}, the fairest measure here given class imbalance)."
    )
    lines.append(
        f"**What the errors mean in practice:** Of the actual '{target_col}' positive cases in "
        f"the test set, the model correctly flagged {rec:.0%} (recall) and missed {fn} of them "
        f"(false negatives). When it predicted a positive case, it was right {prec:.0%} of the "
        f"time (precision), with {fp} false alarms (false positives)."
    )

    recs = []
    if rec < 0.6:
        recs.append("Recall is on the lower side, so treat this model as an early-warning signal "
                     "to investigate further rather than a final determination — missed positive "
                     "cases (false negatives) are currently the bigger risk.")
    else:
        recs.append("Recall is reasonably strong, so the model is catching most true positive "
                     "cases in this test set — worth combining with domain judgment before acting "
                     "on individual predictions.")
    if prec < 0.6:
        recs.append("Precision is moderate, meaning a fair number of flagged cases turn out to be "
                     "false alarms — pair predictions with a secondary check before taking costly action.")
    recs.append("Re-run this pipeline periodically as new data comes in, since relationships in "
                 "economic data can shift over time.")
    lines.append("**Recommendations:**\n" + "\n".join(f"- {r}" for r in recs[:3]))

    lines.append(
        f"**Limitation:** This model reflects correlations in the uploaded historical data, "
        f"not proven causes — and its accuracy depends on how representative that data is. "
        f"Cleaning steps applied: {cleaning_summary_text(cleaning_log)}."
    )
    return "\n\n".join(lines)


def build_regression_insight(metrics_df, best_name, target_col, cleaning_log):
    """Plain-language summary generated locally from the metrics -- no API call."""
    row = metrics_df[metrics_df["Model"] == best_name].iloc[0]
    rmse, mae, r2, corr = row["RMSE"], row["MAE"], row["R2"], row["Correlation (r)"]
    mape = row["MAPE (%)"]

    fit_word = _describe_level(max(r2, 0), [0.3, 0.6], ["weak", "moderate", "strong"])
    lines = []
    lines.append(
        f"**Summary:** The best-performing model, **{best_name}**, shows a {fit_word} fit for "
        f"predicting **{target_col}** — it explains about {max(r2, 0):.0%} of the variation in "
        f"the test data (R²={r2:.3f}), with predictions correlating at r={corr:.2f} with actual values."
    )
    mape_note = (
        f" (MAPE of {mape:.1f}% — treat this cautiously if {target_col} has values near zero, "
        f"since that inflates the percentage error)" if pd.notna(mape) else ""
    )
    lines.append(
        f"**Typical error size:** On average, predictions are off by about {mae:.3f} units "
        f"(MAE) with a root-mean-square error of {rmse:.3f}{mape_note}. Use that margin as a "
        f"rough confidence band around any single prediction."
    )

    recs = []
    if r2 < 0.4:
        recs.append("Explanatory power is limited — consider whether additional features, more "
                     "historical data, or a different target definition might capture the pattern better.")
    else:
        recs.append("The model captures a meaningful share of the pattern — useful for directional "
                     "guidance and trend monitoring rather than precise point forecasts.")
    recs.append(f"Report predictions with the ±{rmse:.2f} error margin attached, so decisions "
                "account for the model's typical uncertainty.")
    recs.append("Re-check performance periodically as new data arrives, since economic "
                 "relationships can drift over time.")
    lines.append("**Recommendations:**\n" + "\n".join(f"- {r}" for r in recs[:3]))

    lines.append(
        f"**Limitation:** This model reflects correlations in the uploaded historical data, not "
        f"proven causes, and can be sensitive to outliers. Cleaning steps applied: "
        f"{cleaning_summary_text(cleaning_log)}."
    )
    return "\n\n".join(lines)


def cleaning_summary_text(cleaning_log):
    return "; ".join(cleaning_log) if cleaning_log else "none needed, data was already tidy"


def render_eda(df, target_col, target_type="Binary classification"):
    """Render exploratory graphs for the raw uploaded dataset."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    c1, c2 = st.columns(2)
    with c1:
        if target_type == "Binary classification":
            st.write("**Target class balance**")
            if target_col in df.columns:
                counts = df[target_col].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.bar(counts.index.astype(str), counts.values, color="#c0392b")
                ax.set_xlabel(target_col)
                ax.set_ylabel("Count")
                st.pyplot(fig)
                imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else float("inf")
                st.caption(f"Imbalance ratio (majority:minority) ≈ {imbalance_ratio:.1f}:1")
        else:
            st.write("**Target distribution**")
            if target_col in df.columns:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.hist(df[target_col].dropna(), bins=30, color="#c0392b")
                ax.set_xlabel(target_col)
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
                st.caption(
                    f"Mean: {df[target_col].mean():.3f} | "
                    f"Std dev: {df[target_col].std():.3f} | "
                    f"Range: [{df[target_col].min():.3f}, {df[target_col].max():.3f}]"
                )
    with c2:
        st.write("**Missing values by column**")
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            st.info("No missing values detected.")
        else:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.barh(missing.index.astype(str), missing.values, color="#e67e22")
            ax.set_xlabel("Missing count")
            ax.invert_yaxis()
            st.pyplot(fig)

    if len(numeric_cols) >= 2:
        st.write("**Correlation heatmap (numeric features)**")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(min(0.5 * len(numeric_cols) + 2, 10),
                                         min(0.5 * len(numeric_cols) + 2, 10)))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=90, fontsize=7)
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_yticklabels(numeric_cols, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

    if numeric_cols:
        st.write("**Feature distribution**")
        feat_pick = st.selectbox("Choose a numeric column to inspect", options=numeric_cols, key="eda_feat_pick")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(df[feat_pick].dropna(), bins=30, color="#2980b9")
        ax.set_xlabel(feat_pick)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)


# ---------------------------------------------------------------------------
# Sidebar: Step 1 — Upload, clean, configure
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1️⃣ Data & setup")
    train_file = st.file_uploader("Upload training dataset (CSV)", type=["csv"], key="train_upload")

    if train_file is not None:
        raw_df = pd.read_csv(train_file)
        st.success(f"Loaded {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

        default_target = "growthbucket" if "growthbucket" in raw_df.columns else raw_df.columns[-1]
        target_col = st.selectbox("Target column", options=list(raw_df.columns),
                                   index=list(raw_df.columns).index(default_target))

        # Guess a sensible default (2 unique values -> binary), but let the user override --
        # they know their data best, e.g. a 0/1-coded column that's actually a count.
        guessed_binary = raw_df[target_col].nunique(dropna=True) <= 2
        target_type = st.radio(
            "Is this target variable binary?",
            options=["Binary classification", "Continuous (regression)"],
            index=0 if guessed_binary else 1,
            help="Choose 'Binary classification' for a yes/no or 0/1 label (like recession "
                 "vs no recession). Choose 'Continuous (regression)' for a numeric value "
                 "like GDP growth rate, inflation, or any other non-binary number.",
        )
        is_classification = target_type == "Binary classification"

        st.session_state.raw_df = raw_df
        st.session_state.target_col = target_col
        st.session_state.target_type = target_type

        st.subheader("2️⃣ Choose models to train")
        available_models = CLASSIFICATION_MODEL_BUILDERS if is_classification else REGRESSION_MODEL_BUILDERS
        model_choices = st.multiselect(
            "Model types", options=list(available_models.keys()),
            default=list(available_models.keys()),
        )

        st.subheader("3️⃣ Train/test split")
        test_size = st.slider("Test set size (%)", 10, 40, 20, 5) / 100

        if is_classification:
            use_smote = st.checkbox("Apply SMOTE to balance classes", value=True, disabled=not SMOTE_AVAILABLE)
            if not SMOTE_AVAILABLE:
                st.warning("imbalanced-learn not installed — SMOTE unavailable.")
        else:
            use_smote = False
            st.caption("SMOTE is for class imbalance and doesn't apply to a continuous target.")

        train_clicked = st.button("🚀 Clean, train & tune", type="primary", use_container_width=True)

        if train_clicked:
            if not model_choices:
                st.error("Select at least one model type.")
            else:
                with st.spinner("Cleaning data..."):
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

                # --- SMOTE (classification only) -- track before/after counts for the graph ---
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
                progress = st.progress(0.0, text="Training models...")

                for i, name in enumerate(model_choices):
                    base_model, param_grid = model_builders[name]()
                    grid_size = int(np.prod([len(v) for v in param_grid.values()]))
                    budget = n_iter_for_search(X_train_s.shape[0], grid_size)

                    if budget is None:
                        search = GridSearchCV(base_model, param_grid, scoring=search_scoring, cv=cv, n_jobs=-1)
                        search_type = "GridSearchCV (exhaustive — small dataset)"
                    else:
                        search = RandomizedSearchCV(base_model, param_grid, n_iter=budget,
                                                     scoring=search_scoring, cv=cv, n_jobs=-1, random_state=42)
                        search_type = f"RandomizedSearchCV ({budget} combos — larger dataset)"

                    search.fit(X_train_s, y_train)
                    best_est = search.best_estimator_
                    preds = best_est.predict(X_test_s)

                    if is_classification:
                        probs = best_est.predict_proba(X_test_s)[:, 1]
                        metrics = {
                            "Model": name,
                            "Accuracy": accuracy_score(y_test, preds),
                            "Precision": precision_score(y_test, preds, zero_division=0),
                            "Recall": recall_score(y_test, preds),
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
                    progress.progress((i + 1) / len(model_choices), text=f"Trained {name}")

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
                st.session_state.insight_text = None
                rank_metric = "AUC-PR" if is_classification else "RMSE"
                st.success(f"Done! Best model by {rank_metric}: **{best_model_name}**")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
if st.session_state.get("raw_df") is not None:
    st.header("0️⃣ Explore your data (EDA)")
    with st.expander("Show exploratory graphs", expanded=not st.session_state.trained):
        render_eda(st.session_state.raw_df, st.session_state.target_col, st.session_state.target_type)

if not st.session_state.trained:
    st.info("👈 Upload your dataset, choose models and split size, then click **Clean, train & tune**.")
    st.stop()

is_classification = st.session_state.target_type == "Binary classification"

with st.expander("🧹 Data cleaning log", expanded=False):
    for line in st.session_state.cleaning_log:
        st.write(f"- {line}")
    if not st.session_state.cleaning_log:
        st.write("No cleaning was necessary — data was already tidy.")

# ---------------------------------------------------------------------------
# SMOTE before/after graph (classification only)
# ---------------------------------------------------------------------------
if is_classification and st.session_state.smote_before is not None:
    st.header("⚖️ Class balance before vs. after SMOTE")
    if st.session_state.smote_applied:
        before, after = st.session_state.smote_before, st.session_state.smote_after
        all_classes = sorted(set(before.index) | set(after.index))
        fig, ax = plt.subplots(figsize=(5, 3))
        x = np.arange(len(all_classes))
        width = 0.35
        before_vals = [before.get(c, 0) for c in all_classes]
        after_vals = [after.get(c, 0) for c in all_classes]
        ax.bar(x - width / 2, before_vals, width, label="Before SMOTE", color="#e67e22")
        ax.bar(x + width / 2, after_vals, width, label="After SMOTE", color="#2980b9")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in all_classes])
        ax.set_xlabel(st.session_state.target_col)
        ax.set_ylabel("Count (training set)")
        ax.legend()
        st.pyplot(fig)
        st.caption("SMOTE generates synthetic minority-class samples in the training set only "
                   "(the test set is untouched) so the model sees a more balanced class distribution.")
    else:
        st.info("SMOTE was not applied for this run — the training set retains its original class balance.")

st.header("1️⃣ Model performance & tuning")
metrics_df = st.session_state.metrics_df
best_name = st.session_state.best_model_name

if is_classification:
    fmt = {"Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
           "F1-score": "{:.3f}", "AUC-PR": "{:.3f}"}
    rank_metric, rank_label = "AUC-PR", "AUC-PR, the fairest metric for imbalanced classes"
else:
    fmt = {"MSE": "{:.4f}", "RMSE": "{:.4f}", "MAE": "{:.4f}",
           "MAPE (%)": "{:.2f}", "R2": "{:.4f}", "Correlation (r)": "{:.4f}"}
    rank_metric, rank_label = "RMSE", "RMSE (lower is better)"

col1, col2 = st.columns([2, 1])
with col1:
    def highlight_best(row):
        return ["background-color: #d4edda" if row["Model"] == best_name else "" for _ in row]
    st.dataframe(
        metrics_df.style.apply(highlight_best, axis=1).format(fmt),
        use_container_width=True, hide_index=True,
    )
    st.success(f"🏆 Best model: **{best_name}** (ranked by {rank_label})")
    if not is_classification:
        st.caption("Note: MAPE can look inflated or unstable when the target has values at or "
                   "near zero, since it divides by the actual value. RMSE/MAE/R² are more "
                   "reliable in that case.")
with col2:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.barh(metrics_df["Model"], metrics_df[rank_metric], color="#c0392b")
    ax.set_xlabel(rank_metric)
    ax.invert_yaxis()
    st.pyplot(fig)

with st.expander("🔧 Tuned hyperparameters per model"):
    for name, r in st.session_state.results.items():
        st.write(f"**{name}** — {r['search_type']}")
        st.json(r["best_params"])

best_result = st.session_state.results[best_name]

if is_classification:
    cm = best_result["confusion_matrix"]
    st.subheader(f"Confusion matrix — {best_name}")
    fig2, ax2 = plt.subplots(figsize=(3, 3))
    ax2.imshow(cm, cmap="Reds")
    for (i, j), val in np.ndenumerate(cm):
        ax2.text(j, i, str(val), ha="center", va="center")
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["No recession", "Recession"])
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["No recession", "Recession"])
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
    st.pyplot(fig2)
else:
    st.subheader(f"Predicted vs. actual — {best_name}")
    y_test_best = best_result["y_test"]
    preds_best = best_result["preds"]
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.scatter(y_test_best, preds_best, alpha=0.6, color="#2980b9", edgecolor="none")
    lims = [min(min(y_test_best), min(preds_best)), max(max(y_test_best), max(preds_best))]
    ax2.plot(lims, lims, "--", color="gray", linewidth=1)
    ax2.set_xlabel(f"Actual {st.session_state.target_col}")
    ax2.set_ylabel(f"Predicted {st.session_state.target_col}")
    st.pyplot(fig2)
    st.caption("Points on the dashed line would be perfect predictions.")

# ---------------------------------------------------------------------------
# Contributing features (feature importance)
# ---------------------------------------------------------------------------
st.subheader(f"📊 Contributing features — {best_name}")
importance = get_feature_importance(best_result["model"], st.session_state.feature_cols)
if importance is not None:
    top_importance = importance.head(15).sort_values(ascending=True)
    fig3, ax3 = plt.subplots(figsize=(6, max(3, 0.3 * len(top_importance))))
    ax3.barh(top_importance.index.astype(str), top_importance.values, color="#27ae60")
    ax3.set_xlabel("Relative importance" if hasattr(best_result["model"], "feature_importances_")
                   else "|Coefficient|")
    st.pyplot(fig3)
else:
    st.info(f"Feature importance isn't available for {best_name} with its current settings "
            f"(e.g. SVM with a non-linear kernel doesn't expose one).")

# ---------------------------------------------------------------------------
# Automated insight summary (generated locally, no API needed)
# ---------------------------------------------------------------------------
st.header("2️⃣ Automated insight & recommendation")
if is_classification:
    cm = best_result["confusion_matrix"]
    insight_text = build_classification_insight(
        metrics_df, best_name, cm, st.session_state.target_col, st.session_state.cleaning_log
    )
else:
    insight_text = build_regression_insight(
        metrics_df, best_name, st.session_state.target_col, st.session_state.cleaning_log
    )
st.markdown(insight_text)
st.caption("Generated locally from the metrics above — no external API call.")

# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
st.header("3️⃣ Get predictions")
model_choice = st.selectbox(
    "Choose a model to use for predictions",
    options=list(st.session_state.results.keys()),
    index=list(st.session_state.results.keys()).index(best_name),
)
model = st.session_state.results[model_choice]["model"]
scaler = st.session_state.scaler
feature_cols = st.session_state.feature_cols
stats = st.session_state.feature_stats

tab1, tab2 = st.tabs(["✍️ Manual entry", "📄 Batch upload (CSV)"])

with tab1:
    if is_classification:
        st.write("Enter values for one country/year to check its recession risk.")
    else:
        st.write(f"Enter values for one row to predict {st.session_state.target_col}.")
    input_vals = {}
    n_cols = 3
    cols = st.columns(n_cols)
    for i, feat in enumerate(feature_cols):
        s = stats[feat]
        with cols[i % n_cols]:
            input_vals[feat] = st.number_input(feat, value=round(s["mean"], 3), format="%.4f")

    predict_label = "Predict recession risk" if is_classification else f"Predict {st.session_state.target_col}"
    if st.button(predict_label, type="primary"):
        X_new = pd.DataFrame([input_vals])[feature_cols]
        X_new_s = scaler.transform(X_new)
        if is_classification:
            prob = model.predict_proba(X_new_s)[0, 1]
            pred = int(prob >= 0.5)
            if pred == 1:
                st.error(f"⚠️ HIGH recession risk — predicted probability: {prob:.1%}")
            else:
                st.success(f"✅ LOW recession risk — predicted probability: {prob:.1%}")
            st.progress(min(max(prob, 0.0), 1.0))
        else:
            value = model.predict(X_new_s)[0]
            st.success(f"Predicted {st.session_state.target_col}: **{value:,.4f}**")

with tab2:
    st.write(f"Upload a CSV with these columns: `{', '.join(feature_cols)}`")
    batch_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch_upload")
    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        missing = [c for c in feature_cols if c not in batch_df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        else:
            X_batch = batch_df[feature_cols]
            X_batch_s = scaler.transform(X_batch)
            out_df = batch_df.copy()
            if is_classification:
                probs = model.predict_proba(X_batch_s)[:, 1]
                preds = (probs >= 0.5).astype(int)
                out_df["recession_risk_probability"] = probs.round(4)
                out_df["recession_predicted"] = np.where(preds == 1, "HIGH RISK", "LOW RISK")
                download_name = "recession_predictions.csv"
            else:
                preds = model.predict(X_batch_s)
                out_df[f"predicted_{st.session_state.target_col}"] = np.round(preds, 4)
                download_name = "predictions.csv"
            st.dataframe(out_df, use_container_width=True)
            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download predictions as CSV", data=csv_bytes,
                                file_name=download_name, mime="text/csv")
