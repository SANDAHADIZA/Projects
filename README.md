# African Recession Predictor — Streamlit App

A no-code interface for your recession prediction capstone. Non-technical
users can upload the training dataset, compare all 5 models, and get
predictions without touching code.

## What it does
1. **Train** — upload your dataset (e.g. the 27-country panel with
   `growthbucket`), pick the target and feature columns, and click one
   button to train Logistic Regression, Random Forest, XGBoost, SVM, and
   AdaBoost with SMOTE balancing.
2. **Compare** — see Recall, F1-score, and AUC-PR for all 5 models side by
   side, with a bar chart.
3. **Predict** — pick a model, then either:
   - fill in a simple form for one country/year, or
   - upload a CSV of multiple countries/years for batch predictions with a
     downloadable results file.

## Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
This opens the app in your browser at `http://localhost:8501`.

## Deploy for free (Streamlit Community Cloud) — recommended for sharing
1. Push `app.py` and `requirements.txt` to a GitHub repo (public or
   private).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, select the repo/branch, set the main file to
   `app.py`, and click **Deploy**.
4. You'll get a shareable link like
   `https://your-app-name.streamlit.app` — send that to anyone; no
   installation needed on their end.

## Notes
- The app trains fresh on whatever CSV is uploaded — it does not reuse
  your saved Colab notebook state, so metrics may differ slightly from
  your capstone report unless you use the exact same train/test split
  and preprocessing.
- If `xgboost` or `imbalanced-learn` fail to install on a deployment
  platform, the app still runs with the remaining 3–4 models — SMOTE and
  XGBoost are auto-detected and skipped gracefully if unavailable.
- Column names are auto-detected from whatever CSV you upload, so this
  works with your dataset's actual schema rather than a hardcoded one.
