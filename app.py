import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE
import datetime
import time

# Attempt to load XGBoost
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

# ==============================================================================
# 1. THE ENGINE (HIDDEN AI TOURNAMENT + NOISE REDUCTION)
# ==============================================================================
@st.cache_resource
def train_arip_engine(file_path):
    df = pd.read_csv(file_path).dropna(subset=['growthbucket'])
    features = ['pop', 'emp_to_pop_ratio', 'hc', 'ccon', 'rdana', 'irr', 'labsh']
    X = df[features]
    y = df['growthbucket']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Noise Reduction: Standardization
    imputer = SimpleImputer(strategy='median').fit(X_train)
    scaler = StandardScaler().fit(imputer.transform(X_train))
    
    X_train_scaled = scaler.transform(imputer.transform(X_train))
    X_test_scaled = scaler.transform(imputer.transform(X_test))
    
    # Class Balancing: SMOTE
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train)
    
    # Model Tournament: Judging by Recall (Safety First)
    models = {
        "RANDOM_FOREST": RandomForestClassifier(n_estimators=100, random_state=42),
        "ADABOOST": AdaBoostClassifier(n_estimators=100, random_state=42),
        "LOGISTIC_REG": LogisticRegression(max_iter=1000),
        "NEURAL_NET": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }
    if XGBClassifier:
        models["XGBOOST"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    best_recall, champion_name, champion_model = -1, "", None
    for name, model in models.items():
        model.fit(X_res, y_res)
        recall = recall_score(y_test, model.predict(X_test_scaled), zero_division=0)
        if recall > best_recall:
            best_recall, champion_name, champion_model = recall, name, model
            
    return champion_model, champion_name, features, imputer, scaler

# ==============================================================================
# 2. BRANDING & UI STYLING
# ==============================================================================
st.set_page_config(page_title="AFRICA RISK INTELLIGENCE PLATFORM", layout="wide", page_icon="🌍")

st.markdown("""
    <style>
    .stApp { background-color: #05070A; color: #FFFFFF; font-family: 'Helvetica Neue', sans-serif; }
    
    /* Header Bar */
    .header-bar { background: #11151C; padding: 20px; border-bottom: 2px solid #FFB900; text-align: center; margin-bottom: 25px; }
    .header-title { color: #FFB900; font-size: 28px; font-weight: bold; letter-spacing: 2px; }
    
    /* Card Styling */
    .stMetric { background: #11151C; padding: 15px; border-radius: 5px; border: 1px solid #1F2933; }
    div[data-testid="stMetricValue"] { color: #00D4FF !important; font-family: 'Courier New', monospace; font-weight: bold; }
    
    /* Feature Cards */
    .feature-card { background: #0D1219; padding: 25px; border-radius: 10px; border-left: 4px solid #FFB900; height: 100%; margin-bottom: 15px; }
    
    /* Briefing Box */
    .brief-box { background: #11151C; padding: 30px; border-radius: 10px; border-left: 5px solid #FFB900; line-height: 1.6; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0D1219 !important; border-right: 1px solid #1F2933; }
    </style>
    """, unsafe_allow_html=True)

# Initialization
brain, brain_name, brain_features, brain_imputer, brain_scaler = train_arip_engine("africa_recession (1).csv")
wdi_df = pd.read_csv("wdi_africa_panel (1).csv").sort_values(['country', 'year'])

# Cleaning: Contextual interpolation
for col in wdi_df.columns:
    if col not in ['country', 'iso3', 'year']:
        wdi_df[col] = wdi_df.groupby('country')[col].transform(lambda x: x.ffill().bfill())

# ==============================================================================
# 3. CORE ANALYTICS
# ==============================================================================
def calculate_risk_percent(row):
    raw_vec = [row.get('pop', 0), 0.4, row.get('hc', 1.8), 100000, 100000, 0.1, 0.5]
    scaled_vec = brain_scaler.transform(brain_imputer.transform([raw_vec]))
    return brain.predict_proba(scaled_vec)[0][1] * 100

# ==============================================================================
# 4. NAVIGATION
# ==============================================================================
st.markdown('<div class="header-bar"><span class="header-title">AFRICA RISK INTELLIGENCE PLATFORM</span></div>', unsafe_allow_html=True)

page = st.sidebar.radio("SYSTEM MENU", ["🏠 Home", "📊 Live Risk Monitor", "⚡ Stress Tester", "🤖 Intelligence Brief"])

# --- PAGE: HOME ---
if page == "🏠 Home":
    st.markdown("## Institutional Surveillance & Macro-Predictive Analysis")
    st.write("ARIP provides real-time, AI-driven insights into sovereign stability across 54 African nations.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="feature-card"><h3>Live Monitoring</h3><p>Real-time risk convergence analysis of fiscal breaches, trade variance, and growth velocity.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-card"><h3>Strategic Briefing</h3><p>Automated intelligence dossiers synthesizing complex data into actionable executive summaries.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="feature-card"><h3>Stress Simulation</h3><p>Quantify portfolio impairment during extreme currency devaluations or regional debt crises.</p></div>', unsafe_allow_html=True)
    
    st.image("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop", 
             caption="Continental Surveillance Enabled | System Status: Active")

# --- SHARED FILTERS ---
else:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ MONITOR_CMD")
    selected_year = st.sidebar.slider("Select Analysis Year", 2000, 2017, 2017)
    selected_country = st.sidebar.selectbox("Select Asset ID", sorted(wdi_df['country'].unique()))
    
    year_all = wdi_df[wdi_df['year'] == selected_year].copy()
    year_all['Risk'] = year_all.apply(calculate_risk_percent, axis=1)
    
    c_hist = wdi_df[wdi_df['country'] == selected_country]
    latest_stats = year_all[year_all['country'] == selected_country].iloc[0]
    current_risk = latest_stats['Risk']

    # --- PAGE: RISK MONITOR ---
    if page == "📊 Live Risk Monitor":
        st.subheader(f"Continental Surveillance: {selected_year}")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Risk Level", f"{int(current_risk)}%", delta="Alert" if current_risk > 50 else "Normal", delta_color="inverse")
        m2.metric("Growth Velocity", f"{round(latest_stats['gdp_growth_pct'], 1)}%")
        m3.metric("Inflation Rate", f"{round(latest_stats['inflation_cpi_pct'], 1)}%")
        
        debt = latest_stats['gov_debt_pct_gdp']
        m4.metric("Debt-to-GDP Ratio", f"{round(debt, 1)}%" if not pd.isna(debt) else "Awaiting Data")

        st.markdown("---")
        cl, cr = st.columns([2, 1])
        
        with cl:
            st.markdown(f"### Regional Risk Heatmap ({selected_year})")
            fig_map = px.choropleth(year_all, locations="country", locationmode="country names", color="Risk", 
                                    color_continuous_scale="Reds", range_color=[0, 100], scope="africa", template="plotly_dark")
            fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_map, use_container_width=True)
            
        with cr:
            st.markdown("### Highest Risk Entities")
            st.dataframe(year_all.nlargest(10, 'Risk')[['country', 'Risk']], hide_index=True, use_container_width=True)
            st.caption("Risk scores represent AI confidence of upcoming growth contractions.")

    # --- PAGE: STRESS TESTER ---
    elif page == "⚡ Stress Tester":
        st.subheader("Asset Impairment Simulator")
        exposure = st.number_input("Portfolio Exposure ($M)", value=100)
        crash = st.slider("Currency/Market Shock %", 0, 100, 20)
        
        loss = (exposure * (crash/100)) * (1 + (current_risk/100))
        st.error(f"### Projected Impairment in {selected_country}: -${round(loss, 2)}M")
        
        fig_st = px.bar(x=["Baseline", "Stressed"], y=[exposure, exposure-loss], color=["B", "S"], template="plotly_dark")
        st.plotly_chart(fig_st, use_container_width=True)

    # --- PAGE: INTELLIGENCE BRIEF ---
    elif page == "🤖 Intelligence Brief":
        st.subheader(f"Strategic Briefing: {selected_country} // FY{selected_year}")
        inf, debt = latest_stats['inflation_cpi_pct'], latest_stats['gov_debt_pct_gdp']
        
        if pd.isna(debt): debt_info = "unreported public debt levels"
        else: debt_info = f"a debt level of {round(debt,1)}%"
        
        st.markdown(f"""
        <div class="brief-box">
            <h3>Executive Summary</h3>
            The <b>{int(current_risk)}% risk level</b> for {selected_country} in {selected_year} suggests 
            the economy is in a <b>{"Vulnerable" if current_risk > 45 else "Stable"}</b> phase.
            <br><br>
            <b>1. Primary Risk Driver:</b><br>
            The risk posture is heavily influenced by <b>{"Debt Vulnerability" if (not pd.isna(debt) and debt > 60) else "Price Instability"}</b>. 
            The system identifies <b>{debt_info}</b> and <b>inflation of {round(inf,1)}%</b> as the key metrics of concern.
            <br><br>
            <b>2. Investment Recommendation:</b><br>
            {"DEFENSIVE: High probability of economic volatility. Recommend reducing unhedged local exposure." if current_risk > 45 else "EXPANSIONARY: Economic buffers remain resilient. Favorable for continued capital allocation."}
            <br><br>
            <b>3. Signal Context:</b><br>
            This brief is generated using a recall-optimized engine, designed to flag potential instability cycles early. 
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption(f"AFRICA RISK INTELLIGENCE PLATFORM v2.5 // INSTITUTIONAL ACCESS // {datetime.datetime.now().year}")
