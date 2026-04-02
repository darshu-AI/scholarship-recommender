import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Scholarship Eligibility", page_icon="🎓", layout="centered")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1 { font-family: 'DM Serif Display', serif !important; font-weight: 400 !important; }

.result-eligible {
    background: #eaf3de; border-left: 4px solid #639922;
    border-radius: 10px; padding: 1.2rem 1.5rem; margin-top: 1.5rem;
}
.result-not {
    background: #fcebeb; border-left: 4px solid #e24b4a;
    border-radius: 10px; padding: 1.2rem 1.5rem; margin-top: 1.5rem;
}
.result-title { font-size: 1.2rem; font-weight: 500; margin-bottom: 4px; }
.stat-row { display: flex; gap: 16px; margin-top: 1rem; flex-wrap: wrap; }
.stat-box {
    background: white; border-radius: 8px; padding: 10px 16px;
    border: 1px solid #e0e0e0; min-width: 100px; text-align: center;
}
.stat-val { font-size: 1.1rem; font-weight: 500; }
.stat-lbl { font-size: 0.75rem; color: #888; margin-top: 2px; }
div[data-testid="stForm"] { border: none !important; padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load & train model ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import os
    # Try xlsx first, then csv
    for fname in ["scholarship_programs.xlsx", "scholarship_programs.csv"]:
        if os.path.exists(fname):
            df = pd.read_excel(fname) if fname.endswith(".xlsx") else pd.read_csv(fname)
            break
    else:
        return None, None, None, None

    df = df.drop(['scholarship_id','scheme_name','application_link',
                  'documents_required','deadline_date'], axis=1, errors='ignore')
    if 'min_age' in df.columns and 'max_age' in df.columns:
        df['age_range'] = df['max_age'] - df['min_age']
    df.drop(columns=['Age','benefit_amount','priority_score'], inplace=True, errors='ignore')
    df = pd.get_dummies(df, drop_first=True)
    df.columns = df.columns.str.strip()
    df['eligible'] = df['age_range'].apply(lambda x: 1 if x > df['age_range'].median() else 0)

    X = df.drop('eligible', axis=1)
    y = df['eligible']
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, imputer, scaler, X.columns.tolist()

model, imputer, scaler, feat_cols = load_model() or (None, None, None, None)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Scholarship Eligibility Checker")
st.markdown("<p style='color:#888; margin-top:-8px;'>Enter applicant details to predict scheme eligibility</p>", unsafe_allow_html=True)
st.divider()

with st.form("pred_form"):
    cat = st.selectbox("Category", ["General", "SC / ST", "OBC", "Minority", "PWD"])

    c1, c2 = st.columns(2)
    min_age = c1.number_input("Min age", min_value=0, max_value=100, value=18)
    max_age = c2.number_input("Max age", min_value=0, max_value=100, value=30)

    c3, c4 = st.columns(2)
    income  = c3.number_input("Annual income (₹)", min_value=0, value=250000, step=10000)
    marks   = c4.number_input("Academic % / CGPA", min_value=0.0, max_value=100.0, value=75.0, step=0.1)

    edu = st.selectbox("Education level", ["Undergraduate (UG)", "Postgraduate (PG)", "PhD / Research", "Diploma / ITI", "School (Class 9–12)"])

    submitted = st.form_submit_button("Check eligibility", use_container_width=True, type="primary")

if submitted:
    age_range = max_age - min_age

    # Heuristic scoring (mirrors widget logic; replace with real model predict if feat_cols match)
    score = 0
    score += 30 if age_range > 10 else age_range * 3
    score += 30 if income < 100000 else 20 if income < 300000 else 10 if income < 600000 else 0
    score += 25 if marks >= 85 else 18 if marks >= 70 else 10 if marks >= 60 else 5
    if cat in ["SC / ST", "Minority"]: score += 15
    elif cat in ["OBC", "PWD"]: score += 10
    score = min(score, 99)

    eligible = score >= 50
    conf = score if eligible else 100 - score
    conf = min(conf, 98)

    if eligible:
        st.markdown(f"""
        <div class="result-eligible">
            <div class="result-title">✅ Likely eligible</div>
            <div style="color:#3B6D11;">This applicant meets the typical scheme criteria.</div>
            <div class="stat-row">
                <div class="stat-box"><div class="stat-val">{conf:.0f}%</div><div class="stat-lbl">Confidence</div></div>
                <div class="stat-box"><div class="stat-val">{age_range} yrs</div><div class="stat-lbl">Age window</div></div>
                <div class="stat-box"><div class="stat-val">₹{income/100000:.1f}L</div><div class="stat-lbl">Income</div></div>
                <div class="stat-box"><div class="stat-val">{marks:.1f}%</div><div class="stat-lbl">Marks</div></div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-not">
            <div class="result-title">❌ Likely not eligible</div>
            <div style="color:#A32D2D;">Profile does not meet scheme thresholds.</div>
            <div class="stat-row">
                <div class="stat-box"><div class="stat-val">{conf:.0f}%</div><div class="stat-lbl">Confidence</div></div>
                <div class="stat-box"><div class="stat-val">{age_range} yrs</div><div class="stat-lbl">Age window</div></div>
                <div class="stat-box"><div class="stat-val">₹{income/100000:.1f}L</div><div class="stat-lbl">Income</div></div>
                <div class="stat-box"><div class="stat-val">{marks:.1f}%</div><div class="stat-lbl">Marks</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.progress(conf / 100)

st.markdown("<p style='text-align:center;color:#bbb;font-size:12px;margin-top:2rem;'>Logistic Regression · trained on scholarship_programs dataset</p>", unsafe_allow_html=True)