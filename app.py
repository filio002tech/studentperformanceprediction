import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────── App Config (SDU Ozoro) ─────────────────────────── #
st.set_page_config(
    page_title="SDU Ozoro • Student Performance Predictor",
    layout="centered",
    page_icon="🎓"
)
# Reserve space under the Streamlit toolbar + style the logo only
st.markdown(""""
<style>
  /* Push main content below the toolbar (Deploy) */
  .main .block-container { padding-top: 7.25rem !important; }

  /* Scoped logo styling so it stands out and doesn't affect other images */
  .sdu-logo {
    display:inline-block;
    border: 2px solid #E5E7EB;
    border-radius: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    overflow:hidden;
  }
  .sdu-logo img { display:block; }
</style>
""", unsafe_allow_html=True)

PRIMARY = "#1E40AF"      # SDU blue
ACCENT = "#10B981"       # soft green accent
BG_SOFT = "#F8FAFF"      # very light background

# ─────────────────────────── Paths ─────────────────────────── #
MODEL_PATH = Path("models/best_random_forest_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")
LOGO_PATH = Path("assets/sdu_logo.png")  # optional; safe if missing

# Feature order MUST match the model's training
FEATURE_NAMES = [
    "Age", "Gender", "Ethnicity", "ParentalEducation", "StudyTimeWeekly",
    "Absences", "Tutoring", "ParentalSupport", "Extracurricular",
    "Sports", "Music", "Volunteering"
]

# Categorical mappings — ensure they match TRAINING
GENDER_MAP = {"Male": 1, "Female": 0}
ETHNICITY_MAP = {"Group A": 0, "Group B": 1, "Group C": 2, "Group D": 3, "Group E": 4}
EDU_MAP = {"High School": 0, "Associate": 1, "Bachelor": 2, "Master": 3, "PhD": 4}
SUPPORT_MAP = {"Low": 0, "Moderate": 1, "High": 2, "Very High": 3, "Excellent": 4}
YESNO_MAP = {"Yes": 1, "No": 0}

# ─────────────────────────── Theming / CSS ─────────────────────────── #
st.markdown(
    f"""
    <style>
        .sdu-header {{
            display: flex; align-items: center; gap: 14px; margin-bottom: 6px;
        }}
        .sdu-title {{
            font-weight: 800; font-size: 1.25rem; color: {PRIMARY}; line-height: 1.1;
        }}
        .sdu-subtitle {{
            color: #334155; font-size: 0.90rem; margin-top: -2px;
        }}
        .sdu-badge {{
            display:inline-block; padding: 4px 10px; border-radius: 999px;
            background: {ACCENT}20; color: {ACCENT}; font-weight: 600; font-size: .75rem;
            border: 1px solid {ACCENT}55;
        }}
        .sdu-card {{
            background: white; border: 1px solid #E5E7EB; border-radius: 14px;
            padding: 16px; box-shadow: 0 1px 5px rgba(0,0,0,0.04);
        }}
        .sdu-footer {{
            color:#64748B; font-size:.8rem; margin-top: 20px;
            border-top:1px dashed #E5E7EB; padding-top:10px;
        }}
        .block-container {{ padding-top: 1rem; }}
        body, .stApp {{ background: {BG_SOFT}; }}
    </style>
    """, unsafe_allow_html=True
)

# ─────────────────────── Session State Defaults ─────────────────────── #
for key, default in {
    "prediction": None,
    "pred_proba": None,
    "submitted": False,
    "show_insights": False,
    "input_raw": {},
    "input_encoded": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────── Utilities ─────────────────────────── #
@st.cache_resource
def load_model_and_scaler():
    if not MODEL_PATH.exists():
        st.error(f"[SDU] Model file not found at: {MODEL_PATH}")
        st.stop()
    if not SCALER_PATH.exists():
        st.error(f"[SDU] Scaler file not found at: {SCALER_PATH}")
        st.stop()
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def encode_inputs(age, gender, ethnicity, parental_education, study_time, absences,
                  tutoring, parental_support, extracurricular, sports, music, volunteering) -> pd.DataFrame:
    try:
        row = [
            int(age),
            GENDER_MAP[gender],
            ETHNICITY_MAP[ethnicity],
            EDU_MAP[parental_education],
            int(study_time),
            int(absences),
            YESNO_MAP[tutoring],
            SUPPORT_MAP[parental_support],
            YESNO_MAP[extracurricular],
            int(sports),
            int(music),
            int(volunteering),
        ]
    except KeyError as e:
        st.error(f"[SDU] Encoding error: value {e} not found in mapping. Ensure dropdown choices match training categories.")
        st.stop()
    X = pd.DataFrame([row], columns=FEATURE_NAMES)
    return X

def plot_feature_importances_plotly(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        if importances.shape[0] != len(feature_names):
            st.warning("[SDU] Feature importance vector length does not match FEATURE_NAMES. Skipping chart.")
            return
        order = np.argsort(importances)
        df = pd.DataFrame({
            "Feature": [feature_names[i] for i in order],
            "Importance": importances[order]
        })
        fig = px.bar(
            df, x="Importance", y="Feature", orientation="h",
            title="Feature Importances", labels={"Importance": "Relative Importance"},
            color="Importance", color_continuous_scale="Blues", height=520, template="plotly_white"
        )
        fig.update_traces(hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("[SDU] Model does not expose feature importances.")

def performance_gauge(pred_label: str, class_names: np.ndarray, proba: np.ndarray):
    try:
        idx = list(class_names).index(pred_label)
        value = float(proba[idx]) * 100.0
    except Exception:
        value = 55.0
    color = "green" if value >= 70 else ("orange" if value >= 40 else "red")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Performance Potential"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "rgba(255,0,0,0.25)"},
                {'range': [40, 70], 'color': "rgba(255,165,0,0.25)"},
                {'range': [70, 100], 'color': "rgba(0,128,0,0.25)"},
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

def download_report_button(payload: dict):
    df = pd.DataFrame([payload])
    json_str = df.to_json(orient="records", indent=2)
    st.download_button("⬇️ Download SDU Prediction Report (JSON)",
                       json_str, file_name="SDU_Prediction_Report.json",
                       mime="application/json")

# ─────────────────────────── Header / Sidebar ─────────────────────────── #
with st.container():
    cols = st.columns([1, 8, 3])
    with cols[0]:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=56)
        else:
            st.markdown(f"<div class='sdu-badge'>SDU Ozoro</div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(
            f"""
            <div class="sdu-header">
              <div>
                <div class="sdu-title">Southern Delta University (SDU), Ozoro</div>
                <div class="sdu-subtitle">Faculty of Computing • Student Performance Prediction Tool</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(f"<div style='text-align:right' class='sdu-badge'>Academic Use</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("About")
    st.write(
        "This tool assists academic advisors at **SDU, Ozoro** in assessing student performance risk "
        "based on study habits and support indicators. It is a decision‑support system and does not replace "
        "professional judgment."
    )
    st.divider()
    st.subheader("Contacts")
    st.caption("Student Affairs • sdu.studentaffairs@dsust.edu.ng")
    st.caption("Dept. of Computer Science • compsci@dsust.edu.ng")
    st.divider()
    st.subheader("Notes")
    st.caption("• Ensure model encodings match training.\n• Use with anonymized data where possible.")

# ─────────────────────────── Main UI ─────────────────────────── #
st.markdown("<div class='sdu-card'>", unsafe_allow_html=True)
st.subheader("🎓 Student Performance Predictor (SDU‑Ozoro)")
st.caption("Predict category and explore model insights. For academic advising only.")

model, scaler = load_model_and_scaler()

with st.form("prediction_form", clear_on_submit=False):
    st.markdown("#### Enter Student Data")
    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input("Age", min_value=10, max_value=30, value=18, step=1)
        gender = st.selectbox("Gender", options=list(GENDER_MAP.keys()))
        ethnicity = st.selectbox("Ethnicity Group", options=list(ETHNICITY_MAP.keys()))
        parental_education = st.selectbox("Parental Education", options=list(EDU_MAP.keys()))
        study_time = st.slider("Weekly Study Time (hours)", 0, 50, 12)

    with c2:
        absences = st.slider("Number of Absences", 0, 60, 5)
        tutoring = st.selectbox("Attends Tutoring?", options=list(YESNO_MAP.keys()))
        parental_support = st.selectbox("Parental Support", options=list(SUPPORT_MAP.keys()))
        extracurricular = st.selectbox("Extracurricular Activities?", options=list(YESNO_MAP.keys()))
        sports = st.slider("Weekly Hours in Sports", 0, 30, 2)

    with st.expander("More factors (optional)", expanded=False):
        music = st.slider("Weekly Hours in Music", 0, 30, 0)
        volunteering = st.slider("Weekly Hours in Volunteering", 0, 30, 0)

    submitted = st.form_submit_button("🔮 Predict (SDU)")

st.markdown("</div>", unsafe_allow_html=True)

if submitted:
    X = encode_inputs(age, gender, ethnicity, parental_education, study_time, absences,
                      tutoring, parental_support, extracurricular, sports, music, volunteering)

    if X.shape[1] != len(FEATURE_NAMES):
        st.error("[SDU] Encoded feature vector length mismatch. Check FEATURE_NAMES and encoders.")
        st.stop()

    try:
        Xs = scaler.transform(X)
    except Exception as e:
        st.error(f"[SDU] Scaler.transform failed: {e}")
        st.stop()

    try:
        y_pred = model.predict(Xs)
        pred_label = str(y_pred[0])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(Xs)[0]
            class_names = getattr(model, "classes_", np.array(["Low", "Average", "Passing"]))
        else:
            proba = np.array([0.0] * 3)
            class_names = np.array(["Low", "Average", "Passing"])
    except Exception as e:
        st.error(f"[SDU] Model prediction failed: {e}")
        st.stop()

    # Save to session
    st.session_state.submitted = True
    st.session_state.prediction = pred_label
    st.session_state.pred_proba = dict(zip(map(str, class_names), map(float, proba))) if proba is not None else None
    st.session_state.input_raw = {
        "Age": age, "Gender": gender, "Ethnicity": ethnicity, "ParentalEducation": parental_education,
        "StudyTimeWeekly": study_time, "Absences": absences, "Tutoring": tutoring,
        "ParentalSupport": parental_support, "Extracurricular": extracurricular,
        "Sports": sports, "Music": music, "Volunteering": volunteering,
    }
    st.session_state.input_encoded = X

# ─────────────────────────── Results & Insights ─────────────────────────── #
if st.session_state.submitted and st.session_state.prediction is not None:
    st.markdown("<div class='sdu-card'>", unsafe_allow_html=True)
    st.subheader("Result (SDU‑Ozoro)")
    pred = st.session_state.prediction

    if pred.lower().startswith("pass"):
        st.success("✅ The student is likely to pass.")
    elif pred.lower().startswith("avg") or pred.lower().startswith("border"):
        st.warning("⚠️ The student is borderline. Recommend extra attention.")
    else:
        st.error("❌ High risk of failing. Immediate intervention advised.")

    if st.session_state.pred_proba:
        st.markdown("**Class probabilities:**")
        st.dataframe(pd.DataFrame([st.session_state.pred_proba]), use_container_width=True, hide_index=True)

    # Download report
    report_payload = {
        "institution": "Southern Delta University (SDU), Ozoro",
        "prediction": st.session_state.prediction,
        "probabilities": st.session_state.pred_proba,
        **{f"input_{k}": v for k, v in st.session_state.input_raw.items()},
    }
    download_report_button(report_payload)

    st.divider()
    left, right = st.columns([1, 1])
    with left:
        if st.button(("Hide" if st.session_state.show_insights else "Show") + " Detailed Insights"):
            st.session_state.show_insights = not st.session_state.show_insights
    with right:
        st.caption("Note: Ensure dropdown encodings match the training pipeline.")

    if st.session_state.show_insights:
        st.markdown("### 📈 Performance Potential")
        class_names_arr = (np.array(list(st.session_state.pred_proba.keys()))
                           if st.session_state.pred_proba else np.array(["Low","Average","Passing"]))
        proba_arr = (np.array(list(st.session_state.pred_proba.values()))
                     if st.session_state.pred_proba else np.array([0.2,0.3,0.5]))
        performance_gauge(st.session_state.prediction, class_names_arr, proba_arr)

        st.markdown("### 🔍 Feature Importances")
        plot_feature_importances_plotly(model, FEATURE_NAMES)

        st.markdown("### 👤 Student Profile Snapshot")
        colA, colB = st.columns(2)
        raw = st.session_state.input_raw
        with colA:
            st.metric("Weekly Study Time", f"{raw['StudyTimeWeekly']} hrs", help="Recommended: 15–25 hrs")
            st.metric("Absences", raw["Absences"], help="Recommended: ≤5")
        with colB:
            st.metric("Parental Support", raw["ParentalSupport"], help="Higher support often correlates with performance")
            st.metric("Extracurricular", "Active" if raw["Extracurricular"] == "Yes" else "Not Active")

        st.markdown("### ⚠️ Risk Factors")
        risks = []
        if raw["StudyTimeWeekly"] < 10: risks.append("Low study time")
        if raw["Absences"] > 10: risks.append("High absences")
        if SUPPORT_MAP[raw["ParentalSupport"]] < 2: risks.append("Limited parental support")
        if YESNO_MAP[raw["Tutoring"]] == 0: risks.append("No tutoring support")
        if risks: 
            for r in risks: st.error(f"• {r}")
        else:
            st.success("No major risk factors identified.")

        st.markdown("### 💡 Performance Improvement Plan")
        tips = []
        if raw["StudyTimeWeekly"] < 15:
            tips.append(("⏱️ Increase study time", "Aim for 15–20 hrs/week in 45‑minute sessions."))
        if raw["Absences"] > 10:
            tips.append(("📝 Reduce absences", "Maintain consistent attendance to improve outcomes."))
        if YESNO_MAP[raw["Tutoring"]] == 0:
            tips.append(("👨‍🏫 Consider tutoring", "Targeted help can lift grades substantially."))
        if SUPPORT_MAP[raw["ParentalSupport"]] < 2:
            tips.append(("👪 Enhance support", "Family/mentor engagement boosts motivation."))
        if raw["Sports"] > 15:
            tips.append(("⚖️ Balance sports", "Over 15 hrs/week may impact study focus."))
        if tips:
            st.info("Based on this profile, consider:")
            for title, note in tips:
                st.markdown(f"**{title}** — {note}")
        else:
            st.success("Great indicators! Keep reinforcing current habits.")
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────── Compliance Footer ─────────────────────────── #
st.markdown(
    f"""
    <div class="sdu-footer">
      <strong>SDU, Ozoro – Academic Use Only.</strong><br/>
      This tool provides decision support and should not be the sole basis for academic actions. 
      Protect student privacy: avoid personally identifiable information. Use anonymized records wherever possible.
    </div>
    """,
    unsafe_allow_html=True
)
