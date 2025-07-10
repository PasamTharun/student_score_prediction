import streamlit as st
import pandas as pd
import joblib
import os
import sys
import shap
import dice_ml
from fpdf import FPDF
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
from raiutils.exceptions import UserConfigValidationException

# --- Robust Path Handling for Deployment ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.recommendations.generate_recommendations import generate_recommendations

# --- Page Configuration ---
st.set_page_config(page_title="Individual Predictor", page_icon="👤", layout="wide")

# --- Define Paths Relative to Project Root ---
PREPROCESSOR_PATH = os.path.join(project_root, 'data', 'processed', 'preprocessor.pkl')
MODEL_PATH = os.path.join(project_root, 'models', 'xgboost', 'model.pkl')
RAW_DATA_PATH = os.path.join(project_root, 'data', 'raw', 'dataset.csv')
ATTENDANCE_THRESHOLD = 75

# --- Initialize Session State ---
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.prediction = 0
    st.session_state.input_data = None
    st.session_state.processed_input = None
    st.session_state.recommendations = []

# --- Load Assets ---
@st.cache_resource
def load_assets():
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(MODEL_PATH)
        return preprocessor, model
    except FileNotFoundError:
        st.error("Error: Model or preprocessor not found. Please run the training scripts.")
        return None, None

preprocessor, model = load_assets()

# --- Sidebar ---
st.sidebar.title("App Controls")
if st.sidebar.button("Clear App Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# --- Main Application ---
st.title("👤 Individual Student Performance Predictor")
st.markdown("Enter a student's details to forecast their score, understand the prediction, and get actionable advice.")

if model is None or preprocessor is None:
    st.stop()

# --- Prediction Section ---
with st.form("prediction_form"):
    st.header("📝 Student Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        hours_studied = st.number_input("Hours Studied per Week", 0.0, 40.0, 10.0)
    with col2:
        previous_score = st.number_input("Previous Exam Score", 0.0, 100.0, 75.0)
    with col3:
        attendance_percentage = st.number_input("Attendance Percentage", 0.0, 100.0, 90.0)
    col4, col5 = st.columns(2)
    with col4:
        stress_level = st.selectbox("Stress Level", ['Low', 'Medium', 'High'])
    with col5:
        confidence_level = st.selectbox("Confidence Level", ['Low', 'Medium', 'High'])

    submit_button = st.form_submit_button("Analyze Student Performance")

if submit_button:
    if attendance_percentage < ATTENDANCE_THRESHOLD:
        st.session_state.prediction_made = False
        st.error(f"**Ineligible for Exam:** This student's attendance ({attendance_percentage}%) is below the required {ATTENDANCE_THRESHOLD}%.")
    else:
        st.session_state.prediction_made = True
        st.session_state.input_data = pd.DataFrame({
            'hours_studied': [hours_studied], 'previous_score': [previous_score],
            'attendance_percentage': [attendance_percentage], 'stress_level': [stress_level],
            'confidence_level': [confidence_level]
        })
        st.session_state.processed_input = preprocessor.transform(st.session_state.input_data)
        prediction_raw = model.predict(st.session_state.processed_input)[0]
        st.session_state.prediction = np.clip(prediction_raw, 0, 100)
        st.session_state.recommendations = generate_recommendations(st.session_state.input_data.iloc[0])

# --- Display Results only if a prediction has been made ---
if st.session_state.prediction_made:
    st.header("📈 Analysis Results")
    st.metric(label="Predicted Final Exam Score", value=f"{st.session_state.prediction:.2f}")

    with st.expander("💡 View Actionable Recommendations"):
        for rec in st.session_state.recommendations:
            st.info(rec)

    st.subheader("🔍 What's Driving This Score?")
    with st.spinner("Generating visual explanation..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(st.session_state.processed_input)
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        plt.title("How Each Factor Contributes to the Score Prediction", fontsize=14, pad=20)
        plt.xlabel("Impact on Model Output (Predicted Score)")
        st.pyplot(fig, bbox_inches='tight')
        plt.close(fig)
        st.caption("The waterfall plot shows how each feature pushes the score from a baseline to the final prediction.")

    st.subheader("🎯 How to Improve? (Counterfactuals)")
    target_score = st.slider("Set a Target Score to Achieve:", min_value=float(st.session_state.prediction), max_value=100.0, value=min(float(st.session_state.prediction) + 5, 100.0))

    if st.button("Generate Improvement Plan"):
        with st.spinner("Finding valid paths to your target score..."):
            try:
                full_raw_data = pd.read_csv(RAW_DATA_PATH)
                d = dice_ml.Data(dataframe=full_raw_data,
                                   continuous_features=['hours_studied', 'previous_score', 'attendance_percentage'],
                                   outcome_name='final_exam_score')
                pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
                m = dice_ml.Model(model=pipeline, backend="sklearn", model_type='regressor')
                exp = dice_ml.Dice(d, m, method="random")

                dice_exp = exp.generate_counterfactuals(st.session_state.input_data, total_CFs=10, desired_range=[target_score, 100.0])
                
                all_suggestions = dice_exp.cf_examples_list[0].final_cfs_df
                valid_suggestions = all_suggestions[all_suggestions['attendance_percentage'] >= ATTENDANCE_THRESHOLD]

                if not valid_suggestions.empty:
                    st.write(f"Here are some **valid** ways you could achieve a score of **{target_score}** or higher:")
                    st.dataframe(valid_suggestions.drop(columns=['final_exam_score']))
                else:
                    st.warning("Could not find any improvement plans that also meet the 75% attendance requirement. The required changes might be too significant.")

            except UserConfigValidationException:
                st.warning("No improvement plans could be generated for this target. This usually means the goal is too ambitious. Please try setting a more modest target score.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

    st.subheader("📄 Download Full Report")
    class PDF(FPDF):
        def header(self): self.set_font('Arial', 'B', 12); self.cell(0, 10, 'Student Performance Report', 0, 1, 'C')
        def footer(self): self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14); pdf.cell(0, 10, "Summary", 0, 1)
    pdf.set_font('Arial', '', 12); pdf.cell(0, 10, f"Predicted Score: {st.session_state.prediction:.2f}", 0, 1)
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 14); pdf.cell(0, 10, "Recommendations", 0, 1)
    pdf.set_font('Arial', '', 12)
    for rec in st.session_state.recommendations:
        pdf.multi_cell(0, 10, f"- {rec}")

    pdf_output = bytes(pdf.output())

    st.download_button(label="Download PDF Report", data=pdf_output, file_name=f"student_report_{st.session_state.input_data.iloc[0]['previous_score']:.0f}.pdf", mime="application/pdf")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown("*\"The best way to predict the future is to create it.\"*")
st.sidebar.markdown("Created by **Pasam Tharun**")
