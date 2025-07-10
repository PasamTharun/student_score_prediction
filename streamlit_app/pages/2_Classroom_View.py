import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import google.generativeai as genai

# --- Page Configuration ---
st.set_page_config(page_title="Classroom Dashboard", page_icon="👩‍🏫", layout="wide")

# --- Initialize Chat History in Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar Controls ---
st.sidebar.title("App Controls")
if st.sidebar.button("Clear App Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.messages = []
    st.rerun()

# --- Load Google Gemini API Key from Secrets ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except (KeyError, FileNotFoundError):
    st.error("Google AI API key not found. Please add it to your .streamlit/secrets.toml file.")
    st.stop()


st.title("👩‍🏫 Interactive Classroom Performance Dashboard")
st.markdown("Analyze class-wide performance, filter by achievement tiers, and chat with your data using Google Gemini.")

# --- Load Assets ---
@st.cache_resource
def load_assets():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    try:
        preprocessor = joblib.load(os.path.join(project_root, 'data/processed/preprocessor.pkl'))
        model = joblib.load(os.path.join(project_root, 'models/xgboost/model.pkl'))
        raw_data = pd.read_csv(os.path.join(project_root, 'data/raw/dataset.csv'))
        return preprocessor, model, raw_data
    except FileNotFoundError:
        st.error("Error: Required data or model files not found. Please run the training scripts.")
        return None, None, None

preprocessor, model, raw_data = load_assets()

if model is None:
    st.stop()

# --- Predict for the entire class ---
@st.cache_data
def predict_class_performance(_raw_data):
    processed_data = preprocessor.transform(_raw_data.drop('final_exam_score', axis=1))
    predicted_scores_raw = model.predict(processed_data)
    _raw_data['predicted_score'] = np.clip(predicted_scores_raw, 0, 100)
    
    conditions = [(_raw_data['predicted_score'] >= 90), (_raw_data['predicted_score'] >= 60)]
    choices = ['Topper', 'Medium']
    _raw_data['performance_tier'] = np.select(conditions, choices, default='At-Risk')
    return _raw_data

class_data = predict_class_performance(raw_data.copy())

# --- Interactive Dashboard Section ---
st.sidebar.header("Dashboard Filters")
performance_filter = st.sidebar.selectbox("Select Performance Tier", ["All Students", "Topper", "Medium", "At-Risk"])

if performance_filter == "All Students":
    filtered_data = class_data
else:
    filtered_data = class_data[class_data['performance_tier'] == performance_filter]

# --- Display Dynamic Metrics (as specified by you) ---
st.header(f"📈 Analysis for: {performance_filter}")
col1, col2, col3 = st.columns(3)
col1.metric("Student Count", len(filtered_data))
if not filtered_data.empty:
    col2.metric("Average Predicted Score", f"{filtered_data['predicted_score'].mean():.2f}")
else:
    col2.metric("Average Predicted Score", "N/A")
if performance_filter == "All Students":
    col3.metric("🏆 Topper (Score >= 90)", len(class_data[class_data['performance_tier'] == 'Topper']))
else:
    col3.empty()

# --- Detailed Analysis Tabs (as specified by you) ---
detail_tab1, detail_tab2 = st.tabs(["📋 Student List & Spotlight", "📈 Distributional Analysis"])

with detail_tab1:
    if not filtered_data.empty:
        st.dataframe(filtered_data[['hours_studied', 'previous_score', 'attendance_percentage', 'predicted_score', 'performance_tier']], use_container_width=True)
        st.subheader("🔦 Student Spotlight")
        student_index = st.selectbox(
            "Select a student to view details (by index):",
            filtered_data.index
        )
        if student_index in filtered_data.index:
            spotlight_student = filtered_data.loc[student_index]
            st.write(f"**Viewing Details for Student (Index: {student_index})**")
            card_col1, card_col2, card_col3 = st.columns(3)
            card_col1.metric("Predicted Score", f"{spotlight_student['predicted_score']:.2f}")
            card_col1.metric("Performance Tier", spotlight_student['performance_tier'])
            card_col2.metric("Hours Studied / Week", f"{spotlight_student['hours_studied']:.1f}")
            card_col2.metric("Attendance", f"{spotlight_student['attendance_percentage']:.1f}%")
            card_col3.metric("Stress Level", spotlight_student['stress_level'])
            card_col3.metric("Confidence Level", spotlight_student['confidence_level'])
    else:
        st.warning(f"No students found in the '{performance_filter}' tier.")

with detail_tab2:
    if not filtered_data.empty:
        st.subheader("Predicted Score Distribution")
        st.bar_chart(filtered_data['predicted_score'], use_container_width=True)
        st.subheader("Stress Level Distribution")
        st.bar_chart(filtered_data['stress_level'].value_counts())
    else:
        st.warning(f"No data to display for the '{performance_filter}' tier.")


# --- Conversational AI Chatbot Section ---
st.markdown("---")
st.header("🤖 Ask Gemini About Your Classroom Data")

# Initialize the Gemini model
gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("e.g., What is the average study time for toppers?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Smartly filter data before sending to the AI
    prompt_lower = prompt.lower()
    if "at-risk" in prompt_lower:
        context_data = class_data[class_data['performance_tier'] == 'At-Risk']
    elif "topper" in prompt_lower:
        context_data = class_data[class_data['performance_tier'] == 'Topper']
    else:
        context_data = class_data.sample(n=min(100, len(class_data)))

    data_for_ai = context_data.to_csv(index=False)
    full_prompt = f"You are a helpful data analyst. Analyze the following CSV data about student performance and answer the user's question. Data:\n{data_for_ai}\n\nQuestion: {prompt}"

    # Call the Gemini API
    with st.chat_message("assistant"):
        with st.spinner("Gemini is thinking..."):
            try:
                response = gemini_model.generate_content(full_prompt)
                response_content = response.text
            except Exception as e:
                response_content = f"An error occurred with the Gemini API. Error: {e}"
            
            st.markdown(response_content)
            st.session_state.messages.append({"role": "assistant", "content": response_content})

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown("*\"The goal is to turn data into information, and information into insight.\"*")
st.sidebar.markdown("Created by **Pasam Tharun**")
