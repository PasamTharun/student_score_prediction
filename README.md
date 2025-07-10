# 🎓 Intelligent Student Performance Advisor

An advanced end-to-end data science application that predicts student exam scores using machine learning, explains its predictions with cutting-edge AI, and provides a rich, interactive dashboard for educators to gain actionable insights.

![image](https://github.com/user-attachments/assets/b190583e-039f-40c8-90ce-224d98a2b347)

---

## 📋 Table of Contents

- [📖 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [🛠️ Technology Stack](#-technology-stack)
- [📂 Project Structure](#-project-structure)
- [🚀 Setup and Installation](#-setup-and-installation)
- [⚙️ Execution Workflow](#-execution-workflow)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [👨‍💻 Author](#-author)

---

## 📖 Project Overview

**Intelligent Student Performance Advisor** is a comprehensive system designed to empower educators by forecasting student performance, identifying at-risk individuals, and offering data-driven insights to improve academic outcomes.

It uses a powerful XGBoost model and integrates academic rules to ensure both accurate and practical predictions. The project includes a multi-page Streamlit dashboard:

- 🎯 **Individual Predictor** for in-depth analysis of a single student.
- 📊 **Classroom Dashboard** for high-level classroom performance monitoring.

---

## ✨ Key Features

| Feature                        | Description |
|-------------------------------|-------------|
| **Accurate Predictive Modeling** | Uses XGBoost to forecast exam scores based on study hours, attendance, and prior performance. |
| **Real-World Rule Engine**       | Implements academic policies like a 75% attendance requirement and caps scores at 100. |
| **Explainable AI (XAI)**         | Generates SHAP waterfall plots to explain how each feature influences predictions. |
| **"What-If" Counterfactuals**    | Suggests realistic changes (via DiCE) for students to reach their target scores. |
| **Interactive Classroom Dashboard** | Filters students by categories like “Topper,” “Medium,” or “At-Risk.” |
| **Student Spotlight**           | Lets educators explore detailed metrics of individual students. |
| **Downloadable PDF Reports**    | Generates clean, printable performance summaries for students. |
| **Robust Error Handling**       | Graceful management of edge cases with user-friendly messages. |

---

## 🛠️ Technology Stack

- **Backend & Modeling:** Python, Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow  
- **Web Framework:** Streamlit  
- **Explainability Tools:** SHAP, DiCE  
- **Visualization:** Matplotlib  
- **PDF Generation:** FPDF2  
- **Deployment:** Streamlit Community Cloud, GitHub  

---

## 📂 Project Structure

![image](https://github.com/user-attachments/assets/a9b2ee96-fb91-43b7-8ce4-f48c8e22d3b6)

---

## 🚀 Setup and Installation

1. **Prerequisites:**
   - Python 3.9+
   - Git

2. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/student_score_prediction.git
   cd student_score_prediction


---

## 🚀 Setup and Installation

### 🔹 Install Dependencies
pip install -r requirements.txt

---

## ⚙️ Execution Workflow

### 🔸 Generate & Preprocess Data
python src/data/generate_data.py

python src/data/preprocess_data.py

### 🔸 Train Models
python src/models/train_linear_regression.py

python src/models/train_random_forest.py

python src/models/train_xgboost.py

python src/models/train_neural_network.py

### 🔸 Evaluate Models
python src/evaluation/evaluate_models.py

### 🔸 Run the Dashboard
streamlit run streamlit_app/1_Individual_Predictor.py

✅ The application will automatically open in your default browser.

---

## 🤝 Contributing

Contributions are welcome! Fork the repository, make your changes, and submit a pull request. Suggestions for improvements and new features are greatly appreciated.

---

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

## 👨‍💻 Author

Created and maintained by **Pasam Tharun**.
