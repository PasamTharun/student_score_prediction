import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

def generate_shap_explanations(model_path='models/xgboost/model.pkl', data_path='data/processed/train_dataset.csv'):
    model = joblib.load(model_path)
    train_df = pd.read_csv(data_path)
    X_train = train_df.drop('final_exam_score', axis=1)

    # Use TreeExplainer for tree-based models like XGBoost and Random Forest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Create visualization directory
    viz_dir = 'visualization'
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig(f'{viz_dir}/shap_summary_plot.png', bbox_inches='tight')
    plt.close()
    
    # Bar plot for top features
    plt.figure()
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.savefig(f'{viz_dir}/shap_feature_importance.png', bbox_inches='tight')
    plt.close()
    
    print("SHAP explanation plots saved to 'visualization/' directory.")

if __name__ == "__main__":
    generate_shap_explanations()