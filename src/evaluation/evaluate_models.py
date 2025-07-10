import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate_models(
    data_path='data/processed/test_dataset.csv',
    report_path='reports/model_performance.md'
):
    """
    Evaluates all trained models, calculates metrics in both raw and
    percentage formats, and saves reports and plots.
    """
    test_df = pd.read_csv(data_path)
    X_test = test_df.drop('final_exam_score', axis=1)
    y_test = test_df['final_exam_score']

    models_to_evaluate = {
        'Linear Regression': 'models/linear_regression/model.pkl',
        'Random Forest': 'models/random_forest/model.pkl',
        'XGBoost': 'models/xgboost/model.pkl',
        'Neural Network': 'models/neural_network/model.h5'
    }

    results = []
    
    viz_dir = 'visualization'
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    for name, path in models_to_evaluate.items():
        if path.endswith('.h5'):
            model = tf.keras.models.load_model(path)
            y_pred = model.predict(X_test).flatten()
        else:
            model = joblib.load(path)
            y_pred = model.predict(X_test)

        # Calculate standard metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # NEW: Calculate percentage-based metrics
        mse_perc = (mse / (100**2)) * 100
        rmse_perc = (rmse / 100) * 100
        mae_perc = (mae / 100) * 100
        r2_perc = r2 * 100

        results.append({
            'Model': name, 
            'MSE (%)': f"{mse_perc:.2f}",
            'RMSE (%)': f"{rmse_perc:.2f}", 
            'MAE (%)': f"{mae_perc:.2f}", 
            'R2 Score (%)': f"{r2_perc:.2f}"
        })

        # Plotting remains the same...
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', lw=2)
        plt.xlabel('Actual Scores')
        plt.ylabel('Predicted Scores')
        plt.title(f'{name} - Predictions vs. Actuals')
        plt.grid(True)
        plt.savefig(f'{viz_dir}/{name.lower().replace(" ", "_")}_predictions.png')
        plt.close()

    results_df = pd.DataFrame(results)
    print("Model Evaluation Results (in Percentage Format):")
    print(results_df)

    # Save the new percentage-based report
    with open(report_path, 'w') as f:
        f.write("# Model Performance Report (Percentage Format)\n\n")
        f.write(results_df.to_markdown(index=False))

    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    evaluate_models()
