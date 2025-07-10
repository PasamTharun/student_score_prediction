import pandas as pd
import joblib
import lime
import lime.lime_tabular
import os

def generate_lime_explanation(model_path='models/xgboost/model.pkl', data_path='data/processed/train_dataset.csv', instance_idx=0):
    model = joblib.load(model_path)
    train_df = pd.read_csv(data_path)
    X_train = train_df.drop('final_exam_score', axis=1)
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['final_exam_score'],
        mode='regression'
    )
    
    # Explain a single instance
    instance_to_explain = X_train.iloc[instance_idx]
    explanation = explainer.explain_instance(
        data_row=instance_to_explain.values,
        predict_fn=model.predict
    )
    
    # Create visualization directory
    viz_dir = 'visualization'
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
        
    explanation.save_to_file(f'{viz_dir}/lime_instance_{instance_idx}_explanation.html')
    print(f"LIME explanation for instance {instance_idx} saved to 'visualization/' directory.")

if __name__ == "__main__":
    generate_lime_explanation()