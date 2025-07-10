import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os # Import the os module

def train_random_forest(data_path='data/processed/train_dataset.csv', model_path='models/random_forest/model.pkl'):
    train_df = pd.read_csv(data_path)
    X_train = train_df.drop('final_exam_score', axis=1)
    y_train = train_df['final_exam_score']

    # Hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }

    # Grid search with cross-validation
    grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                               param_grid=param_grid,
                               cv=3,
                               n_jobs=-1,
                               verbose=2)
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("Random Forest model trained with best parameters:", grid_search.best_params_)
    
    # --- FIX: Ensure the model directory exists before saving ---
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True) # Create directory if it doesn't exist
    
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_random_forest()
