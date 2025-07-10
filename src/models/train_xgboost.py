import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import joblib

def train_xgboost(data_path='data/processed/train_dataset.csv', model_path='models/xgboost/model.pkl'):
    train_df = pd.read_csv(data_path)
    X_train = train_df.drop('final_exam_score', axis=1)
    y_train = train_df['final_exam_score']

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 1.0]
    }

    grid_search = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
                               param_grid=param_grid,
                               cv=3,
                               n_jobs=-1,
                               verbose=2)

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("XGBoost model trained with best parameters:", grid_search.best_params_)
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_xgboost()