import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_linear_regression(data_path='data/processed/train_dataset.csv', model_path='models/linear_regression/model.pkl'):
    train_df = pd.read_csv(data_path)
    X_train = train_df.drop('final_exam_score', axis=1)
    y_train = train_df['final_exam_score']

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Linear Regression model trained.")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_linear_regression()