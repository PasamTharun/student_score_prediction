import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os # Import the os module

def preprocess_data(input_path='data/raw/dataset.csv', output_dir='data/processed/'):
    """Loads, preprocesses, and splits the data, saving processed files."""
    
    # --- Ensure the output directory exists ---
    os.makedirs(output_dir, exist_ok=True) # This line creates the directory if it's missing
    
    df = pd.read_csv(input_path)

    # Define features and target
    X = df.drop('final_exam_score', axis=1)
    y = df['final_exam_score']

    # Define feature types
    numeric_features = ['hours_studied', 'previous_score', 'attendance_percentage']
    categorical_features = ['stress_level', 'confidence_level']

    # Create preprocessing pipelines
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Split data before fitting the preprocessor
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the preprocessor on the training data and transform both sets
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the fitted preprocessor
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.pkl'))

    # Get feature names after transformation
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_feature_names)

    # Convert processed arrays back to DataFrames
    X_train_df = pd.DataFrame(X_train_processed, columns=all_feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=all_feature_names)

    # Concatenate features and target for saving
    train_df = pd.concat([X_train_df, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_df, y_test.reset_index(drop=True)], axis=1)

    # Save the processed data
    train_df.to_csv(os.path.join(output_dir, 'train_dataset.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_dataset.csv'), index=False)
    
    print(f"Data preprocessing complete. Processed files saved to '{output_dir}'")

if __name__ == "__main__":
    preprocess_data()
