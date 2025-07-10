import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_records=2000):
    """Generates a realistic synthetic dataset, ensuring some toppers are included."""
    output_dir = 'data/raw'
    os.makedirs(output_dir, exist_ok=True)

    # --- Generate the main bulk of student data randomly ---
    data = {
        'hours_studied': np.random.uniform(1, 20, num_records - 5),
        'previous_score': np.random.uniform(40, 100, num_records - 5),
        'attendance_percentage': np.random.uniform(60, 100, num_records - 5),
        'stress_level': np.random.choice(['Low', 'Medium', 'High'], num_records - 5, p=[0.4, 0.4, 0.2]),
        'confidence_level': np.random.choice(['Low', 'Medium', 'High'], num_records - 5, p=[0.3, 0.5, 0.2])
    }
    main_df = pd.DataFrame(data)

    # --- FIX: Intentionally create a few "topper" students ---
    topper_data = {
        'hours_studied': np.random.uniform(18, 25, 5),          # High study hours
        'previous_score': np.random.uniform(95, 100, 5),         # Excellent previous scores
        'attendance_percentage': np.random.uniform(98, 100, 5), # Perfect attendance
        'stress_level': ['Low'] * 5,                            # Low stress
        'confidence_level': ['High'] * 5                        # High confidence
    }
    topper_df = pd.DataFrame(topper_data)

    # Combine the two groups of students
    df = pd.concat([main_df, topper_df], ignore_index=True)

    # Create a target variable with some noise
    score = (
        20 +
        (df['hours_studied'] * 2.5) +
        (df['previous_score'] * 0.4) +
        (df['attendance_percentage'] * 0.15) +
        df['stress_level'].replace({'Low': 5, 'Medium': 0, 'High': -5}) +
        df['confidence_level'].replace({'Low': -3, 'Medium': 0, 'High': 3}) +
        np.random.normal(0, 3, num_records) # Reduced noise for more predictable scores
    )
    df['final_exam_score'] = np.clip(score, 0, 100)

    output_path = os.path.join(output_dir, 'dataset.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Synthetic dataset with toppers generated and saved to '{output_path}'")

if __name__ == "__main__":
    generate_synthetic_data()
