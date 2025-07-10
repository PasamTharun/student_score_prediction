import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def train_neural_network(data_path='data/processed/train_dataset.csv', model_path='models/neural_network/model.h5'):
    train_df = pd.read_csv(data_path)
    X_train = train_df.drop('final_exam_score', axis=1)
    y_train = train_df['final_exam_score']

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

    print("Neural Network model trained.")
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_neural_network()