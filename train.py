import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
from joblib import dump
from utils.data_loader import load_data  # Add this import

def train_models():
    # Load and prepare data
    df = load_data('data/financial_data.csv')
    features = ['Year', 'Total Revenue', 'Net Income']
    targets = [col for col in df.columns if col not in features]
    
    # Convert to float32 for TensorFlow compatibility
    X = df[features].astype(np.float32)
    y = df[targets].astype(np.float32)
    
    # Machine Learning Model
    ml_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', MultiOutputRegressor(RandomForestRegressor(n_estimators=100)))
    ])
    ml_pipe.fit(X, y)
    
    # Deep Learning Model
    dl_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(targets))
    ])
    dl_model.compile(optimizer='adam', loss='mse')
    dl_model.fit(X, y, epochs=50, verbose=0)
    
    # Save models
    dump(ml_pipe, 'models/ml_model.pkl')
    dl_model.save('models/dl_model.keras', save_format='tf')

if __name__ == '__main__':
    train_models()
