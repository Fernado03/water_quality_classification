# predict.py

import pandas as pd
import pickle
from pathlib import Path

# Define paths
base_dir = Path(__file__).resolve().parent.parent
model_path = base_dir / 'models' / 'final_gradient_boosting_model.pkl'
scaler_path = base_dir / 'models' / 'min_max_scaler.pkl'
# new_data_path = base_dir / 'data' / 'new' / 'new_sample.csv'  # Uncomment when file is ready

# Load the trained scaler
try:
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded.")
except FileNotFoundError:
    print(f"Error: Scaler file not found at '{scaler_path}'")
    exit()

# Load the trained model
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded.")
except FileNotFoundError:
    print(f"Error: Model file not found at '{model_path}'")
    exit()

# Uncomment and use when you have the input data ready
"""
# Load new data for prediction
try:
    new_data = pd.read_csv(new_data_path)
except FileNotFoundError:
    print(f"Error: '{new_data_path}' not found.")
    exit()

# Apply the same scaling used during training
new_data_scaled = scaler.transform(new_data)

# Predict
predictions = model.predict(new_data_scaled)

# Output results
print("Predictions:", predictions)
"""
