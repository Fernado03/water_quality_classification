# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pathlib import Path

def load_dataset(relative_path='data/raw/water_quality.csv'):
    """Loads dataset from the given path."""
    data_path = Path(__file__).resolve().parents[1] / relative_path
    try:
        df = pd.read_csv(data_path)
        print("Dataset loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: '{relative_path}' not found.")
        return pd.DataFrame()

def preprocess_data(df, features=['EC', 'Cl', 'TDS', 'Na'], target='Water Quality Classification'):
    """
    Encodes target, scales features, and splits dataset.
    Returns:
        - X_train_scaled, X_test_scaled
        - y_train, y_test
        - scaler (fitted)
        - class_names (list of label classes)
    """
    if df.empty:
        raise ValueError("DataFrame is empty. Cannot preprocess.")
    
    # Select features and target
    X = df[features]
    y = df[target]
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to DataFrames (optional but useful for inspection)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)

    print("Data preprocessing complete.")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, class_names

if __name__ == "__main__":
    df = load_dataset()
    X_train, X_test, y_train, y_test, scaler, class_names = preprocess_data(df)
    print("Preprocessing ran successfully.")
