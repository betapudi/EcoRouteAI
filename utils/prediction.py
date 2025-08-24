import joblib
import numpy as np
import pandas as pd

def load_model(path):
    try:
        saved = joblib.load(path)
        return saved["model"], saved["features"]
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def predict_eco_scores(model, features, df):
    # Check if all required features are present
    missing_features = set(features) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")
    
    # Ensure correct data types
    for feature in features:
        if feature in df.columns:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
    
    # Handle missing values
    df_filled = df[features].fillna(df[features].mean())
    
    # Make predictions
    df["pred_eco_score"] = model.predict(df_filled[features])
    
    # Ensure scores are within reasonable bounds
    # df["pred_eco_score"] = np.clip(df["pred_eco_score"], 0, 100)
    
    return df