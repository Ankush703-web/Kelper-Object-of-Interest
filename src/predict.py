"""Prediction application for the trained Kepler model."""

from pathlib import Path
import joblib
import pandas as pd

from src.data import FEATURES

MODEL_PATH = Path("build/kepler_model.joblib")

def predict(sample: dict | pd.DataFrame, model_path: str | Path = MODEL_PATH):
    """Return integer predictions for one or more valid observations."""
    if isinstance(sample, dict):
        sample = pd.DataFrame([sample])
    elif not isinstance(sample, pd.DataFrame):
        raise TypeError("sample must be a dict or pandas DataFrame")

    missing = sorted(set(FEATURES) - set(sample.columns))
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = sample[FEATURES]
    model = joblib.load(model_path)
    return model.predict(X)

if __name__ == "__main__":
    example = {feature: 0.0 for feature in FEATURES}
    print(predict(example).tolist())
