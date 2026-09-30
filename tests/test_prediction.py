"""Application tests required by Lab 4."""

from pathlib import Path
import pandas as pd
import pytest
import joblib

from src.data import FEATURES
from src.predict import predict

MODEL = Path("build/kepler_model.joblib")

def sample():
    # Missing values are intentionally allowed because the pipeline imputes them.
    return {feature: 0.0 for feature in FEATURES}

def test_saved_model_can_be_loaded():
    assert MODEL.exists(), f"Model not found at {MODEL}"
    loaded = joblib.load(MODEL)
    assert hasattr(loaded, "predict")

def test_valid_sample_output_shape_and_type():
    output = predict(sample())
    assert len(output) == 1
    assert output[0] in (0, 1)
    assert type(output[0]).__name__ in {"int", "int64", "int32"}

def test_missing_required_feature_is_rejected():
    bad = sample()
    bad.pop(FEATURES[0])

    with pytest.raises(ValueError, match="Missing required features"):
        predict(bad)
