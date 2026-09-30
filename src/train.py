"""Train baseline and candidate models and enforce the quality gate."""

from pathlib import Path
import argparse
import json
import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import FEATURES, TARGET, load_and_validate, download_dataset

RANDOM_STATE = 42
MARGIN = 0.05
ARTIFACT_DIR = Path("build")

def make_candidate(mode: str = "normal"):
    if mode == "weak":
        # Used ONLY for Failure A demonstration.
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=10,
                max_depth=1,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )),
        ])

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )),
    ])

def train(dataset_path: str, mode: str = "normal"):
    df = load_and_validate(dataset_path)
    X = df[FEATURES]
    y = df["target"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_score = f1_score(y_val, baseline.predict(X_val), zero_division=0)

    model = make_candidate(mode)
    model.fit(X_train, y_train)
    model_score = f1_score(y_val, model.predict(X_val), zero_division=0)

    required_score = baseline_score + MARGIN
    passed = model_score >= required_score

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "kepler_model.joblib")

    metrics = {
        "dataset": "NASA Exoplanet Archive Kepler KOI Cumulative Table",
        "target": TARGET,
        "task": "Binary classification: CANDIDATE vs FALSE POSITIVE",
        "metric": "F1",
        "higher_is_better": True,
        "random_state": RANDOM_STATE,
        "validation_size": 0.20,
        "baseline_score": round(float(baseline_score), 6),
        "model_score": round(float(model_score), 6),
        "margin": MARGIN,
        "required_score": round(float(required_score), 6),
        "gate_passed": bool(passed),
        "model_mode": mode,
        "features": FEATURES,
    }
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(json.dumps(metrics, indent=2))

    if not passed:
        raise SystemExit(
            f"QUALITY GATE FAILED: model F1={model_score:.4f}, "
            f"required F1={required_score:.4f}"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="build/kepler_koi.csv")
    parser.add_argument("--mode", choices=["normal", "weak"], default="normal")
    args = parser.parse_args()

    if not Path(args.dataset).exists():
        download_dataset(args.dataset)

    train(args.dataset, args.mode)
