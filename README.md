# Automating an ML Pipeline with GitHub Actions For Kelper Object of Interest(KOI) dataset

## 1. Project overview

This project implements an end-to-end MLOps pipeline for a NASA Kepler astronomy classification task.

The pipeline:

1. Downloads and validates the Kepler dataset.
2. Creates a reproducible train/validation split.
3. Trains a `DummyClassifier` baseline.
4. Trains a Random Forest candidate model.
5. Evaluates both with F1-score.
6. Fails when the candidate does not beat the baseline by the required margin.
7. Runs application tests in the SAME GitHub Actions job.
8. Publishes a downloadable model package only after every required check passes.

NASA's Exoplanet Archive provides the Kepler Objects of Interest (KOI) cumulative table and a public API for querying it.

Source:
https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html

API documentation:
https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html

---

## 2. Dataset and prediction task

### Dataset

**NASA Exoplanet Archive — Kepler Objects of Interest (KOI) Cumulative Table**

The cumulative table gathers Kepler KOIs and their current dispositions. The dataset is filtered automatically through the public NASA Exoplanet Archive API.

The pipeline keeps only:

- `CANDIDATE`
- `FALSE POSITIVE`

Rows labelled `NOT DISPOSITIONED` are excluded because they do not provide a binary ground-truth label for this assignment.

### Target

`koi_disposition`

It is converted to:

- `CANDIDATE` → `1`
- `FALSE POSITIVE` → `0`

### Input features

The model uses these 11 numerical Kepler measurements:

- `koi_period`
- `koi_duration`
- `koi_depth`
- `koi_prad`
- `koi_teq`
- `koi_insol`
- `koi_steff`
- `koi_srad`
- `koi_smass`
- `koi_impact`
- `koi_model_snr`

These describe transit characteristics and properties of the host star/candidate.

The target/disposition itself and identifiers are never used as input features.

---

## 3. Why this evaluation metric?

The metric is **F1-score**.

This is a binary classification task, and both candidate and false-positive predictions matter. F1 combines precision and recall into one score:

`F1 = 2 * precision * recall / (precision + recall)`

F1 is appropriate because accuracy alone can hide poor performance on one class when the classes are not perfectly balanced.

Higher F1 is better.

---

## 4. Reproducibility

The pipeline uses:

- Python 3.11
- `random_state=42`
- stratified 80/20 train/validation split
- the same feature list and metric on every run
- preprocessing fitted as part of the training pipeline

Missing numerical values are handled by `SimpleImputer(strategy="median")` inside the scikit-learn Pipeline. Therefore the imputer is fitted only on training data and then applied to validation/test data.

---

## 5. Baseline and quality gate

### Baseline

The baseline is:

`DummyClassifier(strategy="most_frequent")`

The candidate is:

`RandomForestClassifier`

### Improvement margin

The fixed improvement margin is:

**0.05 F1 points**

The quality rule is:

`candidate F1 >= baseline F1 + 0.05`

The metric, split and margin remain unchanged during the failure/recovery demonstrations.

### Why 0.05?

A 0.05 F1 improvement is large enough to require a meaningful improvement rather than allowing a tiny random fluctuation to pass the gate.

If the margin were too low, a candidate could pass despite providing only a negligible improvement.

If the margin were too high, a useful model could be rejected even when it provides a substantial practical improvement.

The margin is expressed in F1-score units.

---

## 6. Project structure

```text
kepler-cloud-deployment/
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_prediction.py
├── artifacts/
│   └── .gitkeep
├── requirements.txt
├── README.md
└── .gitignore
```

The dataset is downloaded automatically by `src/data.py`; it does not need to be stored in Git.

---

## 7. Run locally

### Clone

```bash
git clone YOUR_REPOSITORY_URL
```

### Create environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download and validate

```bash
python -m src.data
```

This creates:

```text
build/kepler_koi.csv
```

### Train

```bash
python -m src.train --dataset build/kepler_koi.csv --mode normal
```

The model and metrics are written to:

```text
build/kepler_model.joblib
build/metrics.json
```

### Test

```bash
pytest -q
```

---

## 8. Application tests

The automated tests verify:

### Test 1 — saved model can be loaded

`test_saved_model_can_be_loaded`

The test confirms the trained `.joblib` file exists and can be loaded.

### Test 2 — valid input produces correct output

`test_valid_sample_output_shape_and_type`

The test verifies that one valid sample produces exactly one binary prediction.

### Test 3 — missing feature is rejected

`test_missing_required_feature_is_rejected`

The test removes one required feature and verifies that the application raises a clear `ValueError`.

A failed test returns a non-zero exit code, so GitHub Actions stops and does not reach artifact publication.

---

## 9. GitHub Actions pipeline

The workflow is:

```text
Push to main
     |
     v
Checkout
     |
     v
Setup Python
     |
     v
Install dependencies
     |
     v
Download + validate data
     |
     v
Train baseline + candidate
     |
     v
Quality gate
     |
     +---- FAIL --> non-zero exit --> NO ARTIFACT
     |
     v
Application tests
     |
     +---- FAIL --> non-zero exit --> NO ARTIFACT
     |
     v
Create model package
     |
     v
Upload GitHub Actions artifact
```

Training and testing are deliberately in the **same job**, so the tests can access the model produced during that workflow run.

Artifact upload has no `continue-on-error` and is not protected by `if: always()`. Therefore it is reached only when all previous required steps pass.

---

# 10. Failure and recovery demonstration

The assignment requires two separate failed workflow runs.

## Failure A — model quality failure

First make sure the normal version is restored.

Temporarily change the workflow training command from:

```bash
python -m src.train --dataset build/kepler_koi.csv --mode normal
```

to:

```bash
python -m src.train --dataset build/kepler_koi.csv --mode weak
```

The `weak` mode deliberately uses a very shallow Random Forest:

```python
RandomForestClassifier(
    n_estimators=10,
    max_depth=1,
    random_state=42,
    n_jobs=1
)
```

Do NOT change:

- F1 metric
- train/validation split
- random state
- 0.05 margin
- validation data

Commit and push:

```bash
git add .
git commit -m "Demonstrate model quality gate failure"
git push origin main
```

Expected result:

```text
Train model and enforce quality gate
FAILED
```

The training script exits with a non-zero status when:

```text
model F1 < baseline F1 + 0.05
```

Because the training step fails, the test step and artifact upload are not reached.

### Evidence to record


```text
Failure A run:
https://github.com/Ankush703-web/Kelper-Object-Interest-KOI-/actions/runs/36706476330/job/109857802399
```

---

# 11. Restore before Failure B

Change:

```bash
python -m src.train --dataset build/kepler_koi.csv --mode weak
```

back to:

```bash
python -m src.train --dataset build/kepler_koi.csv --mode normal
```

Commit/push this restoration before making the second failure.

---

# 12. Failure B — application test failure

Temporarily introduce an application bug in `src/predict.py`.

For example, change:

```python
return model.predict(X)
```

to:

```python
return model.predict(X).reshape(-1, 1)
```

This intentionally changes the prediction shape.

Commit and push:

```bash
git add .
git commit -m "Demonstrate application test failure"
git push origin main
```

The training and quality gate should pass, but:

```text
test_valid_sample_output_shape_and_type
```

will fail because the application now returns a two-dimensional array instead of one prediction per input row.

Therefore the GitHub Actions job stops at the application test step and does not publish the artifact.

### Evidence

```text
Failure B run:
https://github.com/Ankush703-web/Kelper-Object-Interest-KOI-/actions/runs/36708608084/job/109864737762
```

---

# 13. Final recovery

Restore:

```python
return model.predict(X)
```

Then commit and push:

```bash
git add .
git commit -m "Fix application and publish validated model"
git push origin main
```

The expected pipeline is:

```text
Validation       PASS
Training         PASS
Quality gate     PASS
Application tests PASS
Artifact upload  PASS
```

```text
Final successful run:
https://github.com/Ankush703-web/Kelper-Object-Interest-KOI-/actions/runs/36710607063
```

---

# 14. Successful artifact

The workflow produces:

```text
kepler-model-package-RUN_NUMBER.zip
```

The package contains:

```text
kepler_model.joblib
metrics.json
predict.py
requirements.txt
README.md
PACKAGE_INFO.txt
```

The model file contains the fitted preprocessing and Random Forest model as one scikit-learn Pipeline.

Download the successful artifact from:

```text
GitHub → Actions → successful workflow run → Artifacts
```

Also download a local copy because GitHub Actions artifacts have a limited retention period.


```text
Successful artifact:
kepler-model-package-<RUN_NUMBER>
```

---

# 15. What is demonstrated as Continuous Integration?

The project demonstrates CI because every push to `main` automatically:

- checks out the repository,
- creates a fresh Python environment,
- installs dependencies,
- downloads and validates the dataset,
- trains the model,
- evaluates the quality gate,
- runs automated application tests.

The workflow therefore verifies the complete ML/application state automatically rather than relying on the developer's local environment.

---

# 16. What is demonstrated as artifact delivery?

After all validation, model-quality and application tests pass, GitHub Actions creates a versioned ZIP model package and uploads it as a workflow artifact.

The package contains:

- trained model,
- fitted preprocessing,
- metrics report,
- prediction code,
- dependency file,
- package/run metadata.

Artifact publication is therefore downstream of all required checks.

---

# 17. MLOps maturity level

This implementation is best described as an **early/intermediate MLOps maturity implementation**.

It has automated:

- data acquisition,
- data validation,
- reproducible training,
- baseline comparison,
- model quality gating,
- application testing,
- model packaging,
- CI execution,
- artifact delivery.

However, it is not a full production MLOps platform.

It does not yet provide:

- model registry,
- production deployment,
- online monitoring,
- data-drift monitoring,
- model-performance monitoring after deployment,
- automated retraining based on drift,
- feature store,
- production serving infrastructure,
- approval/governance workflow.

The next maturity step would therefore be to introduce a model registry and deployment/monitoring pipeline, followed by automated monitoring and retraining.

---

# 18. Answers to the required README questions

### 1. Why does your evaluation metric suit your task?

F1-score balances precision and recall for the binary classification of Kepler objects as candidates or false positives. It is preferable to relying only on accuracy because performance on both classes matters.

### 2. Why did you choose this improvement margin?

The margin is 0.05 F1 points. It prevents a very small improvement over the baseline from being treated as sufficient. A smaller margin could allow negligible improvements, while a much larger margin could reject useful models.

### 3. What caused each failed run?

**Failure A:** The candidate model was deliberately weakened using a shallow Random Forest. The quality gate detected that its F1-score did not meet `baseline F1 + 0.05`, so the workflow stopped before artifact publication.

**Failure B:** The prediction application was deliberately changed to return an incorrect output shape. The application test detected the error and returned a non-zero exit code, so artifact publication did not occur.

The final run restored both the model configuration and prediction code.

### 4. Which parts demonstrate CI and artifact delivery?

The complete GitHub Actions workflow demonstrates CI by automatically installing dependencies, validating data, training, evaluating, and testing on every push to `main`.

Artifact delivery is demonstrated by creating and uploading the validated ZIP package only after all required checks pass.

### 5. Which MLOps maturity level best describes the implementation?

It is an early/intermediate automated MLOps implementation. It provides reproducible training, validation, quality gates, testing, CI and model artifact delivery. It does not yet provide production deployment, model registry, monitoring, drift detection or automated retraining.

---


## 19. References

NASA Exoplanet Archive — Kepler Mission:
https://exoplanetarchive.ipac.caltech.edu/docs/KeplerMission.html

NASA Exoplanet Archive — Kepler KOI documentation:
https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html

NASA Exoplanet Archive — API:
https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html

Scikit-learn Random Forest:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

Scikit-learn model persistence:
https://scikit-learn.org/stable/model_persistence.html
