"""Train logistic regression and export a portable JSON model artifact."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "training" / "data" / "train.csv"
ARTIFACT_PATH = ROOT / "model_artifacts" / "churn_logistic_model.json"
RESULTS_PATH = ROOT / "results.md"

TARGET = "Exited"
NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
CATEGORICAL_FEATURES = {
    "Geography": ["France", "Germany", "Spain"],
    "Gender": ["Female", "Male"],
}
FEATURE_ORDER = NUMERIC_FEATURES + [
    f"{feature}_{category}"
    for feature, categories in CATEGORICAL_FEATURES.items()
    for category in categories
]


def preprocess_frame(df: pd.DataFrame, means: dict[str, float], stds: dict[str, float]) -> np.ndarray:
    """Apply the same numeric scaling and fixed one-hot encoding exported for Go."""
    parts: list[np.ndarray] = []
    for feature in NUMERIC_FEATURES:
        parts.append(((df[feature].astype(float) - means[feature]) / stds[feature]).to_numpy())

    for feature, categories in CATEGORICAL_FEATURES.items():
        for category in categories:
            parts.append((df[feature] == category).astype(float).to_numpy())

    return np.vstack(parts).T


def write_results(
    train_rows: int,
    validation_rows: int,
    churn_rate: float,
    metrics: dict[str, float],
) -> None:
    content = f"""# Go ML Inference Service — Results

## Project Summary

This project trains a logistic regression churn model in Python, exports it as a portable JSON artifact, and serves predictions through a Go REST API.

## Dataset

- Train rows: {train_rows}
- Validation rows: {validation_rows}
- Features: {len(FEATURE_ORDER)}
- Target: {TARGET}
- Churn rate: {churn_rate:.4f}

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | {metrics["accuracy"]:.4f} |
| Precision | {metrics["precision"]:.4f} |
| Recall | {metrics["recall"]:.4f} |
| F1 | {metrics["f1"]:.4f} |
| ROC-AUC | {metrics["roc_auc"]:.4f} |

## Model Artifact

The trained model is exported to:

`model_artifacts/churn_logistic_model.json`

## Inference Service

The Go service loads the JSON model artifact and implements preprocessing and logistic regression inference without calling Python.

## MLE Skills Demonstrated

- Model training
- Model artifact export
- Cross-language inference
- REST API model serving
- Input validation
- Batch prediction
- Latency tracking
- Dockerized deployment
- Go unit testing
"""
    RESULTS_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing training data: {TRAIN_PATH}. Run training/generate_data.py first.")

    df = pd.read_csv(TRAIN_PATH)
    missing = [column for column in NUMERIC_FEATURES + list(CATEGORICAL_FEATURES) + [TARGET] if column not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")

    train_df, validation_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET],
    )

    means = {feature: float(train_df[feature].mean()) for feature in NUMERIC_FEATURES}
    stds = {feature: float(train_df[feature].std(ddof=0)) for feature in NUMERIC_FEATURES}
    stds = {feature: value if value > 0 else 1.0 for feature, value in stds.items()}

    x_train = preprocess_frame(train_df, means, stds)
    y_train = train_df[TARGET].astype(int).to_numpy()
    x_validation = preprocess_frame(validation_df, means, stds)
    y_validation = validation_df[TARGET].astype(int).to_numpy()

    classifier = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    classifier.fit(x_train, y_train)

    probabilities = classifier.predict_proba(x_validation)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_validation, predictions)),
        "precision": float(precision_score(y_validation, predictions, zero_division=0)),
        "recall": float(recall_score(y_validation, predictions, zero_division=0)),
        "f1": float(f1_score(y_validation, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_validation, probabilities)),
    }

    artifact = {
        "model_name": "bank_churn_logistic_regression",
        "model_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "threshold": 0.5,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_order": FEATURE_ORDER,
        "preprocessing": {
            "means": means,
            "stds": stds,
        },
        "coefficients": [float(value) for value in classifier.coef_[0]],
        "intercept": float(classifier.intercept_[0]),
        "metrics": metrics,
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    write_results(len(train_df), len(validation_df), float(df[TARGET].mean()), metrics)

    print(f"Wrote model artifact to {ARTIFACT_PATH}")
    print(f"Wrote results to {RESULTS_PATH}")
    for name, score in metrics.items():
        print(f"{name}: {score:.4f}")


if __name__ == "__main__":
    main()
