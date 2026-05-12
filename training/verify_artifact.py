"""Verify the exported model artifact can produce valid probabilities."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "model_artifacts" / "churn_logistic_model.json"
TEST_PATH = ROOT / "training" / "data" / "test.csv"


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def row_value(row: pd.Series, feature: str) -> float:
    return float(row[feature])


def preprocess(row: pd.Series, artifact: dict) -> list[float]:
    values: dict[str, float] = {}
    means = artifact["preprocessing"]["means"]
    stds = artifact["preprocessing"]["stds"]

    for feature in artifact["numeric_features"]:
        values[feature] = (row_value(row, feature) - float(means[feature])) / float(stds[feature])

    for feature, categories in artifact["categorical_features"].items():
        actual = str(row[feature])
        for category in categories:
            values[f"{feature}_{category}"] = 1.0 if actual == category else 0.0

    return [values[feature] for feature in artifact["feature_order"]]


def predict_probability(row: pd.Series, artifact: dict) -> float:
    features = preprocess(row, artifact)
    coefficients = artifact["coefficients"]
    if len(features) != len(coefficients):
        raise ValueError(f"Feature length {len(features)} does not match coefficients {len(coefficients)}")

    score = float(artifact["intercept"]) + sum(c * x for c, x in zip(coefficients, features))
    return min(1.0, max(0.0, sigmoid(score)))


def main() -> None:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Missing artifact: {ARTIFACT_PATH}. Run training/train_export.py first.")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test data: {TEST_PATH}. Run training/generate_data.py first.")

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    test_df = pd.read_csv(TEST_PATH).head(5)

    print("Example predictions from exported artifact:")
    for _, row in test_df.iterrows():
        probability = predict_probability(row, artifact)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability outside [0, 1]: {probability}")
        prediction = int(probability >= float(artifact["threshold"]))
        print(
            f"CustomerId={int(row['CustomerId'])} probability={probability:.6f} "
            f"prediction={prediction} actual={int(row['Exited'])}"
        )

    print("Artifact verification passed.")


if __name__ == "__main__":
    main()
