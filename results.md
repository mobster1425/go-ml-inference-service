# Go ML Inference Service — Results

## Project Summary

This project trains a logistic regression churn model in Python, exports it as a portable JSON artifact, and serves predictions through a Go REST API.

## Dataset

- Dataset source: real_kaggle_bank_churn
- Train rows: 132027
- Validation rows: 33007
- Features: 13
- Target: Exited
- Churn rate: 0.2116

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.8334 |
| Precision | 0.6933 |
| Recall | 0.3813 |
| F1 | 0.4920 |
| ROC-AUC | 0.8145 |

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
