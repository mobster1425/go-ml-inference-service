# Go ML Inference Service — Results

## Project Summary

This project trains a logistic regression churn model in Python, exports it as a portable JSON artifact, and serves predictions through a Go REST API.

## Dataset

- Train rows: 16000
- Validation rows: 4000
- Features: 13
- Target: Exited
- Churn rate: 0.2255

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.7782 |
| Precision | 0.5882 |
| Recall | 0.0554 |
| F1 | 0.1013 |
| ROC-AUC | 0.6934 |

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
