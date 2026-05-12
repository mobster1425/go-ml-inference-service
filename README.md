# Go ML Inference Service

This is a production-style Go REST API for serving churn predictions from a logistic regression model trained in Python. The trained model is exported as a portable JSON artifact, and the Go service handles preprocessing and inference without calling Python at request time.

## Why I Built This

Model work does not stop at training. A useful ML service needs a reproducible artifact, preserved preprocessing logic, reliable prediction endpoints, input validation, latency tracking, and a deployment path. This project focuses on that end-to-end serving workflow.

## Architecture

```text
Python training -> JSON model artifact -> Go inference service -> REST predictions
```

## Dataset

The training pipeline prefers real Bank Churn data whenever `training/data/train.csv` already exists. In that case, synthetic generation is skipped and the existing file is never overwritten.

Synthetic data is generated only as a fallback when `training/data/train.csv` is missing. The exported JSON artifact includes `dataset_source`, so it is clear whether the current model was trained on real Kaggle-style Bank Churn data or the synthetic fallback dataset.

Latest artifact dataset source: `real_kaggle_bank_churn`

## Tech Stack

- Python
- scikit-learn
- pandas
- Go
- net/http
- Docker

## Project Structure

```text
training/          Python data generation, training, and artifact verification
model_artifacts/   Portable JSON model artifact
service/           Go inference API and tests
examples/          Example prediction request payloads
Dockerfile         Multi-stage container build
results.md         Model evaluation results
```

## How To Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r training/requirements.txt
python training/generate_data.py
python training/train_export.py
python training/verify_artifact.py
cd service
go test ./...
MODEL_PATH=../model_artifacts/churn_logistic_model.json go run ./cmd/server
```

Latest verification: Go unit tests pass with `go test ./...`.

From another terminal at the project root:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/model-info
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  --data @examples/predict_request.json
curl -X POST http://localhost:8080/batch-predict \
  -H "Content-Type: application/json" \
  --data @examples/batch_predict_request.json
curl http://localhost:8080/metrics
```

## Endpoints

- `GET /health`
- `GET /model-info`
- `GET /metrics`
- `POST /predict`
- `POST /batch-predict`

## Outputs

- `model_artifacts/churn_logistic_model.json`
- `training/data/train.csv`
- `training/data/test.csv`
- `results.md`

## Docker

```bash
docker build -t go-ml-inference-service .
docker run --rm -p 8080:8080 go-ml-inference-service
```

## MLE Skills Demonstrated

- Python model training
- Portable model artifact export
- Cross-language inference
- Go REST API model serving
- Input validation
- Batch prediction
- Latency metrics
- Docker deployment
- Unit testing
