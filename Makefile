.PHONY: setup-python generate-data train verify-artifact test-go run-service docker-build docker-run all

setup-python:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r training/requirements.txt

generate-data:
	python training/generate_data.py

train:
	python training/train_export.py

verify-artifact:
	python training/verify_artifact.py

test-go:
	cd service && go test ./...

run-service:
	cd service && MODEL_PATH=../model_artifacts/churn_logistic_model.json go run ./cmd/server

docker-build:
	docker build -t go-ml-inference-service .

docker-run:
	docker run --rm -p 8080:8080 go-ml-inference-service

all: generate-data train verify-artifact test-go
