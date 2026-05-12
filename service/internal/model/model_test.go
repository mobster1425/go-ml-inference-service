package model

import (
	"path/filepath"
	"testing"
)

func TestSigmoidOutputRange(t *testing.T) {
	for _, value := range []float64{-1000, -5, 0, 5, 1000} {
		probability := Sigmoid(value)
		if probability < 0 || probability > 1 {
			t.Fatalf("Sigmoid(%f)=%f outside [0,1]", value, probability)
		}
	}
}

func TestModelArtifactLoads(t *testing.T) {
	artifact := loadTestModel(t)
	if artifact.ModelName != "bank_churn_logistic_regression" {
		t.Fatalf("unexpected model name %q", artifact.ModelName)
	}
}

func TestPreprocessingOutputLengthEqualsCoefficients(t *testing.T) {
	artifact := loadTestModel(t)
	features, err := artifact.Preprocess(validCustomerInput())
	if err != nil {
		t.Fatalf("Preprocess returned error: %v", err)
	}
	if len(features) != len(artifact.Coefficients) {
		t.Fatalf("features=%d coefficients=%d", len(features), len(artifact.Coefficients))
	}
}

func TestPredictionProbabilityIsBetweenZeroAndOne(t *testing.T) {
	artifact := loadTestModel(t)
	probability, _, err := artifact.Predict(validCustomerInput())
	if err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}
	if probability < 0 || probability > 1 {
		t.Fatalf("probability=%f outside [0,1]", probability)
	}
}

func loadTestModel(t *testing.T) *ModelArtifact {
	t.Helper()
	path := filepath.Join("..", "..", "..", "model_artifacts", "churn_logistic_model.json")
	artifact, err := LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel(%q) returned error: %v", path, err)
	}
	return artifact
}

func validCustomerInput() CustomerInput {
	return CustomerInput{
		CreditScore:     650,
		Geography:       "France",
		Gender:          "Female",
		Age:             42,
		Tenure:          5,
		Balance:         125000,
		NumOfProducts:   2,
		HasCrCard:       1,
		IsActiveMember:  1,
		EstimatedSalary: 78000,
	}
}
