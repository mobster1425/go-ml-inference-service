package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/mobster1425/go-ml-inference-service/service/internal/metrics"
	"github.com/mobster1425/go-ml-inference-service/service/internal/model"
)

func TestHealthReturnsOK(t *testing.T) {
	server := testServer(t)
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/health", nil)

	server.Routes().ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status=%d", response.Code)
	}
}

func TestModelInfoIncludesDatasetSource(t *testing.T) {
	server := testServer(t)
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/model-info", nil)

	server.Routes().ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var result struct {
		DatasetSource string `json:"dataset_source"`
	}
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if result.DatasetSource == "" {
		t.Fatal("expected dataset_source in model-info response")
	}
}

func TestPredictReturnsOKForValidInput(t *testing.T) {
	server := testServer(t)
	body := encodeJSON(t, validAPIInput())
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/predict", bytes.NewReader(body))

	server.Routes().ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var result model.PredictionResult
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if result.Probability < 0 || result.Probability > 1 {
		t.Fatalf("probability=%f outside [0,1]", result.Probability)
	}
}

func TestPredictReturnsErrorForInvalidInput(t *testing.T) {
	server := testServer(t)
	input := validAPIInput()
	input.CreditScore = 100
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/predict", bytes.NewReader(encodeJSON(t, input)))

	server.Routes().ServeHTTP(response, request)

	if response.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestBatchPredictReturnsCorrectCount(t *testing.T) {
	server := testServer(t)
	body := encodeJSON(t, map[string][]model.CustomerInput{
		"records": {validAPIInput(), highRiskAPIInput()},
	})
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/batch-predict", bytes.NewReader(body))

	server.Routes().ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var result struct {
		Count int `json:"count"`
	}
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if result.Count != 2 {
		t.Fatalf("count=%d", result.Count)
	}
}

func testServer(t *testing.T) *Server {
	t.Helper()
	path := filepath.Join("..", "..", "..", "model_artifacts", "churn_logistic_model.json")
	artifact, err := model.LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel(%q): %v", path, err)
	}
	return NewServer(artifact, metrics.NewStore())
}

func encodeJSON(t *testing.T, value interface{}) []byte {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("json marshal: %v", err)
	}
	return data
}

func validAPIInput() model.CustomerInput {
	return model.CustomerInput{
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

func highRiskAPIInput() model.CustomerInput {
	return model.CustomerInput{
		CreditScore:     515,
		Geography:       "Germany",
		Gender:          "Female",
		Age:             66,
		Tenure:          1,
		Balance:         210000,
		NumOfProducts:   1,
		HasCrCard:       1,
		IsActiveMember:  0,
		EstimatedSalary: 132000,
	}
}
