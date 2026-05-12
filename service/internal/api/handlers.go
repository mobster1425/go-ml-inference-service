package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/mobster1425/go-ml-inference-service/service/internal/metrics"
	"github.com/mobster1425/go-ml-inference-service/service/internal/model"
	"github.com/mobster1425/go-ml-inference-service/service/internal/validation"
)

type Server struct {
	model   *model.ModelArtifact
	metrics *metrics.Store
}

type batchPredictRequest struct {
	Records []model.CustomerInput `json:"records"`
}

type batchPredictResponse struct {
	Predictions []model.PredictionResult `json:"predictions"`
	Count       int                      `json:"count"`
	LatencyMS   float64                  `json:"latency_ms"`
}

type errorResponse struct {
	Error string `json:"error"`
}

func NewServer(artifact *model.ModelArtifact, store *metrics.Store) *Server {
	return &Server{model: artifact, metrics: store}
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.health)
	mux.HandleFunc("GET /model-info", s.modelInfo)
	mux.HandleFunc("GET /metrics", s.metricsSnapshot)
	mux.HandleFunc("POST /predict", s.predict)
	mux.HandleFunc("POST /batch-predict", s.batchPredict)
	return mux
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{
		"status":  "ok",
		"service": "go-ml-inference-service",
	})
}

func (s *Server) modelInfo(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"model_name":    s.model.ModelName,
		"model_version": s.model.ModelVersion,
		"target":        s.model.Target,
		"features":      s.model.FeatureOrder,
		"metrics":       s.model.Metrics,
	})
}

func (s *Server) metricsSnapshot(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.metrics.Snapshot())
}

func (s *Server) predict(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	var input model.CustomerInput
	if err := decodeJSON(r, &input); err != nil {
		s.recordPredictionFailure(start)
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := validation.ValidateCustomerInput(input); err != nil {
		s.recordPredictionFailure(start)
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	probability, prediction, err := s.model.Predict(input)
	latencyMS := elapsedMS(start)
	if err != nil {
		s.metrics.RecordPrediction(latencyMS, true)
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	s.metrics.RecordPrediction(latencyMS, false)
	writeJSON(w, http.StatusOK, model.PredictionResult{
		Probability:  probability,
		Prediction:   prediction,
		ModelName:    s.model.ModelName,
		ModelVersion: s.model.ModelVersion,
		LatencyMS:    latencyMS,
	})
}

func (s *Server) batchPredict(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	var request batchPredictRequest
	if err := decodeJSON(r, &request); err != nil {
		s.recordBatchFailure(start)
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := validation.ValidateBatch(request.Records); err != nil {
		s.recordBatchFailure(start)
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	predictions := make([]model.PredictionResult, 0, len(request.Records))
	for _, input := range request.Records {
		itemStart := time.Now()
		probability, prediction, err := s.model.Predict(input)
		itemLatencyMS := elapsedMS(itemStart)
		if err != nil {
			latencyMS := elapsedMS(start)
			s.metrics.RecordBatchPrediction(latencyMS, true)
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		predictions = append(predictions, model.PredictionResult{
			Probability:  probability,
			Prediction:   prediction,
			ModelName:    s.model.ModelName,
			ModelVersion: s.model.ModelVersion,
			LatencyMS:    itemLatencyMS,
		})
	}

	latencyMS := elapsedMS(start)
	s.metrics.RecordBatchPrediction(latencyMS, false)
	writeJSON(w, http.StatusOK, batchPredictResponse{
		Predictions: predictions,
		Count:       len(predictions),
		LatencyMS:   latencyMS,
	})
}

func decodeJSON(r *http.Request, target interface{}) error {
	defer r.Body.Close()
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return fmt.Errorf("invalid JSON request body: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return errors.New("invalid JSON request body: multiple JSON values are not allowed")
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, errorResponse{Error: message})
}

func elapsedMS(start time.Time) float64 {
	return float64(time.Since(start).Microseconds()) / 1000.0
}

func (s *Server) recordPredictionFailure(start time.Time) {
	s.metrics.RecordPrediction(elapsedMS(start), true)
}

func (s *Server) recordBatchFailure(start time.Time) {
	s.metrics.RecordBatchPrediction(elapsedMS(start), true)
}
