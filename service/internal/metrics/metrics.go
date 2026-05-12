package metrics

import (
	"sync"
	"time"
)

type Store struct {
	mu                      sync.Mutex
	totalRequests           uint64
	predictionRequests      uint64
	batchPredictionRequests uint64
	failedRequests          uint64
	totalLatencyMS          float64
	latencySamples          uint64
	lastPredictionTimestamp string
}

func NewStore() *Store {
	return &Store{}
}

func (s *Store) RecordPrediction(latencyMS float64, failed bool) {
	s.RecordRequest("prediction", latencyMS, failed)
}

func (s *Store) RecordBatchPrediction(latencyMS float64, failed bool) {
	s.RecordRequest("batch_prediction", latencyMS, failed)
}

func (s *Store) RecordRequest(kind string, latencyMS float64, failed bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.totalRequests++
	switch kind {
	case "prediction":
		s.predictionRequests++
	case "batch_prediction":
		s.batchPredictionRequests++
	}
	if failed {
		s.failedRequests++
	}
	if latencyMS >= 0 {
		s.totalLatencyMS += latencyMS
		s.latencySamples++
	}
	if !failed && (kind == "prediction" || kind == "batch_prediction") {
		s.lastPredictionTimestamp = time.Now().UTC().Format(time.RFC3339)
	}
}

func (s *Store) Snapshot() map[string]interface{} {
	s.mu.Lock()
	defer s.mu.Unlock()

	averageLatency := 0.0
	if s.latencySamples > 0 {
		averageLatency = s.totalLatencyMS / float64(s.latencySamples)
	}

	return map[string]interface{}{
		"total_requests":            s.totalRequests,
		"prediction_requests":       s.predictionRequests,
		"batch_prediction_requests": s.batchPredictionRequests,
		"failed_requests":           s.failedRequests,
		"average_latency_ms":        averageLatency,
		"last_prediction_timestamp": s.lastPredictionTimestamp,
	}
}
