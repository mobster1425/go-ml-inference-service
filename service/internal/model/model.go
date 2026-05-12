package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
)

type Preprocessing struct {
	Means map[string]float64 `json:"means"`
	Stds  map[string]float64 `json:"stds"`
}

type ModelArtifact struct {
	ModelName           string              `json:"model_name"`
	ModelVersion        string              `json:"model_version"`
	DatasetSource       string              `json:"dataset_source"`
	CreatedAt           string              `json:"created_at"`
	Target              string              `json:"target"`
	Threshold           float64             `json:"threshold"`
	NumericFeatures     []string            `json:"numeric_features"`
	CategoricalFeatures map[string][]string `json:"categorical_features"`
	FeatureOrder        []string            `json:"feature_order"`
	Preprocessing       Preprocessing       `json:"preprocessing"`
	Coefficients        []float64           `json:"coefficients"`
	Intercept           float64             `json:"intercept"`
	Metrics             map[string]float64  `json:"metrics"`
}

type CustomerInput struct {
	CreditScore     float64 `json:"CreditScore"`
	Geography       string  `json:"Geography"`
	Gender          string  `json:"Gender"`
	Age             float64 `json:"Age"`
	Tenure          float64 `json:"Tenure"`
	Balance         float64 `json:"Balance"`
	NumOfProducts   float64 `json:"NumOfProducts"`
	HasCrCard       float64 `json:"HasCrCard"`
	IsActiveMember  float64 `json:"IsActiveMember"`
	EstimatedSalary float64 `json:"EstimatedSalary"`
}

type PredictionResult struct {
	Probability  float64 `json:"probability"`
	Prediction   int     `json:"prediction"`
	ModelName    string  `json:"model_name"`
	ModelVersion string  `json:"model_version"`
	LatencyMS    float64 `json:"latency_ms"`
}

func LoadModel(path string) (*ModelArtifact, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read model artifact %q: %w", path, err)
	}

	var artifact ModelArtifact
	if err := json.Unmarshal(data, &artifact); err != nil {
		return nil, fmt.Errorf("decode model artifact: %w", err)
	}
	if err := artifact.Validate(); err != nil {
		return nil, err
	}
	return &artifact, nil
}

func (m *ModelArtifact) Validate() error {
	if m == nil {
		return errors.New("model artifact is nil")
	}
	if m.ModelName == "" || m.ModelVersion == "" {
		return errors.New("model artifact is missing model name or version")
	}
	if len(m.FeatureOrder) == 0 {
		return errors.New("model artifact has empty feature order")
	}
	if len(m.Coefficients) != len(m.FeatureOrder) {
		return fmt.Errorf("coefficient length %d does not match feature order length %d", len(m.Coefficients), len(m.FeatureOrder))
	}
	for _, feature := range m.NumericFeatures {
		if _, ok := m.Preprocessing.Means[feature]; !ok {
			return fmt.Errorf("missing preprocessing mean for %s", feature)
		}
		std, ok := m.Preprocessing.Stds[feature]
		if !ok {
			return fmt.Errorf("missing preprocessing std for %s", feature)
		}
		if std == 0 {
			return fmt.Errorf("preprocessing std for %s is zero", feature)
		}
	}
	return nil
}

func (m *ModelArtifact) Predict(input CustomerInput) (float64, int, error) {
	features, err := m.Preprocess(input)
	if err != nil {
		return 0, 0, err
	}
	if len(features) != len(m.Coefficients) {
		return 0, 0, fmt.Errorf("processed feature length %d does not match coefficient length %d", len(features), len(m.Coefficients))
	}

	score := m.Intercept
	for i, value := range features {
		score += m.Coefficients[i] * value
	}

	probability := clipProbability(Sigmoid(score))
	prediction := 0
	if probability >= m.Threshold {
		prediction = 1
	}
	return probability, prediction, nil
}

func (m *ModelArtifact) Preprocess(input CustomerInput) ([]float64, error) {
	if m == nil {
		return nil, errors.New("model artifact is nil")
	}

	numericValues := map[string]float64{}
	for _, feature := range m.NumericFeatures {
		raw, err := numericValue(input, feature)
		if err != nil {
			return nil, err
		}
		mean, meanOK := m.Preprocessing.Means[feature]
		std, stdOK := m.Preprocessing.Stds[feature]
		if !meanOK || !stdOK {
			return nil, fmt.Errorf("missing preprocessing metadata for %s", feature)
		}
		if std == 0 {
			return nil, fmt.Errorf("standard deviation for %s is zero", feature)
		}
		numericValues[feature] = (raw - mean) / std
	}

	categoricalValues := map[string]float64{}
	for feature, categories := range m.CategoricalFeatures {
		actual, err := categoricalValue(input, feature)
		if err != nil {
			return nil, err
		}
		if !contains(categories, actual) {
			return nil, fmt.Errorf("invalid category %q for %s", actual, feature)
		}
		for _, category := range categories {
			value := 0.0
			if actual == category {
				value = 1.0
			}
			categoricalValues[feature+"_"+category] = value
		}
	}

	processed := make([]float64, 0, len(m.FeatureOrder))
	for _, feature := range m.FeatureOrder {
		if value, ok := numericValues[feature]; ok {
			processed = append(processed, value)
			continue
		}
		if value, ok := categoricalValues[feature]; ok {
			processed = append(processed, value)
			continue
		}
		return nil, fmt.Errorf("feature %q in feature_order was not produced by preprocessing", feature)
	}
	return processed, nil
}

func Sigmoid(x float64) float64 {
	if x >= 0 {
		z := math.Exp(-x)
		return 1 / (1 + z)
	}
	z := math.Exp(x)
	return z / (1 + z)
}

func numericValue(input CustomerInput, feature string) (float64, error) {
	switch feature {
	case "CreditScore":
		return input.CreditScore, nil
	case "Age":
		return input.Age, nil
	case "Tenure":
		return input.Tenure, nil
	case "Balance":
		return input.Balance, nil
	case "NumOfProducts":
		return input.NumOfProducts, nil
	case "HasCrCard":
		return input.HasCrCard, nil
	case "IsActiveMember":
		return input.IsActiveMember, nil
	case "EstimatedSalary":
		return input.EstimatedSalary, nil
	default:
		return 0, fmt.Errorf("unsupported numeric feature %q", feature)
	}
}

func categoricalValue(input CustomerInput, feature string) (string, error) {
	switch feature {
	case "Geography":
		return input.Geography, nil
	case "Gender":
		return input.Gender, nil
	default:
		return "", fmt.Errorf("unsupported categorical feature %q", feature)
	}
}

func contains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func clipProbability(probability float64) float64 {
	if math.IsNaN(probability) {
		return 0
	}
	if probability < 0 {
		return 0
	}
	if probability > 1 {
		return 1
	}
	return probability
}
