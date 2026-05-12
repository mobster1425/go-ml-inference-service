package config

import "os"

type Config struct {
	ModelPath string
	Port      string
}

func Load() Config {
	return Config{
		ModelPath: getenv("MODEL_PATH", "../model_artifacts/churn_logistic_model.json"),
		Port:      getenv("PORT", "8080"),
	}
}

func getenv(key string, fallback string) string {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	return value
}
