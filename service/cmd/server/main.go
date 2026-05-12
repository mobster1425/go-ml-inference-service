package main

import (
	"fmt"
	"log"
	"net/http"

	"github.com/mobster1425/go-ml-inference-service/service/internal/api"
	"github.com/mobster1425/go-ml-inference-service/service/internal/config"
	"github.com/mobster1425/go-ml-inference-service/service/internal/metrics"
	"github.com/mobster1425/go-ml-inference-service/service/internal/model"
)

func main() {
	cfg := config.Load()

	artifact, err := model.LoadModel(cfg.ModelPath)
	if err != nil {
		log.Fatalf("failed to load model: %v", err)
	}

	server := api.NewServer(artifact, metrics.NewStore())
	addr := ":" + cfg.Port

	fmt.Printf("go-ml-inference-service listening on %s\n", addr)
	fmt.Printf("loaded model %s version %s from %s\n", artifact.ModelName, artifact.ModelVersion, cfg.ModelPath)

	if err := http.ListenAndServe(addr, server.Routes()); err != nil {
		log.Fatalf("server stopped: %v", err)
	}
}
