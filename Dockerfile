FROM golang:1.22-bookworm AS builder

WORKDIR /src
COPY service/go.mod ./service/go.mod
COPY service ./service
COPY model_artifacts ./model_artifacts
WORKDIR /src/service
RUN go test ./...
RUN CGO_ENABLED=0 GOOS=linux go build -o /out/inference-service ./cmd/server

FROM gcr.io/distroless/static-debian12:nonroot

WORKDIR /app
COPY --from=builder /out/inference-service /app/inference-service
COPY model_artifacts /app/model_artifacts

ENV MODEL_PATH=/app/model_artifacts/churn_logistic_model.json
ENV PORT=8080
EXPOSE 8080

USER nonroot:nonroot
ENTRYPOINT ["/app/inference-service"]
