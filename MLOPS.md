# Riva AGI MLOps Baseline & Observability

## Current deployment unit

The primary deployable service is the real-time voice gateway, exposed on port `8000`. It is containerized by `Dockerfile`, started locally with Compose, and includes built-in structured logging, distributed request tracing, and telemetry metrics. The gateway requires `GEMINI_API_KEY`; keep it in a local `.env` file or deployment secret store, never in Git.

```bash
copy .env.example .env
docker compose up --build voice-gateway
```

For a local MongoDB instance used by the RAG module, run:

```bash
docker compose --profile local-rag up -d mongodb
```

Set `MONGODB_URI=mongodb://mongodb:27017` when a containerized RAG service is added to the same Compose network. The voice gateway does not require MongoDB to start.

---

## Observability & Telemetry

Riva includes a built-in, lightweight observability subsystem (`observability/`) with zero heavy runtime dependencies:

### 1. Structured JSON Logging
- **Format**: Outputs single-line JSON log objects in containers/production (`LOG_FORMAT=json`) and human-readable formatted logs in local development (`LOG_FORMAT=text`).
- **Standard Fields**: `timestamp` (ISO-8601 UTC), `level`, `service`, `environment`, `version`, `logger`, `message`, `location`, `request_id`, `session_id`, `extra`, and `error` (stack traces).
- **Security & Sanitization**: Automatically scrubs API keys (e.g. `AIza...`), Bearer tokens, passwords, and database connection URIs.

### 2. Request & Session Correlation
- **HTTP**: `RequestIDAndLoggingMiddleware` inspects incoming `X-Request-ID` headers or auto-generates a unique UUID hex. Injects the ID into contextvars and response headers.
- **WebSocket**: Tracks client voice sessions using unique `session_id` (`X-Session-ID`), logging connection lifecycles and measuring session duration.

### 3. Operational Probes & Metrics Endpoints
- **Liveness Probe**: `GET /health` — Returns status `ok`, service name, version, uptime, and active/max concurrent session counts.
- **Prometheus Metrics**: `GET /metrics` — Formatted for Prometheus / Grafana scraping:
  - `riva_http_requests_total{method, route, status}` (Counter)
  - `riva_active_websocket_sessions` (Gauge)
  - `riva_websocket_sessions_total{voice, language}` (Counter)
  - `riva_gemini_api_calls_total{model, operation, status}` (Counter)
  - `riva_system_cpu_percent` & `riva_system_memory_percent` (Gauges)
  - `riva_service_uptime_seconds` (Gauge)
  - `riva_errors_total{error_class, route}` (Counter)
- **JSON Metrics**: `GET /metrics/json` — Real-time telemetry dictionary for instant local inspection.

---

## CI Contract

`.github/workflows/ci.yml` runs on every pull request and push to `main` or `master`. It:

1. installs all Python dependency groups;
2. runs unit tests for orchestration, RAG, voice, and observability without live credentials; and
3. verifies that the production container builds cleanly.

Integration tests stay outside this workflow because they can call live LLMs and require deliberate secrets and quotas.

---

## Release Path

1. Open a pull request against the `main` branch.
2. Verify all CI checks pass (`Python tests` and `Container build`).
3. Build and tag the production container image with the immutable Git commit SHA.
4. Push to container registry (e.g., GitHub Container Registry `ghcr.io` or Docker Hub).
5. Deploy that exact tag to staging, verifying `/health` and `/metrics`.
6. Promote the identical image digest to production. Roll back instantly by redeploying the previous image digest.

---

## MLOps Roadmap & Next Milestones

1. **AI Evaluation Gate**: Versioned evaluation dataset for RAG retrieval accuracy, voice intent classification, latency thresholds, and guardrails.
2. **Prometheus & Grafana Stack**: Local docker-compose profile for visual dashboarding of metrics exposed by `/metrics`.
3. **Model & Prompt Registry**: Pinning model versions (`gemini-2.5-flash`, etc.) and system prompt revisions with release tags.
4. **Staging Cloud Environment**: Moving beyond local Docker to a cloud host (Render, Fly.io, or GCP/AWS).
