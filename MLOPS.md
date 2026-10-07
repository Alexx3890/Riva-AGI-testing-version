# Riva AGI MLOps Baseline

## Current deployment unit

The first deployable service is the real-time voice gateway, exposed on port `8000`. It is containerized by `Dockerfile`, started locally with Compose, and has a `GET /health` liveness endpoint. The gateway requires `GEMINI_API_KEY`; keep it in a local `.env` file or a deployment secret, never in Git.

```bash
copy .env.example .env
docker compose up --build voice-gateway
```

For a local MongoDB instance used by the RAG module, run:

```bash
docker compose --profile local-rag up -d mongodb
```

Set `MONGODB_URI=mongodb://mongodb:27017` when a containerized RAG service is added to the same Compose network. The present voice gateway does not require MongoDB to start.

## CI contract

`.github/workflows/ci.yml` runs on every pull request and push to `main` or `master`. It:

1. installs all Python dependency groups;
2. runs unit tests for orchestration, RAG, and voice without live credentials; and
3. verifies that the production container builds.

Integration tests stay outside this workflow because they can call an LLM and require deliberate secrets, quotas, and stable test fixtures. Add them as a separately protected environment after a test account and budget cap are available.

## Release path

1. Merge a passing pull request to the protected default branch.
2. Build and tag the image with the immutable Git commit SHA.
3. Push it to a container registry using GitHub Actions OIDC or a short-lived registry credential.
4. Deploy that exact tag to staging and wait for `/health` plus a real WebSocket smoke test.
5. Promote the unchanged image digest to production; roll back by redeploying the prior digest.

Do not use mutable tags such as `latest` for promotion or rollback.

## What to measure before production

The PRD requires observability, but the application currently emits only standard logs. Add a centralized log sink and these minimum metrics before calling the environment production-ready:

| Signal | Minimum labels or dimensions | Why it matters |
| --- | --- | --- |
| HTTP and WebSocket request count | route, status, deployment version | Traffic and error-rate visibility |
| WebSocket session duration and concurrent sessions | voice, language, version | Capacity and user-experience trend |
| Gemini call latency and failures | model, operation, error class | Provider reliability and cost control |
| RAG retrieval quality | source, result count, confidence/score | Detect incomplete or weak institutional knowledge |
| CPU, memory and container restarts | service, instance, version | Saturation and stability alerts |

Never label metrics with user prompts, user IDs, raw document text, API keys, or other institutional personal data.

## MLOps gaps to prioritize

1. **Secrets and environments:** separate development, staging, and production secrets; define who can deploy each environment.
2. **Model and prompt registry:** record provider, exact model name, prompt version, evaluation-set version, and Git SHA for every release.
3. **Evaluation gate:** use a versioned KIET ground-truth set for retrieval accuracy, answer quality, Hindi-English handling, latency, and tool-call safety. Require threshold results before promotion.
4. **Observability:** introduce structured JSON logs, traces that propagate a request ID through agents and tools, metrics, dashboards, and alerts.
5. **Data governance:** classify KIET data, restrict access by role, retain audit logs, and define deletion/refresh ownership for the knowledge base.

## Suggested first acceptance criteria

- A pull request cannot merge if tests or image build fail.
- A deployment uses a pinned image digest and succeeds only after the health probe passes.
- Every answer can be traced to a deployment version, model/prompt version, and request ID without storing its private content in metrics.
- A fixed evaluation dataset reports retrieval and answer metrics for every release candidate.
