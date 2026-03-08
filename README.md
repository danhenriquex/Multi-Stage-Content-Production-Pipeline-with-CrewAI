# Multi-Stage Content Production Pipeline

An end-to-end AI content creation system where specialized CrewAI agents collaborate to produce publication-ready marketing materials from a single campaign brief.

## Architecture

```
POST /campaigns
      │
      ▼
  FastAPI (api:8080)
      │  background task
      ▼
┌─────────────────────────────────────────────┐
│           Pipeline Orchestration             │
│                                             │
│  1. Research Crew ──────────────────────►  │
│     Market Analyst + Competitor Intel       │
│     + Trend Scout + Research Director       │
│                  │                          │
│                  ▼                          │
│  2. Writing Crew (parallel) ─────────────► │
│     ┌──────────┬──────────┬─────────┐      │
│     Blog     Social    Email        │      │
│     Writer   Writer    Writer       │      │
│     └──────────┴──────────┴─────────┘      │
│                  │                          │
│                  ▼                          │
│  3. Editing Crew ────────────────────────► │
│     Copy Editor + Brand Voice              │
│     + SEO Optimizer                        │
│                  │                          │
│                  ▼                          │
│  4. Visual Crew ─────────────────────────► │
│     Visual Strategist + Asset Planner      │
└─────────────────────────────────────────────┘
      │
      ▼
PostgreSQL (content DB)
MLflow (experiment tracking)
Prometheus + Grafana (metrics)
```

## Output per Campaign

| Content Type | Description |
|---|---|
| Blog post | 1400-1600 words, SEO-optimized, Markdown |
| Twitter/X thread | 8 tweets with hook + CTA |
| LinkedIn post | 150-300 words with hashtags |
| Email campaign | 3 variants: awareness / nurture / conversion |
| Visual design brief | Color palette, typography, full asset production list |

## Services

| Service | URL | Description |
|---|---|---|
| API | http://localhost:8080/docs | FastAPI — trigger & query campaigns |
| Dagster | http://localhost:3001 | Pipeline orchestration UI |
| MLflow | http://localhost:5050 | Experiment tracking |
| Grafana | http://localhost:3000 | Metrics dashboards |
| Prometheus | http://localhost:9090 | Metrics collection |

## Quickstart

```bash
# 1. Setup
make init                    # creates .env from .env.example
# Fill in: OPENAI_API_KEY, TAVILY_API_KEY, POSTGRES_PASSWORD, GRAFANA_PASSWORD

# 2. Start infrastructure only
docker compose -p content-pipeline up -d postgres chromadb mlflow prometheus grafana pushgateway

# 3. Build and start app services
docker compose -p content-pipeline build api dagster
docker compose -p content-pipeline up -d api dagster

# 4. Verify everything is up
make health

# 5. Trigger your first campaign
make campaign
```

## API Usage

```bash
# Create a campaign
curl -X POST http://localhost:8080/campaigns \
  -H "Content-Type: application/json" \
  -d '{
    "title": "AI CRM Launch",
    "brief": "Launch campaign for new AI-powered CRM targeting SMBs",
    "brand_voice": "Professional yet approachable, data-driven",
    "target_audience": "SMB founders and sales managers",
    "keywords": ["AI CRM", "sales automation", "SMB"]
  }'

# Check campaign status + results
curl http://localhost:8080/campaigns/{campaign_id}

# List all campaigns
curl http://localhost:8080/campaigns
```

## Campaign Status Flow

```
pending → researching → writing → editing → visual_brief → complete
                                                         → failed
```

## Running Tests

```bash
# Unit tests (no Docker needed)
uv run pytest tests/unit/ -v

# With coverage
uv run pytest tests/unit/ --cov=src --cov-report=term-missing
```

## Project Structure

```
content-pipeline/
├── src/
│   ├── api/              # FastAPI app
│   ├── research_crew/    # Market research agents
│   ├── writing_crew/     # Blog + social + email writers (parallel)
│   ├── editing_crew/     # Copy editor + brand voice + SEO
│   ├── visual_crew/      # Visual strategist + asset planner
│   ├── dagster_pipeline/ # Dagster asset definitions
│   └── shared/           # Config, models, DB helpers
├── configs/              # Prometheus, Grafana, MLflow, SQL schema
├── tests/unit/           # Unit tests (all LLM calls mocked)
└── docker-compose.yml    # All 8 services
```

## Environment Variables

```bash
OPENAI_API_KEY=sk-...          # Required
TAVILY_API_KEY=tvly-...        # Required (research web search)
POSTGRES_PASSWORD=...          # Required
GRAFANA_PASSWORD=...           # Required
LOG_LEVEL=INFO                 # Optional (default: INFO)
```# Multi-Stage-Content-Production-Pipeline-with-CrewAI
