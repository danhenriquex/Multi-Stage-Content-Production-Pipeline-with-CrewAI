-- Content Pipeline Database Schema

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Campaigns ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS campaigns (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title           TEXT NOT NULL,
    brief           TEXT NOT NULL,
    brand_voice     TEXT,
    target_audience TEXT,
    keywords        TEXT[],
    status          TEXT NOT NULL DEFAULT 'pending',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Content Pieces ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS content_pieces (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id   UUID REFERENCES campaigns(id) ON DELETE CASCADE,
    content_type  TEXT NOT NULL,
    title         TEXT,
    content       TEXT,
    metadata      JSONB DEFAULT '{}',
    status        TEXT NOT NULL DEFAULT 'draft',
    version       INTEGER DEFAULT 1,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);

-- ── Crew Executions ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS crew_executions (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id   UUID REFERENCES campaigns(id) ON DELETE CASCADE,
    crew_name     TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'running',
    mlflow_run_id TEXT,
    input_data    JSONB DEFAULT '{}',
    output_data   JSONB DEFAULT '{}',
    metrics       JSONB DEFAULT '{}',
    error_message TEXT,
    started_at    TIMESTAMPTZ DEFAULT NOW(),
    completed_at  TIMESTAMPTZ
);

-- ── Quality Scores ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS quality_scores (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_piece_id  UUID REFERENCES content_pieces(id) ON DELETE CASCADE,
    readability       FLOAT,
    seo_score         FLOAT,
    brand_voice_match FLOAT,
    cost_usd          FLOAT,
    execution_time_s  FLOAT,
    scored_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_campaigns_status ON campaigns(status);
CREATE INDEX IF NOT EXISTS idx_content_pieces_campaign ON content_pieces(campaign_id);
CREATE INDEX IF NOT EXISTS idx_content_pieces_type ON content_pieces(content_type);
CREATE INDEX IF NOT EXISTS idx_crew_executions_campaign ON crew_executions(campaign_id);
CREATE INDEX IF NOT EXISTS idx_crew_executions_status ON crew_executions(status);
