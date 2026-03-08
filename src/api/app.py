"""
FastAPI entry point for triggering content campaigns.
"""

import logging
import time
import uuid

import structlog
from fastapi import BackgroundTasks, FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from src.shared.config import settings
from src.shared.db import get_db, save_campaign, update_campaign_status
from src.shared.models import CampaignBrief

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
logging.basicConfig(level=settings.log_level)
log = structlog.get_logger()

# ── Prometheus metrics ────────────────────────────────────────────────────────

campaigns_total = Counter("campaigns_total", "Total campaigns triggered")
campaigns_success = Counter("campaigns_success_total", "Successful campaigns")
campaigns_failed = Counter("campaigns_failed_total", "Failed campaigns")
campaign_duration = Histogram("campaign_duration_seconds", "Campaign execution time")

app = FastAPI(title="Content Pipeline API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/campaigns", status_code=202)
async def create_campaign(brief: CampaignBrief, background_tasks: BackgroundTasks):
    """Trigger a new content campaign from a brief."""
    campaign_id = str(uuid.uuid4())
    campaigns_total.inc()

    save_campaign(campaign_id, brief.model_dump())
    log.info("campaign_created", campaign_id=campaign_id, title=brief.title)

    background_tasks.add_task(trigger_pipeline, campaign_id, brief)

    return {
        "campaign_id": campaign_id,
        "status": "pending",
        "message": f"Campaign '{brief.title}' queued for processing",
    }


@app.get("/campaigns/{campaign_id}")
def get_campaign(campaign_id: str):
    """Get status and all content pieces for a campaign."""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM campaigns WHERE id = %s", (campaign_id,))
            campaign = cur.fetchone()
            if not campaign:
                raise HTTPException(404, f"Campaign {campaign_id} not found")

            cur.execute(
                "SELECT * FROM content_pieces WHERE campaign_id = %s ORDER BY created_at",
                (campaign_id,),
            )
            pieces = cur.fetchall()

            cur.execute(
                "SELECT * FROM crew_executions WHERE campaign_id = %s ORDER BY started_at",
                (campaign_id,),
            )
            executions = cur.fetchall()

    finally:
        conn.close()

    return {
        "campaign": dict(campaign),
        "content_pieces": [dict(p) for p in pieces],
        "crew_executions": [dict(e) for e in executions],
    }


@app.get("/campaigns")
def list_campaigns(limit: int = 20, offset: int = 0):
    """List all campaigns with summary."""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM campaigns ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (limit, offset),
            )
            campaigns = cur.fetchall()
    finally:
        conn.close()
    return {"campaigns": [dict(c) for c in campaigns], "limit": limit, "offset": offset}


@app.delete("/campaigns/{campaign_id}", status_code=204)
def delete_campaign(campaign_id: str):
    """Delete a campaign and all its content."""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM campaigns WHERE id = %s", (campaign_id,))
            if not cur.fetchone():
                raise HTTPException(404, f"Campaign {campaign_id} not found")
            cur.execute("DELETE FROM campaigns WHERE id = %s", (campaign_id,))
        conn.commit()
    finally:
        conn.close()


# ── Pipeline orchestration ────────────────────────────────────────────────────


async def trigger_pipeline(campaign_id: str, brief: CampaignBrief):
    """Run the full 4-crew pipeline as a background task."""
    start = time.time()
    try:
        from src.editing_crew.crew import run_editing_crew
        from src.research_crew.crew import run_research_crew
        from src.visual_crew.crew import run_visual_crew
        from src.writing_crew.crew import run_writing_crew

        # Stage 1 — Research
        update_campaign_status(campaign_id, "researching")
        log.info("pipeline_stage", campaign_id=campaign_id, stage="research")
        research = run_research_crew(campaign_id, brief)

        # Stage 2 — Writing (parallel internally)
        update_campaign_status(campaign_id, "writing")
        log.info("pipeline_stage", campaign_id=campaign_id, stage="writing")
        drafts = run_writing_crew(campaign_id, brief, research)

        # Stage 3 — Editing
        update_campaign_status(campaign_id, "editing")
        log.info("pipeline_stage", campaign_id=campaign_id, stage="editing")
        polished = run_editing_crew(campaign_id, brief, drafts)

        # Stage 4 — Visual Brief
        update_campaign_status(campaign_id, "visual_brief")
        log.info("pipeline_stage", campaign_id=campaign_id, stage="visual_brief")
        run_visual_crew(campaign_id, brief, polished)

        # Done
        update_campaign_status(campaign_id, "complete")
        campaigns_success.inc()
        campaign_duration.observe(time.time() - start)
        log.info(
            "pipeline_complete",
            campaign_id=campaign_id,
            elapsed=round(time.time() - start, 1),
        )

    except Exception as e:
        update_campaign_status(campaign_id, "failed")
        campaigns_failed.inc()
        log.error("pipeline_failed", campaign_id=campaign_id, error=str(e))
        raise


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8080, reload=False)
