import psycopg2
import psycopg2.extras
from src.shared.config import settings


def get_db():
    return psycopg2.connect(
        settings.postgres_url, cursor_factory=psycopg2.extras.RealDictCursor
    )


def save_campaign(campaign_id: str, brief: dict) -> None:
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO campaigns (id, title, brief, brand_voice, target_audience, keywords, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'pending')
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    updated_at = NOW()
                """,
                (
                    campaign_id,
                    brief.get("title", ""),
                    brief.get("brief", ""),
                    brief.get("brand_voice"),
                    brief.get("target_audience"),
                    brief.get("keywords", []),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def update_campaign_status(campaign_id: str, status: str) -> None:
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE campaigns SET status=%s, updated_at=NOW() WHERE id=%s",
                (status, campaign_id),
            )
        conn.commit()
    finally:
        conn.close()


def save_content_piece(
    campaign_id: str, content_type: str, title: str, content: str, metadata: dict
) -> str:
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO content_pieces (campaign_id, content_type, title, content, metadata, status)
                VALUES (%s, %s, %s, %s, %s, 'draft')
                RETURNING id
                """,
                (
                    campaign_id,
                    content_type,
                    title,
                    content,
                    psycopg2.extras.Json(metadata),
                ),
            )
            piece_id = cur.fetchone()["id"]
        conn.commit()
        return str(piece_id)
    finally:
        conn.close()


def save_crew_execution(
    campaign_id: str,
    crew_name: str,
    status: str,
    mlflow_run_id: str,
    output_data: dict,
    metrics: dict,
    error: str = None,
) -> None:
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO crew_executions
                    (campaign_id, crew_name, status, mlflow_run_id, output_data, metrics, error_message, completed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                """,
                (
                    campaign_id,
                    crew_name,
                    status,
                    mlflow_run_id,
                    psycopg2.extras.Json(output_data),
                    psycopg2.extras.Json(metrics),
                    error,
                ),
            )
        conn.commit()
    finally:
        conn.close()
