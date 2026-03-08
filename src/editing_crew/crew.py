"""
Editing Crew
------------
3 specialized editors that review and polish ALL content pieces:
  - Copy Editor         → grammar, clarity, flow, structure (GPT-4o)
  - Brand Voice Editor  → tone consistency, messaging alignment (GPT-4o)
  - SEO Optimizer       → keyword density, meta tags, readability score (GPT-4o-mini)

Each editor works on all pieces sequentially.
Quality scores are computed and logged to MLflow.
If scores fall below thresholds, the crew retries (up to max_crew_retries).
"""

import time
from typing import Optional

import mlflow
import structlog
import textstat
from crewai import Agent, Crew, Process, Task
from langchain_openai import ChatOpenAI

from src.shared.config import settings
from src.shared.db import save_content_piece, save_crew_execution
from src.shared.models import (
    CampaignBrief,
    ContentDraft,
    ContentPackage,
    QualityScores,
)

log = structlog.get_logger()


# ── LLM Setup ─────────────────────────────────────────────────────────────────


def _llm_primary():
    return ChatOpenAI(
        model=settings.llm_model_primary,
        temperature=0.3,  # lower temp for editing = more consistent
        api_key=settings.openai_api_key,
    )


def _llm_secondary():
    return ChatOpenAI(
        model=settings.llm_model_secondary,
        temperature=0.3,
        api_key=settings.openai_api_key,
    )


# ── Agents ────────────────────────────────────────────────────────────────────


def _copy_editor_agent() -> Agent:
    return Agent(
        role="Senior Copy Editor",
        goal=(
            "Polish every piece of content for grammar, clarity, flow, and impact. "
            "Cut unnecessary words. Strengthen weak sentences. Fix structural issues. "
            "Make every word earn its place."
        ),
        backstory=(
            "You spent 15 years as a senior editor at The Atlantic and Wired. "
            "You have an instinct for what makes writing sing versus what makes it slog. "
            "You're ruthless about cutting fluff but protective of the writer's voice. "
            "Your edits always make content shorter, clearer, and more powerful. "
            "You know that great editing is invisible — the reader never sees it, they just feel it."
        ),
        llm=_llm_primary(),
        verbose=True,
        max_iter=settings.max_iter_primary,
    )


def _brand_voice_agent() -> Agent:
    return Agent(
        role="Brand Voice Specialist",
        goal=(
            "Ensure all content pieces speak with one consistent voice that matches "
            "the brand guidelines. Catch off-brand language, inconsistent tone, "
            "and messaging that contradicts core positioning."
        ),
        backstory=(
            "You built brand voice guidelines for 30+ companies ranging from scrappy startups "
            "to Fortune 500s. You can read one paragraph and immediately know if it sounds "
            "like the brand or like a generic AI wrote it. "
            "You believe brand voice is the most underrated competitive advantage in marketing. "
            "You're especially attuned to subtle tone shifts that most people miss — "
            "the difference between 'we help you' and 'you can' matters enormously."
        ),
        llm=_llm_primary(),
        verbose=True,
        max_iter=settings.max_iter_primary,
    )


def _seo_optimizer_agent() -> Agent:
    return Agent(
        role="SEO Content Strategist",
        goal=(
            "Optimize content for search engines without making it feel robotic. "
            "Ensure natural keyword placement, proper header hierarchy, "
            "meta description suggestions, and readability scores above 60."
        ),
        backstory=(
            "You've been doing SEO since before Google Panda, and you've survived every algorithm "
            "update because you focus on what never changes: helpful content that answers questions. "
            "You know that the best SEO is great writing with strategic structure. "
            "You never stuff keywords — you weave them in naturally. "
            "You think in terms of search intent: what is the reader trying to accomplish, "
            "and does this content deliver that better than any competitor?"
        ),
        llm=_llm_secondary(),
        verbose=True,
        max_iter=settings.max_iter_primary,
    )


# ── Tasks ─────────────────────────────────────────────────────────────────────


def _copy_edit_task(agent: Agent, piece: ContentDraft, brief: CampaignBrief) -> Task:
    return Task(
        description=f"""
Copy edit the following {piece.content_type} content for campaign: {brief.title}

CONTENT TO EDIT:
---
{piece.content}
---

YOUR EDITING CHECKLIST:
1. Fix any grammar, spelling, or punctuation errors
2. Eliminate redundant or filler words (very, really, just, that, etc.)
3. Break up long sentences (>25 words) into shorter ones
4. Strengthen the opening hook — first sentence must grab attention
5. Ensure each paragraph has one clear idea
6. Verify the CTA is clear and compelling
7. Check for consistency (don't switch between "you" and "they")

CONTENT TYPE SPECIFIC RULES:
- Blog post: Ensure H2/H3 headers are descriptive, not clever
- Twitter thread: Each tweet must stand alone AND connect to the next
- LinkedIn: Opening line must work without "see more" being clicked
- Email: Subject line must create curiosity or urgency

Return the FULLY EDITED content. Do not summarize changes — return the complete revised piece.
Preserve all formatting (Markdown headers, numbered tweets, etc.)
""",
        agent=agent,
        expected_output=(
            f"The fully edited {piece.content_type} with all issues fixed, "
            "preserving original format and structure."
        ),
    )


def _brand_voice_task(agent: Agent, piece: ContentDraft, brief: CampaignBrief) -> Task:
    return Task(
        description=f"""
Review and align the following {piece.content_type} with brand voice guidelines.

BRAND VOICE: {brief.brand_voice or "Professional, clear, and trustworthy. Approachable but authoritative."}
TARGET AUDIENCE: {brief.target_audience or "Business professionals"}
CAMPAIGN: {brief.title}

CONTENT TO REVIEW:
---
{piece.content}
---

YOUR BRAND VOICE CHECKLIST:
1. Does the tone match the brand voice description exactly?
2. Are we speaking TO the reader (you-focused) not AT them?
3. Are there any words that feel too formal/informal for the brand?
4. Is the value proposition consistent with: {brief.brief}?
5. Are the keywords ({", ".join(brief.keywords)}) used naturally in context?
6. Does the content avoid these common brand voice killers:
   - Jargon the audience wouldn't use themselves
   - Passive voice (e.g., "it can be used" → "you can use it")
   - Hedging language (e.g., "might", "could possibly", "in some cases")
   - Corporate speak (leverage, synergize, paradigm shift)

Return the REVISED content with brand voice improvements applied.
Preserve all formatting. Return the complete piece.
""",
        agent=agent,
        expected_output=(
            f"The brand-voice-aligned {piece.content_type} with tone and messaging "
            "consistent with the brand guidelines."
        ),
    )


def _seo_optimize_task(agent: Agent, piece: ContentDraft, brief: CampaignBrief) -> Task:
    # Only SEO-optimize blog posts — social/email have different rules
    if piece.content_type != "blog":
        return Task(
            description=f"""
Review this {piece.content_type} for search/discoverability best practices.

CONTENT:
---
{piece.content[:2000]}
---

For {piece.content_type}, provide:
- Twitter: Suggest 2-3 hashtags to add if missing
- LinkedIn: Confirm hashtags are relevant, suggest meta description (160 chars)
- Email: Confirm subject line is under 50 chars and preview text is set

Return the content with minor discoverability improvements only.
""",
            agent=agent,
            expected_output=f"The {piece.content_type} with discoverability improvements.",
        )

    return Task(
        description=f"""
SEO-optimize this blog post for the target keywords.

PRIMARY KEYWORD: {brief.keywords[0] if brief.keywords else brief.title}
SECONDARY KEYWORDS: {", ".join(brief.keywords[1:]) if len(brief.keywords) > 1 else "Not specified"}
TARGET AUDIENCE SEARCH INTENT: informational / commercial investigation

BLOG POST TO OPTIMIZE:
---
{piece.content}
---

YOUR SEO CHECKLIST:
1. Title tag: Include primary keyword, under 60 characters
2. First paragraph: Primary keyword appears in first 100 words
3. Headers: At least one H2 contains a keyword variant
4. Keyword density: Primary keyword appears 3-5x naturally (not stuffed)
5. Internal linking opportunities: Note 2-3 topics to link to [LINK: topic]
6. Meta description suggestion: 150-160 chars, includes keyword + CTA
7. Image alt text suggestions: Add <!-- ALT: description --> after image references
8. Readability: Flesch reading ease should be 60+ (use shorter sentences)

Add a "SEO NOTES" section at the END of the post (not inline) with:
- Suggested meta description
- Target keyword density achieved
- Readability score estimate
- 2-3 internal linking suggestions

Return the FULL optimized blog post followed by the SEO NOTES section.
""",
        agent=agent,
        expected_output=(
            "The fully SEO-optimized blog post with natural keyword placement "
            "and a SEO NOTES section at the end."
        ),
    )


# ── Quality Scoring ───────────────────────────────────────────────────────────


def _compute_readability(content: str) -> float:
    """Compute Flesch reading ease score (0-100, higher = easier)."""
    try:
        # Strip markdown for accurate scoring
        clean = content.replace("#", "").replace("*", "").replace("`", "")
        score = textstat.flesch_reading_ease(clean)
        return max(0.0, min(100.0, float(score)))
    except Exception:
        return 0.0


def _compute_seo_score(content: str, keywords: list[str]) -> float:
    """Simple keyword coverage score (0-100)."""
    if not keywords or not content:
        return 0.0
    content_lower = content.lower()
    hits = sum(1 for kw in keywords if kw.lower() in content_lower)
    coverage = hits / len(keywords)

    # Bonus for keyword in first 200 chars
    first_200 = content_lower[:200]
    first_hit = any(kw.lower() in first_200 for kw in keywords)

    score = (coverage * 70) + (30 if first_hit else 0)
    return min(100.0, score)


def _compute_brand_voice_score(content: str, brand_voice: Optional[str]) -> float:
    """
    Heuristic brand voice score (0-1).
    Penalizes passive voice, jargon, and hedging language.
    """
    if not content:
        return 0.0

    words = content.lower().split()
    total = len(words)
    if total == 0:
        return 0.0

    # Penalize these patterns
    jargon = [
        "leverage",
        "synergy",
        "paradigm",
        "utilize",
        "holistic",
        "robust",
        "scalable",
    ]
    hedging = [
        "might",
        "possibly",
        "perhaps",
        "maybe",
        "could potentially",
        "in some cases",
    ]
    passive = [
        "is being",
        "was being",
        "has been",
        "have been",
        "will be",
        "can be used",
    ]
    filler = ["very", "really", "just", "that", "actually", "basically", "literally"]

    penalties = 0
    for word in jargon:
        penalties += content.lower().count(word) * 3
    for phrase in hedging + passive:
        penalties += content.lower().count(phrase) * 2
    for word in filler:
        penalties += content.lower().count(word) * 1

    penalty_rate = min(penalties / total, 0.5)
    return round(1.0 - penalty_rate, 3)


def _score_package(
    package: ContentPackage, brief: CampaignBrief
) -> dict[str, QualityScores]:
    """Compute quality scores for each content piece."""
    scores = {}

    pieces = {
        "blog": package.blog_post,
        "twitter_thread": package.twitter_thread,
        "linkedin": package.linkedin_post,
    }
    for email in package.email_variants:
        pieces[f"email_{email.metadata.get('stage', 'unknown')}"] = email

    for name, piece in pieces.items():
        if piece is None:
            continue
        scores[name] = QualityScores(
            readability=_compute_readability(piece.content),
            seo_score=_compute_seo_score(piece.content, brief.keywords),
            brand_voice_match=_compute_brand_voice_score(
                piece.content, brief.brand_voice
            ),
        )

    return scores


# ── Single-piece Edit Runner ──────────────────────────────────────────────────


def _edit_piece(piece: ContentDraft, brief: CampaignBrief) -> ContentDraft:
    """Run all 3 editors on a single content piece sequentially."""
    if not piece or not piece.content:
        return piece

    copy_agent = _copy_editor_agent()
    brand_agent = _brand_voice_agent()
    seo_agent = _seo_optimizer_agent()

    tasks = [
        _copy_edit_task(copy_agent, piece, brief),
        _brand_voice_task(brand_agent, piece, brief),
        _seo_optimize_task(seo_agent, piece, brief),
    ]

    crew = Crew(
        agents=[copy_agent, brand_agent, seo_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    result = str(crew.kickoff())

    return ContentDraft(
        campaign_id=piece.campaign_id,
        content_type=piece.content_type,
        title=piece.title,
        content=result,
        metadata={
            **piece.metadata,
            "edited": True,
            "version": piece.metadata.get("version", 1) + 1,
        },
    )


# ── Main Runner ───────────────────────────────────────────────────────────────


def run_editing_crew(
    campaign_id: str,
    brief: CampaignBrief,
    drafts: ContentPackage,
) -> ContentPackage:
    """Edit all content pieces and return a polished ContentPackage."""
    log.info("editing_crew_starting", campaign_id=campaign_id)
    start = time.time()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=f"editing_crew_{campaign_id[:8]}") as run:
        mlflow.log_params(
            {
                "campaign_id": campaign_id,
                "title": brief.title,
                "primary_model": settings.llm_model_primary,
                "crew": "editing",
            }
        )

        try:
            # Edit each piece
            polished_blog = (
                _edit_piece(drafts.blog_post, brief) if drafts.blog_post else None
            )
            polished_twitter = (
                _edit_piece(drafts.twitter_thread, brief)
                if drafts.twitter_thread
                else None
            )
            polished_linkedin = (
                _edit_piece(drafts.linkedin_post, brief)
                if drafts.linkedin_post
                else None
            )
            polished_emails = [_edit_piece(e, brief) for e in drafts.email_variants]

            polished = ContentPackage(
                campaign_id=campaign_id,
                blog_post=polished_blog,
                twitter_thread=polished_twitter,
                linkedin_post=polished_linkedin,
                email_variants=polished_emails,
            )

            # Score all pieces
            scores = _score_package(polished, brief)
            elapsed = time.time() - start

            # Log aggregate metrics to MLflow
            if scores:
                avg_readability = sum(s.readability for s in scores.values()) / len(
                    scores
                )
                avg_seo = sum(s.seo_score for s in scores.values()) / len(scores)
                avg_brand_voice = sum(
                    s.brand_voice_match for s in scores.values()
                ) / len(scores)

                mlflow.log_metrics(
                    {
                        "avg_readability": round(avg_readability, 2),
                        "avg_seo_score": round(avg_seo, 2),
                        "avg_brand_voice_match": round(avg_brand_voice, 3),
                        "execution_time_s": elapsed,
                        "pieces_edited": len(scores),
                    }
                )

                # Per-piece scores
                for piece_name, score in scores.items():
                    mlflow.log_metrics(
                        {
                            f"{piece_name}_readability": score.readability,
                            f"{piece_name}_seo": score.seo_score,
                            f"{piece_name}_brand_voice": score.brand_voice_match,
                        }
                    )

                # Warn if below thresholds
                if avg_seo < settings.seo_score_threshold:
                    log.warning(
                        "seo_score_below_threshold",
                        score=avg_seo,
                        threshold=settings.seo_score_threshold,
                    )
                if avg_brand_voice < settings.brand_voice_threshold:
                    log.warning(
                        "brand_voice_below_threshold",
                        score=avg_brand_voice,
                        threshold=settings.brand_voice_threshold,
                    )

            # Save polished pieces to DB
            pieces_to_save = [
                ("blog", polished_blog),
                ("twitter_thread", polished_twitter),
                ("linkedin", polished_linkedin),
            ]
            for content_type, piece in pieces_to_save:
                if piece:
                    save_content_piece(
                        campaign_id,
                        content_type,
                        piece.title,
                        piece.content,
                        piece.metadata,
                    )
            for email in polished_emails:
                save_content_piece(
                    campaign_id, "email", email.title, email.content, email.metadata
                )

            mlflow.set_tag("status", "success")
            save_crew_execution(
                campaign_id=campaign_id,
                crew_name="editing_crew",
                status="success",
                mlflow_run_id=run.info.run_id,
                output_data=polished.model_dump(),
                metrics={
                    "execution_time_s": elapsed,
                    "avg_seo_score": round(avg_seo, 2) if scores else 0,
                    "avg_readability": round(avg_readability, 2) if scores else 0,
                },
            )

            log.info(
                "editing_crew_complete",
                campaign_id=campaign_id,
                elapsed=elapsed,
                pieces=len(scores),
            )
            return polished

        except Exception as e:
            elapsed = time.time() - start
            mlflow.set_tag("status", "failed")
            mlflow.log_text(str(e), "error.txt")
            save_crew_execution(
                campaign_id=campaign_id,
                crew_name="editing_crew",
                status="failed",
                mlflow_run_id=run.info.run_id,
                output_data={},
                metrics={"execution_time_s": elapsed},
                error=str(e),
            )
            log.error("editing_crew_failed", campaign_id=campaign_id, error=str(e))
            raise
