"""
Visual Crew
-----------
2 specialized agents that produce a design brief for human designers:
  - Visual Strategist   → color palette, typography, mood, visual direction (GPT-4o)
  - Asset Planner       → specific asset list: images, graphics, dimensions (GPT-4o-mini)

Output is a structured ContentDraft of type "visual_brief" — a markdown doc
designers can open directly and action without any follow-up questions.
"""

import time

import mlflow
import structlog
from crewai import Agent, Crew, Process, Task
from langchain_openai import ChatOpenAI

from src.shared.config import settings
from src.shared.db import save_content_piece, save_crew_execution
from src.shared.models import (
    CampaignBrief,
    ContentDraft,
    ContentPackage,
)

log = structlog.get_logger()


# ── LLM Setup ─────────────────────────────────────────────────────────────────


def _llm_primary():
    return ChatOpenAI(
        model=settings.llm_model_primary,
        temperature=0.7,
        api_key=settings.openai_api_key,
    )


def _llm_secondary():
    return ChatOpenAI(
        model=settings.llm_model_secondary,
        temperature=0.7,
        api_key=settings.openai_api_key,
    )


# ── Agents ────────────────────────────────────────────────────────────────────


def _visual_strategist_agent() -> Agent:
    return Agent(
        role="Visual Brand Strategist",
        goal=(
            "Define the visual direction for the campaign: mood, color palette, "
            "typography style, photography direction, and overall aesthetic. "
            "The direction must feel cohesive across all channels."
        ),
        backstory=(
            "You spent 10 years as a Creative Director at top-tier agencies including "
            "Wieden+Kennedy and IDEO. You've launched visual identities for brands like "
            "Slack, Notion, and Linear — brands known for their distinctive, memorable look. "
            "You think in systems, not individual assets. Every visual decision serves "
            "the brand story and the audience's emotional journey. "
            "You write design briefs that make designers excited, not confused."
        ),
        llm=_llm_primary(),
        verbose=True,
        max_iter=settings.max_iter_primary,
    )


def _asset_planner_agent() -> Agent:
    return Agent(
        role="Digital Asset Production Planner",
        goal=(
            "Create a comprehensive, channel-by-channel asset list with exact specifications: "
            "dimensions, formats, quantities, and creative direction for each asset. "
            "Leave zero ambiguity for the production team."
        ),
        backstory=(
            "You ran production at a high-volume content studio for 8 years, "
            "delivering assets for brands like Shopify and HubSpot. "
            "You are obsessed with specs because you've seen what happens when they're missing. "
            "You know every platform's requirements by heart: "
            "Twitter card sizes, LinkedIn image ratios, email-safe image widths, blog hero dimensions. "
            "Your asset lists are legendary for their clarity — "
            "a junior designer can execute them without a single question."
        ),
        llm=_llm_secondary(),
        verbose=True,
        max_iter=settings.max_iter_primary,
    )


# ── Content Summary Helper ────────────────────────────────────────────────────


def _content_summary(package: ContentPackage, brief: CampaignBrief) -> str:
    """Extract key themes and messaging from the polished content package."""
    summary_parts = [
        f"CAMPAIGN: {brief.title}",
        f"BRIEF: {brief.brief}",
        f"BRAND VOICE: {brief.brand_voice or 'Professional yet approachable'}",
        f"TARGET AUDIENCE: {brief.target_audience or 'Business professionals'}",
        f"KEYWORDS: {', '.join(brief.keywords) if brief.keywords else 'Not specified'}",
        "",
        "CONTENT PRODUCED:",
    ]

    if package.blog_post:
        summary_parts.append(f"- Blog Post: '{package.blog_post.title}'")
        # First 300 chars of blog as context
        preview = package.blog_post.content[:300].replace("\n", " ")
        summary_parts.append(f"  Preview: {preview}...")

    if package.twitter_thread:
        first_tweet = package.twitter_thread.content.split("\n")[0][:200]
        summary_parts.append(f"- Twitter Thread: {first_tweet}")

    if package.linkedin_post:
        first_line = package.linkedin_post.content.split("\n")[0][:200]
        summary_parts.append(f"- LinkedIn Post: {first_line}")

    if package.email_variants:
        summary_parts.append(f"- Email Campaign: {len(package.email_variants)} variants")
        for email in package.email_variants:
            stage = email.metadata.get("stage", "unknown")
            subject = email.metadata.get("subject", "")
            summary_parts.append(f"  {stage.title()}: '{subject}'")

    return "\n".join(summary_parts)


# ── Tasks ─────────────────────────────────────────────────────────────────────


def _visual_strategy_task(agent: Agent, brief: CampaignBrief, package: ContentPackage) -> Task:
    content_ctx = _content_summary(package, brief)
    return Task(
        description=f"""
Define the complete visual strategy for this content campaign.

{content_ctx}

YOUR DELIVERABLES:

## 1. Campaign Mood & Aesthetic
- 3 mood words that capture the visual feeling
- Visual references (describe, don't link): "Think [Brand X] meets [Brand Y]"
- What this campaign should NOT look like (anti-examples)

## 2. Color Palette
- Primary color: hex code + name + emotion it conveys
- Secondary color: hex code + name + usage context
- Accent color: hex code + name + when to use sparingly
- Background/neutral: hex code + usage
- Text color: hex code

## 3. Typography Direction
- Headline font style: (e.g., bold geometric sans-serif, elegant serif)
- Body font style: (e.g., clean readable sans-serif)
- Font pairing mood: (e.g., "confident and modern", "warm and trustworthy")
- Font size hierarchy: H1 / H2 / Body / Caption guidelines

## 4. Photography & Imagery Direction
- Subject matter: what to show (people, product, abstract, data viz?)
- Composition style: (e.g., clean white backgrounds, lifestyle/in-context, flat lay)
- Color treatment: (e.g., bright and airy, moody and dramatic, desaturated)
- What to avoid in imagery

## 5. Iconography & Graphics Style
- Icon style: (e.g., outlined, filled, rounded, sharp)
- Data visualization style: charts, graphs approach
- Illustration style if applicable

## 6. Channel-Specific Visual Notes
- Blog hero image: art direction guidance
- Social media: frame/overlay style, text treatment on images
- Email: header image approach

Format as a clean, structured Markdown document a designer can hand off directly.
""",
        agent=agent,
        expected_output=(
            "A complete visual strategy document in Markdown with mood, color palette, "
            "typography, photography direction, and channel-specific notes."
        ),
    )


def _asset_list_task(agent: Agent, brief: CampaignBrief, package: ContentPackage) -> Task:
    content_ctx = _content_summary(package, brief)
    return Task(
        description=f"""
Create a complete production asset list for this campaign. Every asset the team needs to create.

{content_ctx}

For EACH asset, specify:
- Asset name (descriptive)
- Channel/placement
- Dimensions (W x H in px)
- Format (PNG/JPG/GIF/MP4/SVG)
- Quantity needed
- Creative direction (1-2 sentences)
- Priority: P1 (launch blocker) / P2 (nice to have) / P3 (optimization)

REQUIRED CHANNELS TO COVER:

### Blog Assets
- Hero image (featured image at top of post)
- In-article graphics or screenshots
- Social share preview image (OG image)

### Twitter/X Assets
- Header card image for thread
- Any data visualization images referenced

### LinkedIn Assets
- Post image or carousel slides
- Company page banner (if applicable)

### Email Assets
- Header image for each email variant
- Any inline images

### Ad/Promotional Assets (for future use)
- Social media ad variants (story format + feed format)

### Miscellaneous
- Favicon / brand mark if needed
- Any animated assets (GIF/Lottie)

Format as a Markdown table for each channel section.
Columns: | Asset Name | Dimensions | Format | Qty | Creative Direction | Priority |

End with a PRODUCTION SUMMARY:
- Total asset count by priority
- Estimated production time (assume 1 designer)
- Recommended production order
""",
        agent=agent,
        expected_output=(
            "A complete asset production list organized by channel, "
            "with dimensions, formats, quantities, creative direction, "
            "and a production summary."
        ),
    )


# ── Main Runner ───────────────────────────────────────────────────────────────


def run_visual_crew(
    campaign_id: str,
    brief: CampaignBrief,
    polished: ContentPackage,
) -> ContentDraft:
    """Generate visual design brief and return as a ContentDraft."""
    log.info("visual_crew_starting", campaign_id=campaign_id)
    start = time.time()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=f"visual_crew_{campaign_id[:8]}") as run:
        mlflow.log_params(
            {
                "campaign_id": campaign_id,
                "title": brief.title,
                "primary_model": settings.llm_model_primary,
                "crew": "visual",
            }
        )

        try:
            strategist = _visual_strategist_agent()
            planner = _asset_planner_agent()

            tasks = [
                _visual_strategy_task(strategist, brief, polished),
                _asset_list_task(planner, brief, polished),
            ]

            crew = Crew(
                agents=[strategist, planner],
                tasks=tasks,
                process=Process.sequential,
                verbose=True,
            )

            result = crew.kickoff()
            raw_output = str(result)
            elapsed = time.time() - start

            # Combine both outputs into one design brief document
            visual_brief_content = _build_brief_document(brief, raw_output)

            visual_draft = ContentDraft(
                campaign_id=campaign_id,
                content_type="visual_brief",
                title=f"Visual Design Brief — {brief.title}",
                content=visual_brief_content,
                metadata={
                    "asset_count": visual_brief_content.count("| P"),
                    "has_color_palette": "#" in visual_brief_content,
                    "version": 1,
                },
            )

            # Save to DB
            save_content_piece(
                campaign_id,
                "visual_brief",
                visual_draft.title,
                visual_draft.content,
                visual_draft.metadata,
            )

            # Log to MLflow
            mlflow.log_metrics(
                {
                    "execution_time_s": elapsed,
                    "brief_length_chars": len(visual_brief_content),
                    "estimated_assets": visual_draft.metadata["asset_count"],
                }
            )
            mlflow.log_text(visual_brief_content, "visual_brief.md")
            mlflow.set_tag("status", "success")

            save_crew_execution(
                campaign_id=campaign_id,
                crew_name="visual_crew",
                status="success",
                mlflow_run_id=run.info.run_id,
                output_data=visual_draft.model_dump(),
                metrics={"execution_time_s": elapsed},
            )

            log.info("visual_crew_complete", campaign_id=campaign_id, elapsed=elapsed)
            return visual_draft

        except Exception as e:
            elapsed = time.time() - start
            mlflow.set_tag("status", "failed")
            mlflow.log_text(str(e), "error.txt")
            save_crew_execution(
                campaign_id=campaign_id,
                crew_name="visual_crew",
                status="failed",
                mlflow_run_id=run.info.run_id,
                output_data={},
                metrics={"execution_time_s": elapsed},
                error=str(e),
            )
            log.error("visual_crew_failed", campaign_id=campaign_id, error=str(e))
            raise


def _build_brief_document(brief: CampaignBrief, raw_output: str) -> str:
    """Wrap the crew output in a clean design brief template."""
    return f"""# Visual Design Brief
## Campaign: {brief.title}

> **Brief:** {brief.brief}
> **Target Audience:** {brief.target_audience or "Business professionals"}
> **Brand Voice:** {brief.brand_voice or "Professional yet approachable"}
> **Keywords:** {", ".join(brief.keywords) if brief.keywords else "See campaign brief"}

---

{raw_output}

---
*Generated by Visual Crew — Content Pipeline*
"""
