"""
Writing Crew
------------
3 specialized writers working in parallel on the same research input:
  - Blog Writer          → 1500-word SEO-optimized blog post (GPT-4o)
  - Social Media Writer  → Twitter/X thread + LinkedIn post (GPT-4o-mini)
  - Email Copywriter     → 3 email variants: awareness, nurture, conversion (GPT-4o-mini)

Manager agent ensures consistent messaging across all formats.
All tasks run in parallel — manager waits for all before finalizing.
MLflow tracks cost + quality metrics.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    ResearchReport,
)

log = structlog.get_logger()


# ── LLM Setup ─────────────────────────────────────────────────────────────────


def _llm_primary():
    return ChatOpenAI(
        model=settings.llm_model_primary,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )


def _llm_secondary():
    return ChatOpenAI(
        model=settings.llm_model_secondary,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )


# ── Agents ────────────────────────────────────────────────────────────────────


def _blog_writer_agent() -> Agent:
    return Agent(
        role="Senior Content Marketing Writer",
        goal=(
            "Write a compelling, SEO-optimized blog post that educates the target audience, "
            "builds trust, and drives organic traffic. The post should be 1400-1600 words."
        ),
        backstory=(
            "You are a former tech journalist turned content marketer with 12 years of experience "
            "writing for publications like TechCrunch and Wired. You have a gift for making complex "
            "technology feel accessible and exciting. You write like a human, not a robot — "
            "with specific examples, surprising insights, and a clear point of view. "
            "You despise generic marketing fluff and always lead with the reader's problem, not the product."
        ),
        llm=_llm_primary(),
        verbose=True,
        max_iter=settings.max_iter_secondary,
    )


def _social_media_agent() -> Agent:
    return Agent(
        role="Social Media Strategist",
        goal=(
            "Create a viral-worthy Twitter/X thread and a high-engagement LinkedIn post "
            "that drive clicks, shares, and conversations."
        ),
        backstory=(
            "You built your personal brand to 50k followers on Twitter and 30k on LinkedIn "
            "by mastering the art of the hook. You know that the first line either stops the scroll "
            "or loses the reader forever. You write in short, punchy sentences. "
            "You use data and counterintuitive insights to grab attention. "
            "You never use corporate-speak or hollow buzzwords like 'synergy' or 'leverage'."
        ),
        llm=_llm_secondary(),
        verbose=True,
        max_iter=settings.max_iter_secondary,
    )


def _email_copywriter_agent() -> Agent:
    return Agent(
        role="Direct Response Email Copywriter",
        goal=(
            "Write 3 email variants for different funnel stages: "
            "awareness (cold), nurture (warm), and conversion (hot). "
            "Each email should have a compelling subject line and clear CTA."
        ),
        backstory=(
            "You trained under the best direct response copywriters and have written emails "
            "that generated millions in revenue. You know that email is the highest-ROI channel "
            "when done right. You obsess over subject lines — you write 20 to pick 1. "
            "You write conversationally, like you're writing to one specific person. "
            "Every email has one job: get the reader to take the next step."
        ),
        llm=_llm_secondary(),
        verbose=True,
        max_iter=settings.max_iter_secondary,
    )


def _writing_manager_agent() -> Agent:
    return Agent(
        role="Content Director",
        goal=(
            "Ensure all content pieces tell a consistent story, reinforce the same key messages, "
            "and match the brand voice perfectly."
        ),
        backstory=(
            "You've led content at three unicorn startups and know how to build a content machine. "
            "You're obsessed with narrative consistency — the blog, social posts, and emails "
            "should feel like they come from the same voice. "
            "You catch inconsistencies, strengthen weak hooks, and make sure every piece "
            "has a clear purpose in the customer journey."
        ),
        llm=_llm_primary(),
        verbose=True,
        allow_delegation=True,
    )


# ── Research Context Helper ───────────────────────────────────────────────────


def _research_context(research: ResearchReport, brief: CampaignBrief) -> str:
    return f"""
CAMPAIGN: {brief.title}
BRIEF: {brief.brief}
BRAND VOICE: {brief.brand_voice or "Professional, clear, and trustworthy"}
TARGET AUDIENCE: {brief.target_audience or "Business professionals"}
KEYWORDS: {", ".join(brief.keywords) if brief.keywords else "Not specified"}

RESEARCH FINDINGS:
Market Size: {research.market_size}

Key Insights:
{chr(10).join(f"- {i}" for i in research.key_insights[:5])}

Top Trends:
{chr(10).join(f"- {t}" for t in research.trends[:3])}

Competitive Context:
{chr(10).join(f"- {c.get('name', '')}: {c.get('description', '')}" for c in research.competitors[:3])}

FULL RESEARCH REPORT:
{research.raw_output[:3000]}
"""


# ── Tasks ─────────────────────────────────────────────────────────────────────


def _blog_post_task(
    agent: Agent, brief: CampaignBrief, research: ResearchReport
) -> Task:
    context = _research_context(research, brief)
    return Task(
        description=f"""
Write a 1400-1600 word SEO-optimized blog post based on the research below.

{context}

REQUIREMENTS:
1. Title: Compelling, includes primary keyword, under 60 characters
2. Structure:
   - Hook (first 2 sentences must stop the scroll)
   - Problem statement (what pain does the reader have?)
   - Main body (3-4 sections with H2 headers)
   - Real examples or data points in each section
   - Conclusion with clear next step
3. SEO: Naturally include these keywords: {", ".join(brief.keywords)}
4. Tone: Match brand voice — {brief.brand_voice or "professional yet approachable"}
5. End with a clear CTA

Format with proper Markdown headers (##, ###).
Write the FULL post — do not summarize or outline.
""",
        agent=agent,
        expected_output=(
            "A complete 1400-1600 word blog post in Markdown format with title, "
            "H2 sections, natural keyword usage, and a CTA."
        ),
    )


def _social_media_task(
    agent: Agent, brief: CampaignBrief, research: ResearchReport
) -> Task:
    context = _research_context(research, brief)
    return Task(
        description=f"""
Create social media content based on the research below.

{context}

DELIVERABLE 1 — Twitter/X Thread (8 tweets):
- Tweet 1: The hook — a bold claim, surprising stat, or counterintuitive insight
- Tweets 2-7: Each tweet expands one key point (max 280 chars each)
- Tweet 8: CTA + link placeholder
- Number each tweet: 1/, 2/, etc.
- Use short sentences. One idea per tweet.

DELIVERABLE 2 — LinkedIn Post (150-300 words):
- Opening line: must stop the scroll
- Body: Tell a mini-story or share a surprising insight from the research
- Use line breaks generously (1-2 sentences per paragraph)
- End with a question to drive comments
- Include 3-5 relevant hashtags

Tone: {brief.brand_voice or "Professional yet conversational"}
Audience: {brief.target_audience or "Business professionals"}

Format clearly with "TWITTER THREAD:" and "LINKEDIN POST:" headers.
""",
        agent=agent,
        expected_output=(
            "A Twitter/X thread with 8 numbered tweets and a LinkedIn post "
            "with hook, body, question, and hashtags."
        ),
    )


def _email_campaign_task(
    agent: Agent, brief: CampaignBrief, research: ResearchReport
) -> Task:
    context = _research_context(research, brief)
    return Task(
        description=f"""
Write 3 email variants for different funnel stages based on the research below.

{context}

EMAIL 1 — AWARENESS (cold outreach / new subscriber):
- Subject line: Curiosity-driven, no selling
- Preview text (40 chars)
- Body: 150-200 words, educate don't sell
- CTA: Soft (read more, learn how)

EMAIL 2 — NURTURE (engaged subscriber / warm lead):
- Subject line: Value-driven, hints at solution
- Preview text (40 chars)
- Body: 200-250 words, build desire
- CTA: Medium (see how it works, watch demo)

EMAIL 3 — CONVERSION (hot lead / trial user):
- Subject line: Urgency or social proof
- Preview text (40 chars)
- Body: 150-200 words, overcome objections
- CTA: Strong (start free trial, get started)

Brand voice: {brief.brand_voice or "Professional, clear, direct"}
Audience: {brief.target_audience or "Business decision makers"}

Format with "EMAIL 1:", "EMAIL 2:", "EMAIL 3:" headers.
Include SUBJECT:, PREVIEW:, BODY:, CTA: labels for each.
""",
        agent=agent,
        expected_output=(
            "3 complete emails with subject lines, preview text, body copy, "
            "and CTAs for awareness, nurture, and conversion stages."
        ),
    )


# ── Parallel Execution ────────────────────────────────────────────────────────


def _run_single_crew(crew_fn, name: str) -> tuple[str, str]:
    """Run a single-agent crew and return (name, output)."""
    try:
        result = crew_fn()
        return name, str(result)
    except Exception as e:
        log.error(f"{name}_failed", error=str(e))
        return name, f"ERROR: {e}"


def _run_blog_crew(brief: CampaignBrief, research: ResearchReport) -> str:
    agent = _blog_writer_agent()
    task = _blog_post_task(agent, brief, research)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    return str(crew.kickoff())


def _run_social_crew(brief: CampaignBrief, research: ResearchReport) -> str:
    agent = _social_media_agent()
    task = _social_media_task(agent, brief, research)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    return str(crew.kickoff())


def _run_email_crew(brief: CampaignBrief, research: ResearchReport) -> str:
    agent = _email_copywriter_agent()
    task = _email_campaign_task(agent, brief, research)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    return str(crew.kickoff())


# ── Output Parsers ────────────────────────────────────────────────────────────


def _parse_blog(campaign_id: str, raw: str) -> ContentDraft:
    lines = raw.strip().split("\n")
    title = ""
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break
    if not title:
        title = "Blog Post"
    return ContentDraft(
        campaign_id=campaign_id,
        content_type="blog",
        title=title,
        content=raw,
        metadata={"word_count": len(raw.split())},
    )


def _parse_social(campaign_id: str, raw: str) -> tuple[ContentDraft, ContentDraft]:
    twitter_content = ""
    linkedin_content = ""

    if "TWITTER THREAD:" in raw and "LINKEDIN POST:" in raw:
        parts = raw.split("LINKEDIN POST:")
        twitter_content = parts[0].replace("TWITTER THREAD:", "").strip()
        linkedin_content = parts[1].strip()
    elif "LINKEDIN POST:" in raw:
        parts = raw.split("LINKEDIN POST:")
        twitter_content = parts[0].strip()
        linkedin_content = parts[1].strip()
    else:
        twitter_content = raw
        linkedin_content = raw

    twitter = ContentDraft(
        campaign_id=campaign_id,
        content_type="twitter_thread",
        title="Twitter/X Thread",
        content=twitter_content,
        metadata={"tweet_count": twitter_content.count("\n1/") + 1},
    )
    linkedin = ContentDraft(
        campaign_id=campaign_id,
        content_type="linkedin",
        title="LinkedIn Post",
        content=linkedin_content,
        metadata={"char_count": len(linkedin_content)},
    )
    return twitter, linkedin


def _parse_emails(campaign_id: str, raw: str) -> list[ContentDraft]:
    emails = []
    stages = [
        ("EMAIL 1:", "awareness"),
        ("EMAIL 2:", "nurture"),
        ("EMAIL 3:", "conversion"),
    ]

    for i, (marker, stage) in enumerate(stages):
        if marker in raw:
            next_marker = stages[i + 1][0] if i + 1 < len(stages) else None
            if next_marker and next_marker in raw:
                content = raw.split(marker)[1].split(next_marker)[0].strip()
            else:
                content = raw.split(marker)[1].strip() if marker in raw else ""

            subject = ""
            for line in content.split("\n"):
                if line.startswith("SUBJECT:"):
                    subject = line.replace("SUBJECT:", "").strip()
                    break

            emails.append(
                ContentDraft(
                    campaign_id=campaign_id,
                    content_type="email",
                    title=f"Email — {stage.title()} ({subject[:50]})"
                    if subject
                    else f"Email — {stage.title()}",
                    content=content,
                    metadata={"stage": stage, "subject": subject},
                )
            )

    if not emails:
        emails.append(
            ContentDraft(
                campaign_id=campaign_id,
                content_type="email",
                title="Email Campaign",
                content=raw,
                metadata={"stage": "all"},
            )
        )

    return emails


# ── Main Runner ───────────────────────────────────────────────────────────────


def run_writing_crew(
    campaign_id: str,
    brief: CampaignBrief,
    research: ResearchReport,
) -> ContentPackage:
    """Run blog, social, and email writers in parallel and return a ContentPackage."""
    log.info("writing_crew_starting", campaign_id=campaign_id)
    start = time.time()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=f"writing_crew_{campaign_id[:8]}") as run:
        mlflow.log_params(
            {
                "campaign_id": campaign_id,
                "title": brief.title,
                "primary_model": settings.llm_model_primary,
                "secondary_model": settings.llm_model_secondary,
                "crew": "writing",
            }
        )

        try:
            # Run all 3 writers in parallel
            results = {}
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(_run_blog_crew, brief, research): "blog",
                    executor.submit(_run_social_crew, brief, research): "social",
                    executor.submit(_run_email_crew, brief, research): "email",
                }
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        results[name] = future.result()
                        log.info(f"{name}_writer_complete", campaign_id=campaign_id)
                    except Exception as e:
                        results[name] = f"ERROR: {e}"
                        log.error(f"{name}_writer_failed", error=str(e))

            # Parse outputs
            blog_draft = _parse_blog(campaign_id, results.get("blog", ""))
            twitter, linkedin = _parse_social(campaign_id, results.get("social", ""))
            email_drafts = _parse_emails(campaign_id, results.get("email", ""))

            # Save to DB
            save_content_piece(
                campaign_id,
                "blog",
                blog_draft.title,
                blog_draft.content,
                blog_draft.metadata,
            )
            save_content_piece(
                campaign_id,
                "twitter_thread",
                twitter.title,
                twitter.content,
                twitter.metadata,
            )
            save_content_piece(
                campaign_id,
                "linkedin",
                linkedin.title,
                linkedin.content,
                linkedin.metadata,
            )
            for email in email_drafts:
                save_content_piece(
                    campaign_id, "email", email.title, email.content, email.metadata
                )

            elapsed = time.time() - start
            package = ContentPackage(
                campaign_id=campaign_id,
                blog_post=blog_draft,
                twitter_thread=twitter,
                linkedin_post=linkedin,
                email_variants=email_drafts,
            )

            # Log metrics
            mlflow.log_metrics(
                {
                    "execution_time_s": elapsed,
                    "blog_word_count": blog_draft.metadata.get("word_count", 0),
                    "email_variants": len(email_drafts),
                }
            )
            mlflow.log_text(blog_draft.content, "blog_post.md")
            mlflow.log_text(twitter.content, "twitter_thread.txt")
            mlflow.log_text(linkedin.content, "linkedin_post.txt")
            mlflow.log_text(results.get("email", ""), "emails.txt")
            mlflow.set_tag("status", "success")

            save_crew_execution(
                campaign_id=campaign_id,
                crew_name="writing_crew",
                status="success",
                mlflow_run_id=run.info.run_id,
                output_data=package.model_dump(),
                metrics={"execution_time_s": elapsed},
            )

            log.info("writing_crew_complete", campaign_id=campaign_id, elapsed=elapsed)
            return package

        except Exception as e:
            elapsed = time.time() - start
            mlflow.set_tag("status", "failed")
            mlflow.log_text(str(e), "error.txt")
            save_crew_execution(
                campaign_id=campaign_id,
                crew_name="writing_crew",
                status="failed",
                mlflow_run_id=run.info.run_id,
                output_data={},
                metrics={"execution_time_s": elapsed},
                error=str(e),
            )
            log.error("writing_crew_failed", campaign_id=campaign_id, error=str(e))
            raise
