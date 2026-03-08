"""
Research Crew
-------------
Hierarchical CrewAI crew with 3 specialized agents:
  - Market Research Agent  → TAM/SAM, market size, growth rates
  - Competitor Analysis Agent → positioning, strengths, weaknesses
  - Trend Scout Agent      → recent trends via Tavily web search

Manager agent synthesizes findings into a structured ResearchReport.
MLflow tracks the run with cost + timing metrics.
"""

import time

import mlflow
import structlog
from crewai import Agent, Crew, Process, Task
from langchain_openai import ChatOpenAI

from src.shared.config import settings
from src.shared.db import save_crew_execution
from src.shared.models import CampaignBrief, ResearchReport

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


# ── Tools ─────────────────────────────────────────────────────────────────────


def _search_tool():
    from crewai.tools import BaseTool

    class DuckDuckGoTool(BaseTool):
        name: str = "DuckDuckGo Search"
        description: str = (
            "Search the web using DuckDuckGo. Input should be a search query string."
        )

        def _run(self, query: str) -> str:
            from ddgs import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5):
                    results.append(f"{r['title']}: {r['body']} ({r['href']})")
            return "\n".join(results) if results else "No results found."

    return DuckDuckGoTool()


# ── Agents ────────────────────────────────────────────────────────────────────


def _market_research_agent() -> Agent:
    return Agent(
        role="Senior Market Research Analyst",
        goal=(
            "Find accurate, data-driven market size, TAM/SAM figures, "
            "growth rates, and key market dynamics for the given industry."
        ),
        backstory=(
            "You are a seasoned market research analyst with 15 years of experience "
            "at top consulting firms. You specialize in technology markets and always "
            "back your findings with concrete numbers and credible sources. "
            "You despise vague claims — every statement needs a figure or a source."
        ),
        tools=[_search_tool()],
        llm=_llm_primary(),
        verbose=True,
        max_iter=settings.max_iter_primary,
    )


def _competitor_analysis_agent() -> Agent:
    return Agent(
        role="Competitive Intelligence Specialist",
        goal=(
            "Identify and analyze the top 3-5 competitors: their positioning, "
            "pricing, strengths, weaknesses, and messaging strategy."
        ),
        backstory=(
            "You worked as a product strategist at multiple SaaS startups. "
            "You have a gift for dissecting competitor strategies and identifying "
            "whitespace opportunities. You think in terms of positioning maps, "
            "differentiation, and customer pain points. "
            "You always look for what competitors are NOT saying."
        ),
        tools=[_search_tool()],
        llm=_llm_secondary(),
        verbose=True,
        max_iter=settings.max_iter_primary,
    )


def _trend_scout_agent() -> Agent:
    return Agent(
        role="Industry Trend Scout",
        goal=(
            "Identify the 5 most relevant emerging trends in the last 6 months "
            "that will impact this market. Focus on technology shifts, "
            "behavioral changes, and regulatory developments."
        ),
        backstory=(
            "You are obsessed with the future. You read hundreds of industry "
            "newsletters, follow thought leaders, and attend every major conference. "
            "Your superpower is connecting weak signals into strong trend narratives "
            "before they become mainstream. "
            "You focus on trends that have real business implications, not hype."
        ),
        tools=[_search_tool()],
        llm=_llm_secondary(),
        verbose=True,
        max_iter=settings.max_iter_primary,
    )


def _manager_agent() -> Agent:
    return Agent(
        role="Research Director",
        goal=(
            "Synthesize findings from all research agents into a coherent, "
            "actionable research report that directly informs content strategy."
        ),
        backstory=(
            "You are a former McKinsey partner who now leads research at a top "
            "content marketing agency. You excel at synthesizing diverse information "
            "into clear strategic narratives. "
            "You know that great content starts with great research, "
            "and great research starts with the right questions."
        ),
        llm=_llm_primary(),
        verbose=True,
        allow_delegation=True,
    )


# ── Tasks ─────────────────────────────────────────────────────────────────────


def _market_research_task(agent: Agent, brief: CampaignBrief) -> Task:
    return Task(
        description=f"""
Research the market for: {brief.title}

Brief context: {brief.brief}
Target audience: {brief.target_audience or "Not specified"}
Keywords to focus on: {", ".join(brief.keywords) if brief.keywords else "Not specified"}

Your deliverables:
1. Market size (TAM and SAM with figures and sources)
2. Market growth rate (CAGR, YoY growth)
3. Key market segments
4. Top 3 market drivers
5. Top 3 market challenges

Format as structured text with clear sections and numbers.
""",
        agent=agent,
        expected_output=(
            "A structured market research report with TAM/SAM figures, "
            "growth rates, segments, drivers, and challenges — all with sources."
        ),
    )


def _competitor_analysis_task(agent: Agent, brief: CampaignBrief) -> Task:
    return Task(
        description=f"""
Analyze competitors for: {brief.title}

Brief context: {brief.brief}
Keywords: {", ".join(brief.keywords) if brief.keywords else "Not specified"}

Your deliverables:
1. Top 3-5 direct competitors (name, website, positioning statement)
2. For each competitor:
   - Core value proposition
   - Target customer
   - Key strengths (2-3)
   - Key weaknesses (2-3)
   - Pricing model (if available)
3. Competitive whitespace: What gap exists that our product can fill?
4. Key messaging themes competitors use

Be specific. Use actual competitor names and real data.
""",
        agent=agent,
        expected_output=(
            "A competitor analysis with 3-5 competitors profiled, "
            "including strengths, weaknesses, and identified market gaps."
        ),
    )


def _trend_scout_task(agent: Agent, brief: CampaignBrief) -> Task:
    return Task(
        description=f"""
Identify emerging trends relevant to: {brief.title}

Brief context: {brief.brief}
Target audience: {brief.target_audience or "Not specified"}

Your deliverables:
1. Top 5 emerging trends (last 6 months) with:
   - Trend name
   - Why it matters (business impact)
   - Evidence/examples
   - Implication for content strategy
2. One contrarian insight: what trend is overhyped?
3. One underrated opportunity most marketers are missing

Focus on trends with real business implications, not buzzwords.
""",
        agent=agent,
        expected_output=(
            "5 emerging trends with evidence and content implications, "
            "plus one contrarian insight and one underrated opportunity."
        ),
    )


def _synthesis_task(agent: Agent, brief: CampaignBrief) -> Task:
    return Task(
        description=f"""
Synthesize all research findings for campaign: {brief.title}

Using the outputs from the market research, competitor analysis, and trend scout agents,
create a final research report that:

1. Executive Summary (3-4 sentences) — the most important findings
2. Market Overview — size, growth, key dynamics
3. Competitive Landscape — top competitors and our differentiation opportunity
4. Key Trends — top 3 most relevant trends for this campaign
5. Target Audience Insights — who they are, what they care about, what language resonates
6. Content Strategy Implications — 5 specific angles/hooks for content creation
7. Key Messages — 3 core messages our content should reinforce

Brief context: {brief.brief}
Brand voice: {brief.brand_voice or "Not specified"}
Keywords: {", ".join(brief.keywords) if brief.keywords else "Not specified"}

This report will be handed directly to the writing crew — make it actionable.
""",
        agent=agent,
        expected_output=(
            "A complete research report with executive summary, market overview, "
            "competitive analysis, trends, audience insights, content angles, "
            "and key messages — ready for the writing crew."
        ),
    )


# ── Crew Runner ───────────────────────────────────────────────────────────────


def run_research_crew(campaign_id: str, brief: CampaignBrief) -> ResearchReport:
    """Run the research crew and return a structured ResearchReport."""
    log.info("research_crew_starting", campaign_id=campaign_id, title=brief.title)
    start = time.time()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=f"research_crew_{campaign_id[:8]}") as run:
        mlflow.log_params(
            {
                "campaign_id": campaign_id,
                "title": brief.title,
                "primary_model": settings.llm_model_primary,
                "secondary_model": settings.llm_model_secondary,
                "crew": "research",
            }
        )

        try:
            # Build agents
            market_agent = _market_research_agent()
            competitor_agent = _competitor_analysis_agent()
            trend_agent = _trend_scout_agent()
            manager_agent = _manager_agent()

            # Build tasks
            tasks = [
                _market_research_task(market_agent, brief),
                _competitor_analysis_task(competitor_agent, brief),
                _trend_scout_task(trend_agent, brief),
                _synthesis_task(manager_agent, brief),
            ]

            # Build and run crew
            crew = Crew(
                agents=[market_agent, competitor_agent, trend_agent, manager_agent],
                tasks=tasks,
                process=Process.sequential,
                verbose=True,
                manager_llm=_llm_primary(),
            )

            result = crew.kickoff()
            elapsed = time.time() - start
            raw_output = str(result)

            # Parse structured data from the synthesis output
            report = _parse_research_output(campaign_id, raw_output, brief)

            # Log metrics
            mlflow.log_metrics(
                {
                    "execution_time_s": elapsed,
                    "num_insights": len(report.key_insights),
                    "num_competitors": len(report.competitors),
                    "num_trends": len(report.trends),
                }
            )
            mlflow.log_text(raw_output, "research_report.txt")
            mlflow.set_tag("status", "success")

            save_crew_execution(
                campaign_id=campaign_id,
                crew_name="research_crew",
                status="success",
                mlflow_run_id=run.info.run_id,
                output_data=report.model_dump(),
                metrics={"execution_time_s": elapsed},
            )

            log.info("research_crew_complete", campaign_id=campaign_id, elapsed=elapsed)
            return report

        except Exception as e:
            elapsed = time.time() - start
            mlflow.set_tag("status", "failed")
            mlflow.log_text(str(e), "error.txt")

            save_crew_execution(
                campaign_id=campaign_id,
                crew_name="research_crew",
                status="failed",
                mlflow_run_id=run.info.run_id,
                output_data={},
                metrics={"execution_time_s": elapsed},
                error=str(e),
            )
            log.error("research_crew_failed", campaign_id=campaign_id, error=str(e))
            raise


def _parse_research_output(
    campaign_id: str, raw_output: str, brief: CampaignBrief
) -> ResearchReport:
    """Extract structured fields from the crew's text output."""
    lines = raw_output.split("\n")

    # Extract key insights — lines that look like bullet points
    insights = []
    competitors = []
    trends = []
    market_size = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Market size — look for $ figures
        if "$" in line and (
            "billion" in line.lower()
            or "million" in line.lower()
            or "TAM" in line
            or "SAM" in line
        ):
            if not market_size:
                market_size = line

        # Bullet points → insights
        if line.startswith(("- ", "• ", "* ", "→ ")):
            content = line[2:].strip()
            if len(content) > 20:
                insights.append(content)

        # Numbered items that mention competitors
        if any(
            comp in line
            for comp in ["Salesforce", "HubSpot", "Pipedrive", "Zoho", "Monday"]
        ):
            competitors.append({"name": line[:50], "description": line})

        # Trend lines
        if any(
            word in line.lower() for word in ["trend", "emerging", "growing", "shift"]
        ):
            if len(line) > 20:
                trends.append(line)

    return ResearchReport(
        campaign_id=campaign_id,
        market_size=market_size or "See full report",
        competitors=competitors[:5],
        trends=trends[:5],
        key_insights=insights[:10],
        raw_output=raw_output,
    )
