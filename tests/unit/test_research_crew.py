"""Unit tests for research crew — all LLM/API calls mocked."""

from unittest.mock import MagicMock, patch

import pytest

from src.research_crew.crew import _parse_research_output
from src.shared.models import CampaignBrief, ResearchReport

SAMPLE_BRIEF = CampaignBrief(
    title="AI CRM Launch",
    brief="Launch campaign for new AI-powered CRM targeting SMBs",
    brand_voice="Professional yet approachable",
    target_audience="SMB founders and sales managers",
    keywords=["AI CRM", "sales automation", "SMB"],
)

SAMPLE_OUTPUT = """
Executive Summary
The AI CRM market is growing rapidly, with significant opportunity for SMB-focused players.

Market Overview
- The global CRM market is valued at $65 billion TAM with 12% CAGR
- SAM for AI-powered CRM tools is approximately $8 billion
- SMB segment represents the fastest growing subsegment

Competitive Landscape
- Salesforce dominates enterprise but weak in SMB UX
- HubSpot strong in marketing but limited AI features
- Pipedrive focused on sales pipeline, limited automation

Key Trends
- Emerging trend: AI-first CRM adoption growing 40% YoY
- Shift toward conversational AI in sales workflows
- Growing demand for no-code automation tools

Content Strategy
- Focus on ROI and time savings for busy SMB founders
- Use case stories resonate better than feature lists
"""


@pytest.mark.unit
class TestParseResearchOutput:
    def test_extracts_market_size(self):
        report = _parse_research_output("test-id", SAMPLE_OUTPUT, SAMPLE_BRIEF)
        assert report.campaign_id == "test-id"
        assert "$" in report.market_size or report.market_size == "See full report"

    def test_extracts_insights_from_bullets(self):
        report = _parse_research_output("test-id", SAMPLE_OUTPUT, SAMPLE_BRIEF)
        assert len(report.key_insights) > 0

    def test_extracts_competitors(self):
        report = _parse_research_output("test-id", SAMPLE_OUTPUT, SAMPLE_BRIEF)
        assert len(report.competitors) > 0
        assert any("Salesforce" in c["name"] or "HubSpot" in c["name"] for c in report.competitors)

    def test_extracts_trends(self):
        report = _parse_research_output("test-id", SAMPLE_OUTPUT, SAMPLE_BRIEF)
        assert len(report.trends) > 0

    def test_raw_output_preserved(self):
        report = _parse_research_output("test-id", SAMPLE_OUTPUT, SAMPLE_BRIEF)
        assert report.raw_output == SAMPLE_OUTPUT

    def test_empty_output_returns_defaults(self):
        report = _parse_research_output("test-id", "", SAMPLE_BRIEF)
        assert report.campaign_id == "test-id"
        assert report.market_size == "See full report"
        assert report.key_insights == []

    def test_limits_insights_to_ten(self):
        long_output = "\n".join([f"- Insight number {i} with enough detail here" for i in range(20)])
        report = _parse_research_output("test-id", long_output, SAMPLE_BRIEF)
        assert len(report.key_insights) <= 10

    def test_limits_competitors_to_five(self):
        report = _parse_research_output("test-id", SAMPLE_OUTPUT, SAMPLE_BRIEF)
        assert len(report.competitors) <= 5


@pytest.mark.unit
class TestResearchCrewAgents:
    def test_agents_have_required_fields(self):
        from src.research_crew.crew import (
            _competitor_analysis_agent,
            _manager_agent,
            _market_research_agent,
            _trend_scout_agent,
        )

        with (
            patch("src.research_crew.crew.ChatOpenAI"),
            patch("src.research_crew.crew.DuckDuckGoSearchRun", create=True),
        ):
            market = _market_research_agent()
            assert market.role != ""
            assert market.goal != ""
            assert market.backstory != ""

            competitor = _competitor_analysis_agent()
            assert competitor.role != ""

            trend = _trend_scout_agent()
            assert trend.role != ""

            manager = _manager_agent()
            assert manager.allow_delegation is True

    def test_tasks_include_brief_context(self):
        from src.research_crew.crew import (
            _competitor_analysis_task,
            _market_research_task,
        )

        with (
            patch("src.research_crew.crew.ChatOpenAI"),
            patch("src.research_crew.crew.DuckDuckGoSearchRun", create=True),
        ):
            from src.research_crew.crew import _market_research_agent

            agent = _market_research_agent()

            task = _market_research_task(agent, SAMPLE_BRIEF)
            assert "AI CRM Launch" in task.description
            assert "SMB" in task.description

            task2 = _competitor_analysis_task(agent, SAMPLE_BRIEF)
            assert "AI CRM Launch" in task2.description


@pytest.mark.unit
class TestRunResearchCrew:
    @patch("src.research_crew.crew.save_crew_execution")
    @patch("src.research_crew.crew.mlflow")
    @patch("src.research_crew.crew.Crew")
    @patch("src.research_crew.crew.DuckDuckGoSearchRun", create=True)
    @patch("src.research_crew.crew.ChatOpenAI")
    def test_successful_run_returns_report(self, mock_llm, mock_tavily, mock_crew_cls, mock_mlflow, mock_save):
        from src.research_crew.crew import run_research_crew

        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = SAMPLE_OUTPUT
        mock_crew_cls.return_value = mock_crew

        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value = mock_run

        report = run_research_crew("campaign-001", SAMPLE_BRIEF)

        assert isinstance(report, ResearchReport)
        assert report.campaign_id == "campaign-001"
        mock_save.assert_called_once()
        assert mock_save.call_args.kwargs["status"] == "success"

    @patch("src.research_crew.crew.save_crew_execution")
    @patch("src.research_crew.crew.mlflow")
    @patch("src.research_crew.crew.Crew")
    @patch("src.research_crew.crew.DuckDuckGoSearchRun", create=True)
    @patch("src.research_crew.crew.ChatOpenAI")
    def test_failed_run_saves_error(self, mock_llm, mock_tavily, mock_crew_cls, mock_mlflow, mock_save):
        from src.research_crew.crew import run_research_crew

        mock_crew = MagicMock()
        mock_crew.kickoff.side_effect = Exception("OpenAI API error")
        mock_crew_cls.return_value = mock_crew

        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value = mock_run

        with pytest.raises(Exception, match="OpenAI API error"):
            run_research_crew("campaign-001", SAMPLE_BRIEF)

        mock_save.assert_called_once()
        assert mock_save.call_args.kwargs["status"] == "failed"
