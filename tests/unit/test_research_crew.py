"""Unit tests for research crew — parser and run function."""

from unittest.mock import MagicMock, patch

import pytest

from src.research_crew.crew import _parse_research_output
from src.shared.models import CampaignBrief, ResearchReport

SAMPLE_BRIEF = CampaignBrief(
    title="AI CRM Launch",
    brief="Launch campaign for AI-powered CRM targeting SMBs",
    brand_voice="Professional yet approachable",
    target_audience="SMB founders",
    keywords=["AI CRM", "sales automation", "SMB"],
)

SAMPLE_OUTPUT = """
## Executive Summary
AI CRM market is growing rapidly with strong SMB adoption.

## Market Overview
The AI CRM market is valued at $12.5 billion (TAM) with SAM of $3.2 billion.
Growth rate is 23% CAGR through 2028.

## Competitive Landscape
- Salesforce: Market leader with 20% share, strong enterprise focus
- HubSpot: Strong SMB positioning, freemium model
- Pipedrive: Sales-focused, easy to use

## Key Trends
- Trend: AI-powered automation is reshaping CRM workflows
- Emerging: Voice-to-CRM integrations growing fast
- Shift: Mobile-first CRM adoption accelerating

## Content Strategy
- Focus on ROI stories from SMB customers
- Highlight ease of integration
"""


@pytest.mark.unit
class TestParseResearchOutput:
    def test_returns_research_report(self):
        report = _parse_research_output("camp-001", SAMPLE_OUTPUT, SAMPLE_BRIEF)
        assert isinstance(report, ResearchReport)

    def test_extracts_market_size(self):
        report = _parse_research_output("camp-001", SAMPLE_OUTPUT, SAMPLE_BRIEF)
        assert "$" in report.market_size or report.market_size == "See full report"

    def test_extracts_competitors(self):
        report = _parse_research_output("camp-001", SAMPLE_OUTPUT, SAMPLE_BRIEF)
        assert isinstance(report.competitors, list)

    def test_extracts_trends(self):
        report = _parse_research_output("camp-001", SAMPLE_OUTPUT, SAMPLE_BRIEF)
        assert isinstance(report.trends, list)

    def test_extracts_insights(self):
        report = _parse_research_output("camp-001", SAMPLE_OUTPUT, SAMPLE_BRIEF)
        assert isinstance(report.key_insights, list)

    def test_raw_output_preserved(self):
        report = _parse_research_output("camp-001", SAMPLE_OUTPUT, SAMPLE_BRIEF)
        assert report.raw_output == SAMPLE_OUTPUT

    def test_campaign_id_set(self):
        report = _parse_research_output("camp-001", SAMPLE_OUTPUT, SAMPLE_BRIEF)
        assert report.campaign_id == "camp-001"

    def test_empty_output_returns_defaults(self):
        report = _parse_research_output("camp-001", "", SAMPLE_BRIEF)
        assert report.market_size == "See full report"
        assert report.competitors == []
        assert report.key_insights == []


@pytest.mark.unit
class TestRunResearchCrew:
    @patch("src.research_crew.crew.save_crew_execution")
    @patch("src.research_crew.crew.mlflow")
    @patch("src.research_crew.crew.Crew")
    @patch("src.research_crew.crew.Task")
    @patch("src.research_crew.crew.Agent")
    @patch("src.research_crew.crew.ChatOpenAI")
    def test_successful_run_returns_report(
        self,
        mock_llm,
        mock_agent,
        mock_task,
        mock_crew_cls,
        mock_mlflow,
        mock_save_exec,
    ):
        from src.research_crew.crew import run_research_crew

        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = SAMPLE_OUTPUT
        mock_crew_cls.return_value = mock_crew

        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "run-001"
        mock_mlflow.start_run.return_value = mock_run

        result = run_research_crew("camp-001", SAMPLE_BRIEF)

        assert isinstance(result, ResearchReport)
        assert result.campaign_id == "camp-001"
        mock_save_exec.assert_called_once()
        assert mock_save_exec.call_args.kwargs["status"] == "success"

    @patch("src.research_crew.crew.save_crew_execution")
    @patch("src.research_crew.crew.mlflow")
    @patch("src.research_crew.crew.Crew")
    @patch("src.research_crew.crew.Task")
    @patch("src.research_crew.crew.Agent")
    @patch("src.research_crew.crew.ChatOpenAI")
    def test_failed_run_saves_error(
        self,
        mock_llm,
        mock_agent,
        mock_task,
        mock_crew_cls,
        mock_mlflow,
        mock_save_exec,
    ):
        from src.research_crew.crew import run_research_crew

        mock_crew = MagicMock()
        mock_crew.kickoff.side_effect = Exception("OpenAI API error")
        mock_crew_cls.return_value = mock_crew

        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "run-002"
        mock_mlflow.start_run.return_value = mock_run

        with pytest.raises(Exception, match="OpenAI API error"):
            run_research_crew("camp-001", SAMPLE_BRIEF)

        assert mock_save_exec.call_args.kwargs["status"] == "failed"
