"""Unit tests for writing crew — all LLM/API calls mocked."""

from unittest.mock import MagicMock, patch

import pytest

from src.shared.models import CampaignBrief, ContentPackage, ResearchReport
from src.writing_crew.crew import (
    _parse_blog,
    _parse_emails,
    _parse_social,
    _research_context,
)

SAMPLE_BRIEF = CampaignBrief(
    title="AI CRM Launch",
    brief="Launch campaign for new AI-powered CRM targeting SMBs",
    brand_voice="Professional yet approachable",
    target_audience="SMB founders and sales managers",
    keywords=["AI CRM", "sales automation", "SMB"],
)

SAMPLE_RESEARCH = ResearchReport(
    campaign_id="test-123",
    market_size="$65B TAM, 12% CAGR",
    competitors=[{"name": "Salesforce", "description": "Enterprise CRM leader"}],
    trends=["AI-first CRM adoption", "No-code automation"],
    key_insights=["SMB segment fastest growing", "Buyers prefer self-serve"],
    raw_output="Full research report text here...",
)

SAMPLE_BLOG = """# How AI CRM Is Transforming SMB Sales in 2026

## The Problem Every SMB Sales Team Faces

Most small business sales teams are drowning in manual data entry.

## Why AI Changes Everything

AI CRM tools reduce admin time by 40%.

### Real Results from Real Teams

Companies using AI CRM close 25% more deals.

## Getting Started

Start with a free trial today.

## Conclusion

The future of SMB sales is AI-powered. Don't get left behind.

[CTA: Start your free trial]
"""

SAMPLE_SOCIAL = """TWITTER THREAD:
1/ Most SMB sales teams waste 3 hours/day on data entry. AI CRM fixes this.

2/ The global CRM market hit $65B. SMBs are the fastest-growing segment.

3/ Top competitors ignore SMBs. That's your opportunity.

4/ AI CRM reduces admin time by 40%. More time selling.

5/ No-code automation means setup in hours, not months.

6/ Early adopters close 25% more deals in year one.

7/ The best part? It gets smarter the more you use it.

8/ Ready to transform your sales process? Link in bio.

LINKEDIN POST:
I spent 10 years watching SMB sales teams struggle with the same problem.

Too much admin. Not enough selling.

AI CRM changes that equation completely.

Here's what the data shows:
- 40% reduction in admin time
- 25% more deals closed
- Setup in hours, not months

What's your biggest sales challenge right now?

#AICRM #SalesAutomation #SMB #SalesTech
"""

SAMPLE_EMAILS = """EMAIL 1:
SUBJECT: The CRM problem nobody talks about
PREVIEW: Most sales teams waste 3hrs/day on this
BODY: Most SMB sales teams spend more time updating their CRM than actually selling...
CTA: Read the full breakdown →

EMAIL 2:
SUBJECT: How [Company] cut admin time by 40%
PREVIEW: Real results from a real SMB team
BODY: Last quarter, a 12-person sales team using AI CRM closed 25% more deals...
CTA: See how it works →

EMAIL 3:
SUBJECT: Your free trial expires in 48 hours
PREVIEW: Don't lose your progress
BODY: You've seen what AI CRM can do. Now it's time to make it official...
CTA: Activate your account now →
"""


@pytest.mark.unit
class TestParseBlog:
    def test_extracts_title_from_h1(self):
        draft = _parse_blog("test-123", SAMPLE_BLOG)
        assert draft.title == "How AI CRM Is Transforming SMB Sales in 2026"

    def test_fallback_title_when_no_h1(self):
        draft = _parse_blog("test-123", "No heading here\nJust content")
        assert draft.title == "Blog Post"

    def test_content_type_is_blog(self):
        draft = _parse_blog("test-123", SAMPLE_BLOG)
        assert draft.content_type == "blog"

    def test_word_count_in_metadata(self):
        draft = _parse_blog("test-123", SAMPLE_BLOG)
        assert draft.metadata["word_count"] > 0

    def test_full_content_preserved(self):
        draft = _parse_blog("test-123", SAMPLE_BLOG)
        assert draft.content == SAMPLE_BLOG

    def test_campaign_id_set(self):
        draft = _parse_blog("camp-999", SAMPLE_BLOG)
        assert draft.campaign_id == "camp-999"


@pytest.mark.unit
class TestParseSocial:
    def test_splits_twitter_and_linkedin(self):
        twitter, linkedin = _parse_social("test-123", SAMPLE_SOCIAL)
        assert twitter.content_type == "twitter_thread"
        assert linkedin.content_type == "linkedin"

    def test_twitter_content_extracted(self):
        twitter, _ = _parse_social("test-123", SAMPLE_SOCIAL)
        assert "1/" in twitter.content
        assert "LINKEDIN POST:" not in twitter.content

    def test_linkedin_content_extracted(self):
        _, linkedin = _parse_social("test-123", SAMPLE_SOCIAL)
        assert "#AICRM" in linkedin.content
        assert "TWITTER THREAD:" not in linkedin.content

    def test_fallback_when_no_markers(self):
        twitter, linkedin = _parse_social("test-123", "Some generic social content")
        assert twitter.content == "Some generic social content"

    def test_campaign_id_set_on_both(self):
        twitter, linkedin = _parse_social("camp-42", SAMPLE_SOCIAL)
        assert twitter.campaign_id == "camp-42"
        assert linkedin.campaign_id == "camp-42"


@pytest.mark.unit
class TestParseEmails:
    def test_returns_three_emails(self):
        emails = _parse_emails("test-123", SAMPLE_EMAILS)
        assert len(emails) == 3

    def test_email_stages(self):
        emails = _parse_emails("test-123", SAMPLE_EMAILS)
        stages = [e.metadata["stage"] for e in emails]
        assert "awareness" in stages
        assert "nurture" in stages
        assert "conversion" in stages

    def test_subject_lines_extracted(self):
        emails = _parse_emails("test-123", SAMPLE_EMAILS)
        awareness = next(e for e in emails if e.metadata["stage"] == "awareness")
        assert "CRM problem" in awareness.metadata["subject"]

    def test_content_type_is_email(self):
        emails = _parse_emails("test-123", SAMPLE_EMAILS)
        assert all(e.content_type == "email" for e in emails)

    def test_fallback_on_no_markers(self):
        emails = _parse_emails("test-123", "Just some email content with no markers")
        assert len(emails) == 1
        assert emails[0].metadata["stage"] == "all"


@pytest.mark.unit
class TestResearchContext:
    def test_includes_campaign_title(self):
        ctx = _research_context(SAMPLE_RESEARCH, SAMPLE_BRIEF)
        assert "AI CRM Launch" in ctx

    def test_includes_market_size(self):
        ctx = _research_context(SAMPLE_RESEARCH, SAMPLE_BRIEF)
        assert "$65B TAM" in ctx

    def test_includes_keywords(self):
        ctx = _research_context(SAMPLE_RESEARCH, SAMPLE_BRIEF)
        assert "AI CRM" in ctx

    def test_includes_insights(self):
        ctx = _research_context(SAMPLE_RESEARCH, SAMPLE_BRIEF)
        assert "SMB segment fastest growing" in ctx

    def test_handles_empty_research(self):
        empty = ResearchReport(campaign_id="x")
        ctx = _research_context(empty, SAMPLE_BRIEF)
        assert "AI CRM Launch" in ctx


@pytest.mark.unit
class TestRunWritingCrew:
    @patch("src.writing_crew.crew.save_crew_execution")
    @patch("src.writing_crew.crew.save_content_piece")
    @patch("src.writing_crew.crew.mlflow")
    @patch("src.writing_crew.crew.Crew")
    @patch("src.writing_crew.crew.ChatOpenAI")
    def test_successful_run_returns_package(
        self, mock_llm, mock_crew_cls, mock_mlflow, mock_save_piece, mock_save_exec
    ):
        from src.writing_crew.crew import run_writing_crew

        mock_crew = MagicMock()
        mock_crew.kickoff.side_effect = [SAMPLE_BLOG, SAMPLE_SOCIAL, SAMPLE_EMAILS]
        mock_crew_cls.return_value = mock_crew

        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "run-abc"
        mock_mlflow.start_run.return_value = mock_run

        package = run_writing_crew("camp-001", SAMPLE_BRIEF, SAMPLE_RESEARCH)

        assert isinstance(package, ContentPackage)
        assert package.campaign_id == "camp-001"
        assert mock_save_piece.call_count >= 4  # blog + twitter + linkedin + emails
        mock_save_exec.assert_called_once()
        assert mock_save_exec.call_args.kwargs["status"] == "success"

    @patch("src.writing_crew.crew.save_crew_execution")
    @patch("src.writing_crew.crew.save_content_piece")
    @patch("src.writing_crew.crew.mlflow")
    @patch("src.writing_crew.crew._run_blog_crew")
    @patch("src.writing_crew.crew.ChatOpenAI")
    def test_failed_run_saves_error(self, mock_llm, mock_run_blog_crew, mock_mlflow, mock_save_piece, mock_save_exec):
        from src.writing_crew.crew import run_writing_crew

        mock_run_blog_crew.side_effect = Exception("Rate limit exceeded")

        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "run-abc"
        mock_mlflow.start_run.return_value = mock_run

        with pytest.raises(Exception):
            run_writing_crew("camp-001", SAMPLE_BRIEF, SAMPLE_RESEARCH)

        mock_save_exec.assert_called_once()
        assert mock_save_exec.call_args.kwargs["status"] == "failed"
