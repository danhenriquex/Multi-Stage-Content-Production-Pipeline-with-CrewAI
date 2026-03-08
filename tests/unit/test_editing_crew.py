"""Unit tests for editing crew — all LLM calls mocked."""

from unittest.mock import MagicMock, patch

import pytest

from src.editing_crew.crew import (
    _compute_brand_voice_score,
    _compute_readability,
    _compute_seo_score,
    _score_package,
)
from src.shared.models import CampaignBrief, ContentDraft, ContentPackage, QualityScores

SAMPLE_BRIEF = CampaignBrief(
    title="AI CRM Launch",
    brief="Launch campaign for new AI-powered CRM targeting SMBs",
    brand_voice="Professional yet approachable",
    target_audience="SMB founders and sales managers",
    keywords=["AI CRM", "sales automation", "SMB"],
)

SAMPLE_BLOG = ContentDraft(
    campaign_id="test-123",
    content_type="blog",
    title="How AI CRM Transforms SMB Sales",
    content="""
## How AI CRM Is Changing SMB Sales in 2026

Most SMB sales teams waste hours on manual data entry every single day.
AI CRM tools solve this problem by automating the tedious work.

## Why Sales Automation Matters for SMBs

The SMB market is growing rapidly. Sales automation gives small teams
the power of enterprise tools without enterprise complexity.

### Real Results

Companies using AI CRM see measurable improvements in close rates.
The data shows consistent gains across industries.

## Getting Started

Start your free trial today and see results within 30 days.
""",
    metadata={"word_count": 95},
)

SAMPLE_TWITTER = ContentDraft(
    campaign_id="test-123",
    content_type="twitter_thread",
    title="Twitter Thread",
    content="1/ Most SMB teams waste 3 hours/day on CRM data entry.\n\n2/ AI CRM fixes this.",
    metadata={},
)

SAMPLE_LINKEDIN = ContentDraft(
    campaign_id="test-123",
    content_type="linkedin",
    title="LinkedIn Post",
    content="I spent years watching sales teams struggle with manual CRM work.\nAI changes everything.\n\n#AICRM #SMB",
    metadata={},
)

SAMPLE_EMAIL = ContentDraft(
    campaign_id="test-123",
    content_type="email",
    title="Email — Awareness",
    content=(
        "SUBJECT: The CRM problem nobody talks about\n"
        "BODY: Most teams waste time on data entry...\n"
        "CTA: Learn more"
    ),
    metadata={"stage": "awareness"},
)

SAMPLE_PACKAGE = ContentPackage(
    campaign_id="test-123",
    blog_post=SAMPLE_BLOG,
    twitter_thread=SAMPLE_TWITTER,
    linkedin_post=SAMPLE_LINKEDIN,
    email_variants=[SAMPLE_EMAIL],
)


@pytest.mark.unit
class TestComputeReadability:
    def test_returns_float(self):
        score = _compute_readability(SAMPLE_BLOG.content)
        assert isinstance(score, float)

    def test_score_in_valid_range(self):
        score = _compute_readability(SAMPLE_BLOG.content)
        assert 0.0 <= score <= 100.0

    def test_empty_content_returns_zero(self):
        score = _compute_readability("")
        assert score == 0.0

    def test_simple_text_scores_higher(self):
        simple = "The cat sat on the mat. It was a good cat."
        complex_text = (
            "The implementation of sophisticated technological paradigms"
            " necessitates comprehensive organizational restructuring."
        )
        assert _compute_readability(simple) > _compute_readability(complex_text)


@pytest.mark.unit
class TestComputeSeoScore:
    def test_all_keywords_present_scores_high(self):
        content = "AI CRM helps with sales automation for every SMB business."
        score = _compute_seo_score(content, ["AI CRM", "sales automation", "SMB"])
        assert score > 70

    def test_no_keywords_scores_zero(self):
        score = _compute_seo_score("Some content with no relevant terms", ["AI CRM", "SMB"])
        assert score == 0.0

    def test_empty_keywords_returns_zero(self):
        score = _compute_seo_score("Some content here", [])
        assert score == 0.0

    def test_keyword_in_first_200_chars_gets_bonus(self):
        content_early = "AI CRM is transforming sales. " + "x " * 100
        content_late = "x " * 100 + " AI CRM is transforming sales."
        score_early = _compute_seo_score(content_early, ["AI CRM"])
        score_late = _compute_seo_score(content_late, ["AI CRM"])
        assert score_early > score_late

    def test_score_capped_at_100(self):
        content = "AI CRM AI CRM AI CRM sales automation SMB " * 10
        score = _compute_seo_score(content, ["AI CRM", "sales automation", "SMB"])
        assert score <= 100.0


@pytest.mark.unit
class TestComputeBrandVoiceScore:
    def test_clean_content_scores_high(self):
        content = "You can set up your CRM in minutes. No training needed."
        score = _compute_brand_voice_score(content, "Professional yet approachable")
        assert score > 0.8

    def test_jargon_heavy_content_scores_lower(self):
        clean = "You can set up your CRM in minutes."
        jargon = "Leverage synergy to utilize holistic paradigms and robust scalable solutions."
        assert _compute_brand_voice_score(clean, None) > _compute_brand_voice_score(jargon, None)

    def test_hedging_language_penalized(self):
        clean = "This works well for your team."
        hedging = "This might possibly work well in some cases for your team perhaps."
        assert _compute_brand_voice_score(clean, None) > _compute_brand_voice_score(hedging, None)

    def test_empty_content_returns_zero(self):
        score = _compute_brand_voice_score("", None)
        assert score == 0.0

    def test_score_between_zero_and_one(self):
        score = _compute_brand_voice_score(SAMPLE_BLOG.content, "Professional")
        assert 0.0 <= score <= 1.0


@pytest.mark.unit
class TestScorePackage:
    def test_returns_scores_for_all_pieces(self):
        scores = _score_package(SAMPLE_PACKAGE, SAMPLE_BRIEF)
        assert "blog" in scores
        assert "twitter_thread" in scores
        assert "linkedin" in scores
        assert "email_awareness" in scores

    def test_each_score_is_quality_scores(self):
        scores = _score_package(SAMPLE_PACKAGE, SAMPLE_BRIEF)
        for name, score in scores.items():
            assert isinstance(score, QualityScores)

    def test_skips_none_pieces(self):
        package = ContentPackage(
            campaign_id="test-123",
            blog_post=SAMPLE_BLOG,
            # twitter, linkedin, emails all None/empty
        )
        scores = _score_package(package, SAMPLE_BRIEF)
        assert "blog" in scores
        assert "twitter_thread" not in scores

    def test_blog_readability_is_nonzero(self):
        scores = _score_package(SAMPLE_PACKAGE, SAMPLE_BRIEF)
        assert scores["blog"].readability > 0

    def test_blog_seo_score_nonzero_with_keywords(self):
        scores = _score_package(SAMPLE_PACKAGE, SAMPLE_BRIEF)
        assert scores["blog"].seo_score > 0


@pytest.mark.unit
class TestRunEditingCrew:
    @patch("src.editing_crew.crew.save_crew_execution")
    @patch("src.editing_crew.crew.save_content_piece")
    @patch("src.editing_crew.crew.mlflow")
    @patch("src.editing_crew.crew.Crew")
    @patch("src.editing_crew.crew.Agent")
    @patch("src.editing_crew.crew.ChatOpenAI")
    def test_successful_run_returns_package(
        self,
        mock_llm,
        mock_agent,
        mock_crew_cls,
        mock_mlflow,
        mock_save_piece,
        mock_save_exec,
    ):
        from src.editing_crew.crew import run_editing_crew

        edited_content = SAMPLE_BLOG.content + "\n\nSEO NOTES: Meta description here."
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = edited_content
        mock_crew_cls.return_value = mock_crew

        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "run-edit-001"
        mock_mlflow.start_run.return_value = mock_run

        result = run_editing_crew("camp-001", SAMPLE_BRIEF, SAMPLE_PACKAGE)

        assert isinstance(result, ContentPackage)
        assert result.campaign_id == "camp-001"
        assert result.blog_post is not None
        mock_save_exec.assert_called_once()
        assert mock_save_exec.call_args.kwargs["status"] == "success"

    @patch("src.editing_crew.crew.save_crew_execution")
    @patch("src.editing_crew.crew.save_content_piece")
    @patch("src.editing_crew.crew.mlflow")
    @patch("src.editing_crew.crew.Crew")
    @patch("src.editing_crew.crew.Agent")
    @patch("src.editing_crew.crew.ChatOpenAI")
    def test_edited_pieces_have_version_incremented(
        self,
        mock_llm,
        mock_agent,
        mock_crew_cls,
        mock_mlflow,
        mock_save_piece,
        mock_save_exec,
    ):
        from src.editing_crew.crew import run_editing_crew

        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = SAMPLE_BLOG.content
        mock_crew_cls.return_value = mock_crew

        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "run-edit-002"
        mock_mlflow.start_run.return_value = mock_run

        result = run_editing_crew("camp-001", SAMPLE_BRIEF, SAMPLE_PACKAGE)

        assert result.blog_post.metadata.get("edited") is True
        assert result.blog_post.metadata.get("version", 0) > 1

    @patch("src.editing_crew.crew.save_crew_execution")
    @patch("src.editing_crew.crew.save_content_piece")
    @patch("src.editing_crew.crew.mlflow")
    @patch("src.editing_crew.crew.Crew")
    @patch("src.editing_crew.crew.Agent")
    @patch("src.editing_crew.crew.ChatOpenAI")
    def test_failed_run_saves_error(
        self,
        mock_llm,
        mock_agent,
        mock_crew_cls,
        mock_mlflow,
        mock_save_piece,
        mock_save_exec,
    ):
        from src.editing_crew.crew import run_editing_crew

        mock_crew = MagicMock()
        mock_crew.kickoff.side_effect = Exception("LLM timeout")
        mock_crew_cls.return_value = mock_crew

        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "run-edit-003"
        mock_mlflow.start_run.return_value = mock_run

        with pytest.raises(Exception, match="LLM timeout"):
            run_editing_crew("camp-001", SAMPLE_BRIEF, SAMPLE_PACKAGE)

        mock_save_exec.assert_called_once()
        assert mock_save_exec.call_args.kwargs["status"] == "failed"
