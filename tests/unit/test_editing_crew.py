"""Unit tests for editing crew — scoring functions and run function."""

from unittest.mock import MagicMock, patch

import pytest

from src.editing_crew.crew import (
    _compute_brand_voice_score,
    _compute_readability,
    _compute_seo_score,
    _score_package,
)
from src.shared.models import CampaignBrief, ContentDraft, ContentPackage

SAMPLE_BRIEF = CampaignBrief(
    title="AI CRM Launch",
    brief="Launch campaign for AI-powered CRM targeting SMBs",
    brand_voice="Professional yet approachable",
    target_audience="SMB founders",
    keywords=["AI CRM", "sales automation", "SMB"],
)

SAMPLE_BLOG = ContentDraft(
    campaign_id="test-123",
    content_type="blog_post",
    title="How AI CRM Transforms Sales Automation for SMB",
    content=(
        "AI CRM solutions are revolutionizing sales automation for SMB businesses. "
        "By leveraging artificial intelligence, small and medium businesses can now "
        "automate repetitive tasks and focus on building customer relationships. "
        "Sales automation powered by AI CRM reduces manual data entry by 60%. "
        "SMB founders report significant ROI improvements within the first quarter."
    ),
    metadata={"word_count": 60, "version": 1},
)

SAMPLE_TWITTER = ContentDraft(
    campaign_id="test-123",
    content_type="twitter_thread",
    title="Twitter Thread",
    content="1/ AI CRM is changing how SMBs do sales automation\n2/ Here's what you need to know",
    metadata={},
)

SAMPLE_LINKEDIN = ContentDraft(
    campaign_id="test-123",
    content_type="linkedin_post",
    title="LinkedIn Post",
    content="AI CRM is transforming sales automation for SMB businesses everywhere.",
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
        score = _compute_seo_score("The weather is nice today.", ["AI CRM", "SMB"])
        assert score == 0.0

    def test_returns_float(self):
        score = _compute_seo_score(SAMPLE_BLOG.content, SAMPLE_BRIEF.keywords)
        assert isinstance(score, float)

    def test_score_capped_at_100(self):
        content = " ".join(["AI CRM sales automation SMB"] * 50)
        score = _compute_seo_score(content, ["AI CRM", "sales automation", "SMB"])
        assert score <= 100.0


@pytest.mark.unit
class TestComputeBrandVoiceScore:
    def test_clean_content_scores_high(self):
        clean = "Our AI CRM delivers measurable results for SMB sales teams."
        score = _compute_brand_voice_score(clean, "Professional")
        assert score > 0.7

    def test_jargon_heavy_content_scores_lower(self):
        jargon = (
            "We leverage synergistic paradigms to optimize holistic solutions "
            "and streamline end-to-end deliverables for stakeholders."
        )
        clean = "Our product helps sales teams close more deals faster."
        assert _compute_brand_voice_score(jargon, "Professional") < _compute_brand_voice_score(clean, "Professional")

    def test_returns_float_between_0_and_1(self):
        score = _compute_brand_voice_score(SAMPLE_BLOG.content, "Professional")
        assert 0.0 <= score <= 1.0


@pytest.mark.unit
class TestScorePackage:
    def test_returns_dict(self):
        scores = _score_package(SAMPLE_PACKAGE, SAMPLE_BRIEF)
        assert isinstance(scores, dict)

    def test_blog_has_scores(self):
        scores = _score_package(SAMPLE_PACKAGE, SAMPLE_BRIEF)
        assert "blog" in scores

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
    @patch("src.editing_crew.crew.Task")
    @patch("src.editing_crew.crew.Agent")
    @patch("src.editing_crew.crew.ChatOpenAI")
    def test_successful_run_returns_package(
        self,
        mock_llm,
        mock_agent,
        mock_task,
        mock_crew_cls,
        mock_mlflow,
        mock_save_piece,
        mock_save_exec,
    ):
        from src.editing_crew.crew import run_editing_crew

        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = SAMPLE_BLOG.content + "\n\nSEO NOTES: Meta description."
        mock_crew_cls.return_value = mock_crew

        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "run-edit-001"
        mock_mlflow.start_run.return_value = mock_run

        result = run_editing_crew("camp-001", SAMPLE_BRIEF, SAMPLE_PACKAGE)

        assert isinstance(result, ContentPackage)
        assert result.campaign_id == "camp-001"
        mock_save_exec.assert_called_once()
        assert mock_save_exec.call_args.kwargs["status"] == "success"

    @patch("src.editing_crew.crew.save_crew_execution")
    @patch("src.editing_crew.crew.save_content_piece")
    @patch("src.editing_crew.crew.mlflow")
    @patch("src.editing_crew.crew.Crew")
    @patch("src.editing_crew.crew.Task")
    @patch("src.editing_crew.crew.Agent")
    @patch("src.editing_crew.crew.ChatOpenAI")
    def test_edited_pieces_have_version_incremented(
        self,
        mock_llm,
        mock_agent,
        mock_task,
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
    @patch("src.editing_crew.crew.Task")
    @patch("src.editing_crew.crew.Agent")
    @patch("src.editing_crew.crew.ChatOpenAI")
    def test_failed_run_saves_error(
        self,
        mock_llm,
        mock_agent,
        mock_task,
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

        assert mock_save_exec.call_args.kwargs["status"] == "failed"
