from typing import Optional
from pydantic import BaseModel, Field


class CampaignBrief(BaseModel):
    title: str
    brief: str
    brand_voice: Optional[str] = None
    target_audience: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    campaign_id: str
    market_size: str = ""
    competitors: list[dict] = Field(default_factory=list)
    trends: list[str] = Field(default_factory=list)
    key_insights: list[str] = Field(default_factory=list)
    raw_output: str = ""


class ContentDraft(BaseModel):
    campaign_id: str
    content_type: str  # blog | twitter_thread | linkedin | email | visual_brief
    title: str = ""
    content: str = ""
    metadata: dict = Field(default_factory=dict)


class ContentPackage(BaseModel):
    campaign_id: str
    blog_post: Optional[ContentDraft] = None
    twitter_thread: Optional[ContentDraft] = None
    linkedin_post: Optional[ContentDraft] = None
    email_variants: list[ContentDraft] = Field(default_factory=list)
    visual_brief: Optional[ContentDraft] = None


class QualityScores(BaseModel):
    readability: float = 0.0
    seo_score: float = 0.0
    brand_voice_match: float = 0.0
    cost_usd: float = 0.0
    execution_time_s: float = 0.0


class CrewResult(BaseModel):
    crew_name: str
    status: str  # success | failed
    output: dict = Field(default_factory=dict)
    metrics: QualityScores = Field(default_factory=QualityScores)
    error: Optional[str] = None
