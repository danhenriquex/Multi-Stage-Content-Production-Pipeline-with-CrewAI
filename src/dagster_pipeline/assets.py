"""
Dagster assets — each crew maps to one asset.
Dependency chain: research → writing → editing → visual
"""

import dagster as dg
from src.shared.models import CampaignBrief, ContentPackage, ResearchReport


def _brief_from(config: dict) -> tuple[str, CampaignBrief]:
    campaign_id = config["campaign_id"]
    brief = CampaignBrief(**{k: v for k, v in config.items() if k != "campaign_id"})
    return campaign_id, brief


# ── Assets ────────────────────────────────────────────────────────────────────


@dg.asset(
    description="Market research: TAM/SAM, competitor analysis, trends.",
    group_name="content_pipeline",
    kinds={"python"},
    config_schema={
        "campaign_id": str,
        "title": str,
        "brief": str,
        "brand_voice": dg.Field(str, is_required=False, default_value=""),
        "target_audience": dg.Field(str, is_required=False, default_value=""),
        "keywords": dg.Field([str], is_required=False, default_value=[]),
    },
)
def research_crew_output(context: dg.AssetExecutionContext) -> dict:
    from src.research_crew.crew import run_research_crew

    campaign_id, brief = _brief_from(context.op_config)
    context.log.info(f"Research crew starting — {brief.title}")

    report = run_research_crew(campaign_id, brief)

    context.add_output_metadata(
        {
            "campaign_id": campaign_id,
            "insights_count": len(report.key_insights),
            "trends_count": len(report.trends),
            "competitors": len(report.competitors),
        }
    )
    return report.model_dump()


@dg.asset(
    description="Parallel writing: blog post, Twitter thread, LinkedIn, 3 email variants.",
    group_name="content_pipeline",
    kinds={"python"},
    config_schema={
        "campaign_id": str,
        "title": str,
        "brief": str,
        "brand_voice": dg.Field(str, is_required=False, default_value=""),
        "target_audience": dg.Field(str, is_required=False, default_value=""),
        "keywords": dg.Field([str], is_required=False, default_value=[]),
    },
)
def writing_crew_output(
    context: dg.AssetExecutionContext,
    research_crew_output: dict,
) -> dict:
    from src.writing_crew.crew import run_writing_crew

    campaign_id, brief = _brief_from(context.op_config)
    research = ResearchReport(**research_crew_output)
    context.log.info(f"Writing crew starting — {brief.title}")

    package = run_writing_crew(campaign_id, brief, research)

    context.add_output_metadata(
        {
            "campaign_id": campaign_id,
            "has_blog": package.blog_post is not None,
            "email_variants": len(package.email_variants),
        }
    )
    return package.model_dump()


@dg.asset(
    description="Editing: copy editor, brand voice alignment, SEO optimization.",
    group_name="content_pipeline",
    kinds={"python"},
    config_schema={
        "campaign_id": str,
        "title": str,
        "brief": str,
        "brand_voice": dg.Field(str, is_required=False, default_value=""),
        "target_audience": dg.Field(str, is_required=False, default_value=""),
        "keywords": dg.Field([str], is_required=False, default_value=[]),
    },
)
def editing_crew_output(
    context: dg.AssetExecutionContext,
    writing_crew_output: dict,
) -> dict:
    from src.editing_crew.crew import run_editing_crew

    campaign_id, brief = _brief_from(context.op_config)
    drafts = ContentPackage(**writing_crew_output)
    context.log.info(f"Editing crew starting — {brief.title}")

    polished = run_editing_crew(campaign_id, brief, drafts)

    context.add_output_metadata(
        {
            "campaign_id": campaign_id,
            "pieces_edited": sum(
                1
                for p in [
                    polished.blog_post,
                    polished.twitter_thread,
                    polished.linkedin_post,
                ]
                if p
            )
            + len(polished.email_variants),
        }
    )
    return polished.model_dump()


@dg.asset(
    description="Visual design brief: color palette, typography, asset production list.",
    group_name="content_pipeline",
    kinds={"python"},
    config_schema={
        "campaign_id": str,
        "title": str,
        "brief": str,
        "brand_voice": dg.Field(str, is_required=False, default_value=""),
        "target_audience": dg.Field(str, is_required=False, default_value=""),
        "keywords": dg.Field([str], is_required=False, default_value=[]),
    },
)
def visual_brief_output(
    context: dg.AssetExecutionContext,
    editing_crew_output: dict,
) -> dict:
    from src.visual_crew.crew import run_visual_crew

    campaign_id, brief = _brief_from(context.op_config)
    polished = ContentPackage(**editing_crew_output)
    context.log.info(f"Visual crew starting — {brief.title}")

    brief_draft = run_visual_crew(campaign_id, brief, polished)

    context.add_output_metadata(
        {
            "campaign_id": campaign_id,
            "brief_length": len(brief_draft.content),
            "asset_count": brief_draft.metadata.get("asset_count", 0),
        }
    )
    return brief_draft.model_dump()


# ── Dagster Definitions ───────────────────────────────────────────────────────

defs = dg.Definitions(
    assets=[
        research_crew_output,
        writing_crew_output,
        editing_crew_output,
        visual_brief_output,
    ],
)
