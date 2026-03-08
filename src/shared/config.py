from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    # LLM
    openai_api_key: str = ""
    tavily_api_key: str = ""
    llm_model_primary: str = "gpt-4o-mini"  # cheap mode: gpt-4o-mini everywhere
    llm_model_secondary: str = "gpt-4o-mini"
    llm_temperature: float = 0.7

    # Cheap mode — limits agent iterations to reduce API cost
    cheap_mode: bool = True
    max_iter_primary: int = 1  # research/editing/visual agents
    max_iter_secondary: int = 1  # writing agents

    # Postgres
    postgres_user: str = "content"
    postgres_password: str = ""
    postgres_db: str = "content_pipeline"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8011
    chroma_collection_brand: str = "brand_voice_examples"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5050"
    mlflow_experiment_name: str = "content-pipeline"

    # Prometheus
    prometheus_pushgateway: str = "http://localhost:9091"

    # App
    log_level: str = "INFO"
    max_crew_retries: int = 2
    seo_score_threshold: float = 70.0
    brand_voice_threshold: float = 0.75

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
