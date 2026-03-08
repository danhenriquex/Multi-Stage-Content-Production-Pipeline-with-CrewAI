FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    dagster dagster-webserver \
    psycopg2-binary structlog pydantic pydantic-settings \
    crewai crewai-tools langchain-openai langchain-community ddgs \
    textstat mlflow prometheus-client tenacity

COPY src/ ./src/
ENV PYTHONPATH=/app
ENV DAGSTER_HOME=/dagster_home

EXPOSE 3000
CMD ["dagster", "dev", "-f", "src/dagster_pipeline/assets.py", "-h", "0.0.0.0", "-p", "3000"]
