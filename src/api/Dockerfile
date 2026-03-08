FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] psycopg2-binary structlog \
    pydantic pydantic-settings prometheus-client \
    crewai crewai-tools langchain-openai langchain-community ddgs \
    textstat mlflow tenacity

COPY src/ ./src/
ENV PYTHONPATH=/app

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
