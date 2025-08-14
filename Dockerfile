
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY README.md pyproject.toml ./

RUN mkdir -p /app/model

RUN useradd -r -m -d /home/app app && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
CMD curl -fsS http://127.0.0.1:8000/version || exit 1

ENTRYPOINT ["uvicorn", "src.api:app"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
