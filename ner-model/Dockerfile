# --- Stage 1: Training ---
FROM python:3.10.5-slim as trainer
LABEL description="Named entity recognition training image"
ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.6.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONPATH="/myapp:${PYTHONPATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    python3-dev
RUN pip install poetry==${POETRY_VERSION}

COPY pyproject.toml poetry.lock /app/
WORKDIR /app
RUN python -m poetry install --no-root --only main && \
    apt-get remove -y gcc python3-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

RUN poetry run python src/main/fine_tune/run_training.py

# --- Stage 2: Inference ---
FROM python:3.10.5-slim as inferencer
LABEL description="Named entity recognition inference image"
ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.6.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONPATH="/myapp:${PYTHONPATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    python3-dev
RUN pip install poetry==${POETRY_VERSION}

COPY pyproject.toml poetry.lock /app/
WORKDIR /app
RUN python -m poetry install --no-root --only main && \
    apt-get remove -y gcc python3-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . /app
COPY --from=trainer /app/data/model /app/data/model

EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "src.inference.main.run_api:app", "--host", "0.0.0.0", "--port", "8000"]
