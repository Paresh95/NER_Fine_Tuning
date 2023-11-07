# --- Stage 1: Training ---
FROM python:3.10.5-slim as trainer
LABEL description="Named entity recognition training image"
ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.6.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONPATH="/app:${PYTHONPATH}"

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

COPY data/ /app/data/
COPY logs/ /app/logs/
COPY src/data_utils /app/src/data_utils
COPY src/training /app/src/training
COPY src/__init__.py /app/src/
COPY src/static.yaml /app/src/
COPY Makefile /app/src/

RUN poetry run python src/training/main/run_training.py

# --- Stage 2: Inference ---
FROM python:3.10.5-slim as inferencer
LABEL description="Named entity recognition inference image"
ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.6.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONPATH="/app:${PYTHONPATH}"

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

COPY logs/ /app/logs/
COPY src/data_utils /app/src/data_utils
COPY src/inference /app/src/inference
COPY src/__init__.py /app/src/
COPY src/static.yaml /app/src/
COPY Makefile /app/src/
COPY --from=trainer /app/data/model /app/data/model
COPY --from=trainer /app/data/tokenizer /app/data/tokenizer
COPY --from=trainer /app/data/id2label.json /app/data/id2label.json

EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "src.inference.main.run_api:app", "--host", "0.0.0.0", "--port", "8000"]
