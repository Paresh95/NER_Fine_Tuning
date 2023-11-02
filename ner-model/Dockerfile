FROM python:3.10.5-slim
LABEL description="Named entity recognition service"
ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.6.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONPATH="/myapp:${PYTHONPATH}"

RUN apt-get update && apt-get install -y --no-install-recommends curl
RUN pip install poetry==${POETRY_VERSION}

COPY pyproject.toml poetry.lock /myapp/
WORKDIR /myapp
RUN python -m poetry install --no-root --no-dev

COPY . /myapp

EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "src.main.fine_tune.run_api:app", "--host", "0.0.0.0", "--port", "8000"]
