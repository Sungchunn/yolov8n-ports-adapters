FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --extra vision --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --extra vision --no-dev

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "inference.entrypoints.app:app", "--host", "0.0.0.0", "--port", "8000"]
