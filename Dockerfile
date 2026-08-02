# Backend image.
#
# Runs the remote embedding backend: BGE-M3 is called over the HuggingFace
# Inference API rather than loaded in-process. The vectors are identical — same
# model, same 1024 dimensions — so the corpus in Postgres is valid either way,
# and retrieval metrics are unchanged (measured: P@5 0.340, R@5 0.775,
# MRR 0.742, hit-rate 0.80 on both backends).
#
# What changes is the image. Baking the model in meant torch (~2.5GB installed)
# plus 4.5GB of weights, for a ~9GB image: HuggingFace Spaces moved Docker
# behind a paid plan, Fly.io's registry push died on export after 158s, and
# Cloud Run needs an open billing account. Without them the image is ~400MB and
# fits any free tier.
#
# The cost is latency: ~376ms per query against ~27ms in-process, plus a cold
# start on the Hub's side. Embedding cache entries are content-addressed and
# keyed by model, so repeated queries skip the call entirely.
#
# Vercel still cannot host this — its functions cap at 250MB unzipped and 10s of
# execution, while an audit runs for minutes.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBEDDING_BACKEND=remote

RUN useradd -m -u 1000 user
WORKDIR /app

# Build tools for any package without a wheel, removed in the same layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=user backend/pyproject.toml backend/uv.lock ./backend/

# The base dependency set only. `local-embeddings` is deliberately not installed:
# it pulls torch, and pip resolves the CUDA build by default — 2.9GB of NVIDIA
# libraries and 650MB of Triton that a CPU-only host can never execute.
RUN pip install --no-cache-dir uv==0.5.11 \
    && cd backend && uv pip install --system --no-cache -r pyproject.toml

COPY --chown=user backend/ ./backend/
COPY --chown=user ml/ ./ml/

USER user
WORKDIR /app/backend

# Overridden by the host; the CMD reads ${PORT}, so the image works anywhere
# without a rebuild.
ENV PORT=8080
EXPOSE 8080

# The container is healthy once /health answers, which needs no database — a
# slow upstream must not cause a restart loop.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

# Uvicorn's factory form keeps settings out of module import, so a
# misconfiguration surfaces at startup with a readable error rather than an
# import traceback.
CMD ["sh", "-c", "uvicorn api.main:create_app --factory --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 120"]
