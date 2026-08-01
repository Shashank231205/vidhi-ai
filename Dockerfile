# Backend image, built for HuggingFace Spaces.
#
# Spaces is the deployment target because it is the only free tier that fits:
# BGE-M3 is ~2.2GB resident and the risk classifier another ~268MB, against
# Vercel's 250MB function limit and Render's 512MB free instance. Spaces gives
# 16GB and 2 vCPU at no cost.
#
# The model is baked into the image rather than downloaded on boot. A Space
# sleeps after inactivity, and pulling 2.2GB on every wake would put a
# multi-minute stall in front of the first request instead of ~30s.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Spaces runs as a non-root user with only /home writable, so every cache
    # HuggingFace and Torch use has to live there or the build fails at runtime.
    HF_HOME=/home/user/.cache/huggingface \
    TORCH_HOME=/home/user/.cache/torch \
    # Threads, not processes: the model is loaded once per process, so extra
    # workers would multiply 2.2GB of resident memory for no throughput gain
    # on 2 vCPU.
    OMP_NUM_THREADS=2

RUN useradd -m -u 1000 user
WORKDIR /app

# Build tools for any package without a wheel, removed in the same layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=user backend/pyproject.toml backend/uv.lock ./backend/

# CPU-only torch, installed *before* anything that depends on it.
#
# Order matters and the cost of getting it wrong is large: install torch after
# sentence-transformers and pip resolves the default wheel, which drags in
# 2.9GB of NVIDIA CUDA libraries and 650MB of Triton that a CPU-only Space can
# never execute. Reinstalling afterwards does not remove them — they are
# separate packages, and the image measured 18.7GB. Installing the CPU wheel
# first means the CUDA variant is never a candidate.
RUN pip install --no-cache-dir uv==0.5.11 \
    && uv pip install --system --no-cache \
       --index-url https://download.pytorch.org/whl/cpu \
       torch

# Everything else resolves against the lockfile, so the deployed dependency set
# is the one that was tested. Torch is already satisfied.
RUN cd backend && uv pip install --system --no-cache -r pyproject.toml

COPY --chown=user backend/ ./backend/
COPY --chown=user ml/ ./ml/

USER user

# Bake the embedding model in. This is the slow half of the build and the
# reason a cold Space wakes in seconds rather than minutes.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-m3')" \
    && echo "embedding model cached"

WORKDIR /app/backend

# Spaces routes to 7860 by convention.
ENV PORT=7860
EXPOSE 7860

# The container is healthy once /health answers, which needs no database — a
# slow upstream must not cause a restart loop.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

# One worker, for the memory reason above. Uvicorn's factory form keeps
# settings out of module import, so a misconfiguration surfaces at startup
# with a readable error rather than an import traceback.
CMD ["sh", "-c", "uvicorn api.main:create_app --factory --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 120"]
