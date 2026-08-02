# Backend image.
#
# Deployed to Google Cloud Run (see deploy.sh). Vercel cannot host this — its
# functions cap at 250MB unzipped and 10s of execution, while this process holds
# a 2.2GB embedding model resident and an audit runs for minutes.
#
# Two earlier targets failed: HuggingFace Spaces moved Docker behind a paid
# plan, and Fly.io's registry push died on export after 158s because the image
# is ~9GB. Cloud Run states no limit on image size and streams images
# block-by-block at boot, so a large image starts in roughly the time a small
# one does.
#
# The model is baked into the image rather than downloaded on boot. The service
# scales to zero, and pulling 2.2GB from the Hub on every cold start would put a
# multi-minute stall in front of the first request.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # The container runs as a non-root user with only /home writable, so every
    # cache HuggingFace and Torch use has to live there or the build fails at
    # runtime.
    HF_HOME=/home/user/.cache/huggingface \
    TORCH_HOME=/home/user/.cache/torch \
    # Threads, not processes: the model is loaded once per process, so extra
    # workers would multiply 2.2GB of resident memory for no throughput gain on
    # the 2 vCPU the service is deployed with.
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
# reason a cold start is seconds rather than minutes.
#
# Formats the loader never reads are pruned. pytorch_model.bin is deliberately
# not among them, despite being a 2.2GB duplicate of model.safetensors:
# sentence-transformers probes for it while loading, and deleting it sent the
# container to huggingface.co on every boot, where startup hung indefinitely.
# 2.2GB of image is the price of a container that starts.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-m3')" \
    && python - <<'PRUNE'
import os, pathlib
# Snapshot entries are symlinks into blobs/, so the blob has to be unlinked
# through the link target — deleting the symlink alone frees nothing.
unused = ("model.onnx", "tf_model.h5", "flax_model.msgpack")
freed = 0
for root, _, files in os.walk("/home/user/.cache/huggingface/hub"):
    for name in files:
        if name not in unused:
            continue
        entry = pathlib.Path(root, name)
        target = entry.resolve()
        if target.exists():
            freed += target.stat().st_size
            target.unlink()
        entry.unlink(missing_ok=True)
print(f"pruned {freed / 1e9:.2f} GB of duplicate weights")
PRUNE

# Offline only from here: the download above needed the network, but at runtime
# the baked model must be used without revalidating against the Hub. Without
# this the container reaches out on every boot and startup hangs — verified by
# running the image with the flag absent.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app/backend

# Overridden by the host: Cloud Run injects PORT (8080). The CMD reads ${PORT},
# so the image works on any of these targets without a rebuild.
ENV PORT=8080
EXPOSE 8080

# The container is healthy once /health answers, which needs no database — a
# slow upstream must not cause a restart loop.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

# One worker, for the memory reason above. Uvicorn's factory form keeps
# settings out of module import, so a misconfiguration surfaces at startup
# with a readable error rather than an import traceback.
CMD ["sh", "-c", "uvicorn api.main:create_app --factory --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 120"]
