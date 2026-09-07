# syntax=docker/dockerfile:1.7
# SCRAG - multi-stage, CPU-only.
#
# Image size is the binding constraint on free tiers, and torch plus two
# transformer checkpoints gets large fast. Three things keep it down:
#
#   1. CPU-only torch from the PyTorch CPU index. The default PyPI wheel pulls
#      CUDA libraries that are dead weight here - several GB of them.
#   2. A builder stage that compiles nothing into the final image: only the
#      installed site-packages are copied forward.
#   3. Model weights pre-warmed at BUILD time, so no request ever waits on a
#      HuggingFace download, and no HF cache metadata bloats the layer.
#
# Secrets are NEVER baked in. ANTHROPIC_API_KEY arrives from the host
# environment at run time. A key in any layer is a leaked key even if a later
# layer removes it.

# --------------------------------------------------------------------------
# Stage 1 - build the virtualenv
# --------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential is needed to compile a few wheels; it stays in this stage.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-runtime.txt .

# CPU-only torch first, from the PyTorch CPU index, so the CUDA-enabled wheel
# is never resolved. Installing it before the rest pins the choice for every
# package that depends on torch.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.0" \
    && pip install -r requirements-runtime.txt

# --------------------------------------------------------------------------
# Stage 2 - pre-warm the model weights
# --------------------------------------------------------------------------
FROM builder AS models

ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/models \
    HF_HUB_DISABLE_XET=1

# Download at build time, not per request. The Xet transfer backend stalled
# repeatedly during development, hence HF_HUB_DISABLE_XET above.
#
# The embedding model and the NLI verifier are public and pinned here. The
# Module B evaluator checkpoint is NOT public - it is produced by
# train/finetune_evaluator.py and mounted or copied in separately; see
# DEPLOY.md.
RUN python - <<'PY'
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
snapshot_download(
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    allow_patterns=["config.json", "model.safetensors", "spm.model",
                    "tokenizer_config.json", "special_tokens_map.json",
                    "added_tokens.json"],
)
print("model weights pre-warmed")
PY

# --------------------------------------------------------------------------
# Stage 3 - runtime
# --------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/models \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_XET=1 \
    TRANSFORMERS_VERBOSITY=error \
    SCRAG_HOST=0.0.0.0 \
    PORT=8000

# curl is used by the container HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=models /opt/models /opt/models

# Run as a non-root user. HF Spaces expects uid 1000 and a writable home.
RUN useradd -m -u 1000 scrag
WORKDIR /home/scrag/app

COPY --chown=scrag:scrag config.py ./
COPY --chown=scrag:scrag core/ ./core/
COPY --chown=scrag:scrag app/ ./app/
COPY --chown=scrag:scrag data/demo/ ./data/demo/

USER scrag

# Build the demo index NOW, so the first query is not a cold rebuild. Built
# rather than copied: a committed FAISS file would drift from the chunker that
# produced it and would not be reproducible from a clean clone.
RUN python -m app.build_demo_index

EXPOSE 8000 8501

# Readiness, not just liveness: the app reports index and model state so a
# warming container is distinguishable from a broken one.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

# Default to the Streamlit app; docker-compose overrides this for the API.
CMD ["sh", "-c", "streamlit run app/ui.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true"]
