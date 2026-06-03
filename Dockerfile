# syntax=docker/dockerfile:1.7
#
# vala + lichess-bot, single image. Multi-stage to keep runtime lean.
# Mirrors the rorschach image; the one wrinkle is torch:
#   pyproject pins torch==2.4.0 from the cu121 index (the 4090 dev box keeps
#   its GPU). For this CPU-only deployment we install the +cpu wheel instead
#   and exclude torch from the locked sync so the CUDA wheel is never pulled.
#
# Build:   docker build -t vala .
# Run:     see compose.yaml

ARG PYTHON_VERSION=3.11
ARG LICHESS_BOT_REF=master

# ---------- builder ----------
FROM python:${PYTHON_VERSION}-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

# git is needed because maia3 is a git+https dependency.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/vala

# torch: the committed pyproject pins the cu121 wheel so the 4090 dev box keeps
# its GPU. For this CPU-only image we repoint the torch index to the +cpu wheel
# and re-lock IN THE IMAGE. This also drops the nvidia-cuda-* packages that the
# cu121 resolution drags in (~2.5 GB of dead weight on a CPU host). The committed
# pyproject.toml / uv.lock are never modified — only this build's copy is.
COPY pyproject.toml uv.lock ./
RUN sed -i 's#download.pytorch.org/whl/cu121#download.pytorch.org/whl/cpu#g' pyproject.toml \
 && uv lock
# Resolve deps first for layer caching; project install happens after sources copy.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY vala ./vala
COPY bin ./bin
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# lichess-bot lives next to us; install its deps into the same venv.
ARG LICHESS_BOT_REF
RUN git clone --depth 1 --branch "${LICHESS_BOT_REF}" \
        https://github.com/lichess-bot-devs/lichess-bot.git /opt/lichess-bot \
 && rm -rf /opt/lichess-bot/.git
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/vala/.venv/bin/python \
        -r /opt/lichess-bot/requirements.txt

# ---------- runtime ----------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/vala/.venv/bin:${PATH}" \
    HF_HOME=/data/huggingface \
    LICHESS_BOT_DIR=/opt/lichess-bot

# libgomp1: torch CPU wheel pulls libgomp at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/vala /opt/vala
COPY --from=builder /opt/lichess-bot /opt/lichess-bot

# Repo-tracked config baked in — edit configs/config.docker.yml, push,
# redeploy. The bot token is injected at runtime via the LICHESS_BOT_TOKEN env
# (lichess-bot reads it natively, overriding the placeholder in config.yml).
COPY configs/config.docker.yml /opt/lichess-bot/config.yml

RUN chmod +x /opt/vala/bin/patricia \
 && mkdir -p /data/huggingface

WORKDIR /opt/lichess-bot

# vala has no chat personality, so run vanilla lichess-bot. Config is baked in
# (see COPY above); the token comes from LICHESS_BOT_TOKEN. The engine entry
# point is /opt/vala/.venv/bin/vala-uci and Patricia is /opt/vala/bin/patricia
# (vala/engine.py resolves it relative to the package).
CMD ["python", "lichess-bot.py"]
