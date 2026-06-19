# OPTIONAL: bake the Cosmos-Predict2 cu130 DEPENDENCY env into /opt/venv for reproducible / fast
# cold-start instances. NOT required — on a VM with the home dir mounted, env.sh uses the existing
# model/.venv directly. Only DEPENDENCIES are baked (--no-install-project); cosmos_predict2 code +
# datasets + checkpoints come from the mounted home, so the agent's edits take effect live.
#
# Build it from a build-provisioned researcher pod (this needs `bake`/buildx, which a plain GPU box
# does NOT have):   cd ~/mimic-video && bake build -f research/Dockerfile .
#   -> europe-west4-docker.pkg.dev/micro-research-shared/apps/mimic-video:latest
FROM europe-west4-docker.pkg.dev/micro-research-shared/dockerhub-mirror/nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:/usr/local/bin:/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git build-essential pkg-config ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
RUN curl -fsSL https://opencode.ai/install | bash

WORKDIR /build/model
COPY model/pyproject.toml model/uv.lock ./
RUN uv sync --no-install-project --extra cu130 --index-strategy unsafe-best-match

CMD ["/bin/bash"]
