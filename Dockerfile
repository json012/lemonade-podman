FROM ubuntu:24.04

# Install system dependencies including Vulkan support
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip pipx \
    curl wget unzip \
    build-essential cmake \
    vulkan-tools vulkan-validationlayers \
    libvulkan-dev \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/lemonade && chown -R ubuntu:ubuntu /opt/lemonade

USER ubuntu

RUN pipx install uv
ENV PATH="/home/ubuntu/.local/bin:$PATH"

# Use /opt/lemonade to avoid conflicts with base image's /app
RUN mkdir -p /opt/lemonade
WORKDIR /opt/lemonade

# Locked Python deps
COPY --chown=ubuntu:ubuntu pyproject.toml uv.lock .
RUN uv sync --no-dev --frozen

# If this isn't done at build time, lemonade server will try to download it at runtime.
RUN uv run lemonade-install --llamacpp vulkan

# Ensure the entire .venv directory is accessible to non-root users
# This is crucial when using UserNS=keep-id in the quadlet
RUN chmod -R 755 /opt/lemonade/.venv && \
    chown -R 1000:1000 /opt/lemonade/.venv || true

# Lemonade runtime envs
ENV LEMONADE_HOST=0.0.0.0 \
    LEMONADE_PORT=8000 \
    LEMONADE_LLAMACPP=vulkan

EXPOSE 8000

CMD ["uv", "run", "lemonade-server-dev", "serve", "--llamacpp", "vulkan"]
