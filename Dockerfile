# Lemonade Server with llama.cpp (Vulkan) using uv
FROM ghcr.io/ggml-org/llama.cpp:server-vulkan

# Install only what we need that the base image doesn't have
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip pipx \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/lemonade && chown -R ubuntu:ubuntu /opt/lemonade

USER ubuntu

RUN pipx install uv
ENV PATH="/home/ubuntu/.local/bin:$PATH"

# Use /opt/lemonade to avoid conflicts with base image's /app
RUN mkdir -p /opt/lemonade
WORKDIR /opt/lemonade

# Locked Python deps
COPY pyproject.toml uv.lock .
RUN uv sync --no-dev --frozen

# Create symlinks to the existing llama-server binary in the expected location
# The base image has llama-server in /app, so we link it to our venv structure
# If this isn't done lemonade will try to download the binary itself and run into permissions issues
RUN mkdir -p /opt/lemonade/.venv/bin/vulkan/llama_server/build/bin && \
    ln -sf /app/llama-server /opt/lemonade/.venv/bin/vulkan/llama_server/build/bin/llama-server && \
    /app/llama-server --version 2>&1 | grep "version:" | awk '{print $2}' > /opt/lemonade/.venv/bin/vulkan/llama_server/version.txt && \
    echo "Successfully created symlink to existing llama-server binary and extracted version"

# Ensure the entire .venv directory is accessible to non-root users
# This is crucial when using UserNS=keep-id in the quadlet
RUN chmod -R 755 /opt/lemonade/.venv && \
    chown -R 1000:1000 /opt/lemonade/.venv || true

# Lemonade runtime envs
ENV LEMONADE_HOST=0.0.0.0 \
    LEMONADE_PORT=8000 \
    LEMONADE_LLAMACPP=vulkan

EXPOSE 8000

# Override any entrypoint from the base image and run server directly
ENTRYPOINT []

CMD ["/bin/bash", "-c", "/opt/lemonade/.venv/bin/lemonade-server-dev serve --llamacpp vulkan"]
