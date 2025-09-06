# Lemonade Server with llama.cpp (Vulkan) using uv
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    python3 python3-venv python3-pip \
    libvulkan1 mesa-vulkan-drivers vulkan-tools \
    pipx \
  && rm -rf /var/lib/apt/lists/*

RUN pipx install uv

# Add pipx bin to PATH
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock .
RUN uv sync --no-dev --frozen

ENV LEMONADE_HOST=0.0.0.0 \
    LEMONADE_PORT=8000 \
    LEMONADE_LLAMACPP=vulkan

EXPOSE 8000

CMD ["uv", "run", "lemonade-server-dev", "serve", "--llamacpp", "vulkan"]
