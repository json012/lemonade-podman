FROM docker.io/rocm/pytorch:latest

RUN apt-get update && apt-get install -y pipx && rm -rf /var/lib/apt/lists/*

# Prepare HF cache dir (as root) so ubuntu can write to it
RUN mkdir -p /models/hub && chown -R ubuntu:ubuntu /models

USER ubuntu

RUN pipx install uv

ENV PATH="/home/ubuntu/.local/bin:$PATH"

ENV HF_HOME=/models \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/inference

COPY --chown=ubuntu:ubuntu pyproject.toml uv.lock .python-version .

RUN uv sync --no-dev --frozen

COPY --chown=ubuntu:ubuntu app /opt/inference/app
RUN ls -lah /opt/inference/app

# Download models at build time to avoid cold starts by downloading at runtime
RUN uv run -m app.src.inference.install

EXPOSE 8000

# Single worker handles concurrent requests via async/thread pool
# Scale workers only if needed after profiling (1 worker = 3 models in memory)
# GPU will serialise requests anyway, so multiple workers will only help if you have multiple GPUs
CMD ["uv", "run", "fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
