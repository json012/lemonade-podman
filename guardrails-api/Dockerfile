FROM docker.io/rocm/pytorch:latest

# Accept a build arg for the Guardrails token
# We'll add this to the config using the configure command below
ARG GUARDRAILS_TOKEN

RUN apt-get update && apt-get install -y pipx git && rm -rf /var/lib/apt/lists/*

# Prepare HF cache dir (as root) so ubuntu can write to it
RUN mkdir -p /models/hub && chown -R ubuntu:ubuntu /models && \
    mkdir -p /opt/nltk_data && chown -R ubuntu:ubuntu /opt/nltk_data && \
    touch /home/ubuntu/.guardrailsrc && chown ubuntu:ubuntu /home/ubuntu/.guardrailsrc

USER ubuntu

RUN pipx install uv

ENV PATH="/home/ubuntu/.local/bin:$PATH"

ENV HF_HOME=/models \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    GUARDRAILS_RUN_SYNC=true

WORKDIR /opt/inference

COPY --chown=ubuntu:ubuntu pyproject.toml uv.lock .python-version .

RUN --mount=type=cache,target=/home/ubuntu/.cache/uv,uid=1000,gid=1000 \
    uv sync --no-dev --frozen

# Set the directory for nltk data
ENV NLTK_DATA=/opt/nltk_data

# Download punkt data
RUN --mount=type=cache,target=/home/ubuntu/.cache/uv,uid=1000,gid=1000 \
    uv run -m nltk.downloader -d /opt/nltk_data punkt

# Run the Guardrails configure command to create a .guardrailsrc file
RUN uv run guardrails configure --enable-metrics --disable-remote-inferencing --token $GUARDRAILS_TOKEN

# Install any validators from the hub you want
RUN --mount=type=cache,target=/home/ubuntu/.cache/uv,uid=1000,gid=1000 \
    uv run guardrails hub install hub://guardrails/detect_jailbreak && \
    # uv run guardrails hub install hub://guardrails/detect_pii && \
    uv run guardrails hub install hub://guardrails/gibberish_text && \
    uv run guardrails hub install hub://guardrails/nsfw_text && \
    uv run guardrails hub install hub://guardrails/toxic_language

COPY --chown=ubuntu:ubuntu app /opt/inference/app
RUN ls -lah /opt/inference/app

# Download models at build time to avoid cold starts by downloading at runtime
RUN --mount=type=cache,target=/models,uid=1000,gid=1000 \
    uv run -m app.src.inference.install

EXPOSE 8000

# Single worker handles concurrent requests via async/thread pool
# Scale workers only if needed after profiling (1 worker = 3 models in memory)
# GPU will serialise requests anyway, so multiple workers will only help if you have multiple GPUs
CMD ["uv", "run", "fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
