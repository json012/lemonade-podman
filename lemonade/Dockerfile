FROM ghcr.io/ggml-org/llama.cpp:server-vulkan

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget unzip \
 && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/lemonade-sdk/lemonade/releases/latest/download/lemonade-server-minimal_9.1.1_amd64.deb
RUN dpkg -i lemonade-server-minimal_9.1.1_amd64.deb

# Add to PATH
ENV PATH="/usr/local/lemonade/bin:${PATH}"

# Lemonade runtime envs
ENV LEMONADE_HOST=0.0.0.0 \
    LEMONADE_PORT=8000 \
    LEMONADE_LLAMACPP=vulkan \
    LEMONADE_LLAMACPP_VULKAN_BIN=/app/llama-server

EXPOSE 8000

ENTRYPOINT ["lemonade-server"]
CMD ["serve", "--llamacpp", "vulkan"]
