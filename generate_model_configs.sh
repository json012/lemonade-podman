#!/bin/bash

# Generate model configuration files
# This script creates litellm.yaml and docker-compose.proxied-models.yml
# based on the selected models

set -e

# Check if models were provided as argument
if [ $# -gt 0 ]; then
    # Models provided as argument, use them directly
    models="$1"
    echo "Using provided models: $models"
else
    # No argument provided, fetch available models and prompt user
    echo "Fetching available models..."
    models=$(curl -s https://raw.githubusercontent.com/lemonade-sdk/lemonade/refs/heads/main/src/lemonade_server/server_models.json | jq -r 'to_entries
           | map(select(.value.recipe == "llamacpp" and .value.suggested == true))
           | .[] | .key')

    echo "Available models: $models"
    echo "Choose models (comma separated key names): "
    read models
fi

# Convert comma-separated to space-separated
models=$(echo "$models" | sed 's/,/ /g')

echo "Generating configuration files for models: $models"

# Generate litellm.yaml
echo "Creating litellm.yaml..."
cat > ./litellm.yaml << 'EOF'
# LiteLLM Configuration File
# This file configures the LiteLLM proxy server

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
  store_model_in_db: true
  set_verbose: false

# Model routing configuration
model_list:
EOF

# Add each model to the litellm.yaml configuration
for model in $models; do
    if [ -n "$model" ]; then
        cat >> ./litellm.yaml << EOF
  - model_name: "$model"
    litellm_params:
      model: "openai/$model"
      api_base: "http://lemonade-$model:8000/api/v1"
      api_key: "lemonade"
EOF
    fi
done

# Generate docker-compose.proxied-models.yml
echo "Creating docker-compose.proxied-models.yml..."
cat > ./docker-compose.proxied-models.yml << 'EOF'
# Model-specific lemonade services
# This file extends docker-compose.yml to add specific model instances
# Usage: docker-compose -f docker-compose.yml -f docker-compose.proxied-models.yml up
# Note: This file will not work as a standalone file, it must be chained after docker-compose.yml

services:
EOF

# Add each model service to the docker-compose configuration
for model in $models; do
    if [ -n "$model" ]; then
        cat >> ./docker-compose.proxied-models.yml << EOF
  lemonade-$model:
    container_name: lemonade-$model
    image: ghcr.io/json012/lemonade:latest
    pull_policy: newer
    user: ubuntu
    environment:
      LEMONADE_LLAMACPP: vulkan
      LEMONADE_HOST: 0.0.0.0
      LEMONADE_CACHE: /home/ubuntu/.cache
    volumes:
      - lemonade-cache:/home/ubuntu/.cache/huggingface/hub
    networks:
      - litellm-network
    devices:
      # Vulkan needs the render node; ROCm/HIP would need /dev/kfd (not used here)
      - /dev/dri/renderD128:/dev/dri/renderD128
    restart: always
    command: [
      "uv",
      "run",
      "lemonade-server-dev",
      "run",
      "$model",
      "--llamacpp",
      "vulkan"
    ]
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 5s
      start_period: 15s

EOF
    fi
done

echo "Configuration files generated successfully!"
echo "- ./litellm.yaml"
echo "- ./docker-compose.proxied-models.yml"
echo ""
echo "Usage: docker-compose -f docker-compose.yml -f docker-compose.proxied-models.yml up"
