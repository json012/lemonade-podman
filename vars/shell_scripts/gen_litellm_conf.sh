#!/bin/bash

# Generate litellm.yaml from models.txt

# get the directory of the script
SCRIPT_DIR=$(dirname "$0")

# If we're in a temp directory (contains _conf_gen), use it as working dir
if [[ "$PWD" == *"_conf_gen"* ]]; then
    WORKING_DIR="$PWD"
    OUTPUT_DIR="$PWD"
else
    WORKING_DIR="$SCRIPT_DIR"
    # Determine output directory - use generated if it exists, otherwise user config
    if [ -d "$SCRIPT_DIR/../generated" ]; then
        OUTPUT_DIR="$SCRIPT_DIR/../generated"
    else
        OUTPUT_DIR="$HOME/.config/containers/vars"
        mkdir -p "$OUTPUT_DIR"
    fi
fi

if [ -f "$OUTPUT_DIR/models.txt" ]; then
    models=$(cat "$OUTPUT_DIR/models.txt")
else
    echo "models.txt not found in $OUTPUT_DIR"
    exit 1
fi

# Generate litellm.yaml
cat > "$OUTPUT_DIR/litellm.yaml" << 'EOF'
# LiteLLM Configuration File
# This file configures the LiteLLM proxy server

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
  store_model_in_db: false
  set_verbose: false

# Model routing configuration
model_list:
EOF

# Add each model to the configuration
for model in $models; do
    if [ -n "$model" ]; then
        cat >> "$OUTPUT_DIR/litellm.yaml" << EOF
  - model_name: "$model"
    litellm_params:
      model: "openai/$model"
      api_base: "http://lemonade-$model:8000/api/v1"
      api_key: "lemonade"
EOF
    fi
done

echo "$OUTPUT_DIR/litellm.yaml generated"
