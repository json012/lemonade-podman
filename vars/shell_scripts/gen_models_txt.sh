#!/bin/bash

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

# Check if models were provided as argument
if [ $# -gt 0 ]; then
    # Models provided as argument, use them directly
    models="$1"
    echo "Using provided models: $models"
else
    # No argument provided, fetch available models and prompt user
    models=$(curl -s https://raw.githubusercontent.com/lemonade-sdk/lemonade/refs/heads/main/src/lemonade_server/server_models.json | jq -r 'to_entries
           | map(select(.value.recipe == "llamacpp" and .value.suggested == true))
           | .[] | .key')

    echo "Available models: $models"
    echo "Choose models (comma separated key names): "
    read models
fi

# Convert comma-separated to space-separated
models=$(echo "$models" | sed 's/,/ /g')

echo "$models" > "$OUTPUT_DIR/models.txt"

echo "Models saved to $OUTPUT_DIR/models.txt"
