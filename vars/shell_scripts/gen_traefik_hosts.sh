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

# Check required files exist
if [ ! -f "$SCRIPT_DIR/../example/traefik-hosts.sample.yaml" ]; then
    echo "traefik-hosts.sample.yaml not found. This shouldn't happen."
    exit 1
else
    SAMPLE_HOSTS_FILE="$SCRIPT_DIR/../example/traefik-hosts.sample.yaml"
fi

if [ ! -f "$OUTPUT_DIR/litellm.env" ]; then
    echo "litellm.env not found in $OUTPUT_DIR. Run gen_env_files.sh first."
    exit 1
else
    source "$OUTPUT_DIR/litellm.env"
fi

# Create a copy of the sample file to modify
cp "$SAMPLE_HOSTS_FILE" "$OUTPUT_DIR/traefik-hosts.yaml"

# Find and replace example.com with LITELLM_HOST
sed -i "s/example.com/${LITELLM_HOST}/g" "$OUTPUT_DIR/traefik-hosts.yaml"

echo "$OUTPUT_DIR/traefik-hosts.yaml generated"
