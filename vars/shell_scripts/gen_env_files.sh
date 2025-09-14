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

cat > "$OUTPUT_DIR/litellm-db.env" << EOF
POSTGRES_DB=litellm
POSTGRES_USER=litellm
POSTGRES_PASSWORD=$(openssl rand -hex 16)
EOF

source "$OUTPUT_DIR/litellm-db.env"

# Check if LiteLLM domain was provided as first argument
if [ $# -gt 0 ]; then
    # LiteLLM domain provided as argument, use it directly
    LITELLM_HOST="$1"
    echo "Using provided LiteLLM domain: $LITELLM_HOST"
else
    # No argument provided, prompt user
    read -p "Enter LiteLLM domain name: " LITELLM_HOST
fi

cat > "$OUTPUT_DIR/litellm.env" << EOF
LITELLM_HOST=${LITELLM_HOST}
LITELLM_MASTER_KEY=$(openssl rand -hex 32)
LITELLM_SALT_KEY=$(openssl rand -hex 32)
DATABASE_URL=postgres://${POSTGRES_USER}:${POSTGRES_PASSWORD}@litellm-db:5432/${POSTGRES_DB}
EOF

# Check if tunnel token was provided as second argument
if [ $# -gt 1 ]; then
    # Tunnel token provided as argument, use it directly
    TUNNEL_TOKEN="$2"
    echo "Using provided tunnel token"
else
    # No second argument provided, prompt user
    read -p "Enter Cloudflared tunnel token (https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/create-remote-tunnel/#2a-connect-an-application): " TUNNEL_TOKEN
fi

cat > "$OUTPUT_DIR/cloudflared.env" << EOF
TUNNEL_TOKEN=${TUNNEL_TOKEN}
EOF

echo "Environment files generated in $OUTPUT_DIR"
