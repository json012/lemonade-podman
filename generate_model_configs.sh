#!/bin/bash

# Generate model configuration files
# This script creates litellm.yaml and docker-compose.proxied-models.yml

echo "Fetching models data..."
models_data=$(curl -s https://raw.githubusercontent.com/lemonade-sdk/lemonade/refs/heads/main/src/lemonade_server/server_models.json)

# Parse command line arguments
available_gpu_memory="32"  # Default to 32GB GPU memory
models=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-memory|-g)
            available_gpu_memory="$2"
            shift 2
            ;;
        --memory|-m)
            echo "Warning: --memory is deprecated. Use --gpu-memory instead."
            available_gpu_memory="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--gpu-memory GB] [MODELS...]"
            echo "  --gpu-memory, -g  Available GPU memory in GB (default: 32)"
            echo "  --memory, -m      Deprecated: use --gpu-memory instead"
            echo "  MODELS            Comma-separated list of model names"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --gpu-memory 24 Gemma-3-4b-it-GGUF,nomic-embed-text-v2-moe-GGUF"
            echo "  $0 --gpu-memory 64  # Interactive mode with 64GB GPU memory"
            echo ""
            echo "Note: This script allocates based on GPU memory constraints, not system RAM."
            echo "      System RAM should be sufficient for Docker overhead and model loading."
            echo "      NO DOCKER RESOURCE LIMITS - Let system use memory naturally."
            exit 0
            ;;
        *)
            if [ -z "$models" ]; then
    models="$1"
            else
                models="$models,$1"
            fi
            shift
            ;;
    esac
done

# If no models provided, fetch available models and prompt user
if [ -z "$models" ]; then
    echo "Available models:"
    echo "$models_data" | jq -r 'to_entries
           | map(select(.value.recipe == "llamacpp" and .value.suggested == true))
        | .[] | .key'
    echo "Choose models (comma separated key names): "
    read models
fi

# Convert comma-separated to space-separated
models=$(echo "$models" | sed 's/,/ /g')

# Function to calculate RPM based on model size and type
calculate_rpm() {
    local model_size=$1
    local is_embeddings=$2
    local max_parallel=$3
    
    # Convert size to integer for comparison (multiply by 100 to handle decimals)
    local size_int=$(echo "$model_size" | awk '{print int($1 * 100)}')
    
    if [ "$is_embeddings" = "true" ]; then
        # Embedding models: higher RPM due to faster processing
        echo $((max_parallel * 20))  # 20 requests per minute per parallel slot
    else
        # Calculate RPM based on model size and parallel capacity
        if [ "$size_int" -lt 400 ]; then
            # Small models (<4GB): faster processing, higher RPM
            echo $((max_parallel * 15))  # 15 requests per minute per parallel slot
        elif [ "$size_int" -lt 800 ]; then
            # Medium models (4-8GB): moderate processing speed
            echo $((max_parallel * 12))  # 12 requests per minute per parallel slot
        elif [ "$size_int" -lt 2000 ]; then
            # Large models (8-20GB): slower processing
            echo $((max_parallel * 8))   # 8 requests per minute per parallel slot
        else
            # Very large models (>20GB): much slower processing
            echo $((max_parallel * 5))   # 5 requests per minute per parallel slot
        fi
    fi
}


# Function to calculate dynamic resource allocation based on available GPU memory
# MAX 2 INSTANCES PER MODEL - VRAM optimization for 96GB total
calculate_resources() {
    local model_size=$1
    local is_embeddings=$2
    local available_gpu_mem=$3
    
    # Convert size to integer for comparison (multiply by 100 to handle decimals)
    local size_int=$(echo "$model_size" | awk '{print int($1 * 100)}')
    
    if [ "$is_embeddings" = "true" ]; then
        # Embedding models: lightweight, minimal GPU resources
        echo "2 2"  # instances, max_parallel (no memory limits)
    else
        # Calculate instances based on available GPU memory - MAX 2 INSTANCES PER MODEL
        local instances
        local max_parallel
        
        if [ "$size_int" -lt 400 ]; then
            # Small models (<4GB): Q4_K_M quantization
            # MAX 2 instances per model for VRAM optimization
            if [ "$available_gpu_mem" -ge 96 ]; then
                instances=2
                max_parallel=3  # 2 instances × 3 = 6 total for this model
            elif [ "$available_gpu_mem" -ge 48 ]; then
                instances=2
                max_parallel=2  # 2 instances × 2 = 4 total for this model
            else
                instances=1
                max_parallel=4  # 1 instance × 4 = 4 total for this model
            fi
        elif [ "$size_int" -lt 800 ]; then
            # Medium models (4-8GB): Q4_1 quantization
            # MAX 2 instances per model for VRAM optimization
            if [ "$available_gpu_mem" -ge 96 ]; then
                instances=2
                max_parallel=2  # 2 instances × 2 = 4 total for this model
            else
                instances=1
                max_parallel=3  # 1 instance × 3 = 3 total for this model
            fi
        elif [ "$size_int" -lt 2000 ]; then
            # Large models (8-20GB): Q4_0 quantization
            # MAX 2 instances per model for VRAM optimization
            if [ "$available_gpu_mem" -ge 96 ]; then
                instances=2
                max_parallel=2  # 2 instances × 2 = 4 total for this model
            else
                instances=1
                max_parallel=3  # 1 instance × 3 = 3 total for this model
            fi
        else
            # Very large models (>20GB): Q4_0 quantization
            # Very large models: always 1 instance, adjust parallel based on GPU memory
            instances=1
            if [ "$available_gpu_mem" -ge 96 ]; then
                max_parallel=4  # 1 instance × 4 = 4 total for this model
            elif [ "$available_gpu_mem" -ge 64 ]; then
                max_parallel=3  # 1 instance × 3 = 3 total for this model
            else
                max_parallel=2  # 1 instance × 2 = 2 total for this model
            fi
        fi
        
        echo "$instances $max_parallel"
    fi
}

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
  allow_requests_on_db_unavailable: false
  disable_error_logs: true
  database_connection_pool_limit: 30
  max_parallel_requests: 20

litellm_settings:
  set_verbose: false
  request_timeout: 600

  # Caching settings
  cache: true 
  cache_params:
    type: redis
    host: litellm-redis
    port: 6379
    password: os.environ/LITELLM_REDIS_PASSWORD
    namespace: litellm.caching.caching

router_settings:
  redis_host: litellm-redis
  redis_password: os.environ/LITELLM_REDIS_PASSWORD
  redis_port: 6379
  optional_pre_call_checks: ["responses_api_deployment_check"]
  fallbacks: [
    {"gpt-oss-120b-mxfp-GGUF": ["gpt-oss-20b-mxfp4-GGUF", "o3-mini"]},
    {"gpt-oss-20b-mxfp4-GGUF": ["o3-mini"]},
  ]

guardrails:
  - guardrail_name: "inference/prompt_injection"
    litellm_params:
      guardrail: custom_guardrail.CustomGuardrailAPI
      mode: "pre_call"
      default_on: false
  - guardrail_name: "inference/non_english"
    litellm_params:
      guardrail: custom_guardrail.CustomGuardrailAPI
      mode: "pre_call"
      default_on: false

# Model routing configuration
model_list:
EOF

# Generate docker-compose.proxied-models.yml
echo "Creating docker-compose.proxied-models.yml..."
cat > ./docker-compose.proxied-models.yml << 'EOF'
services:
EOF

# Process each model
for model in $models; do
    echo "Processing model: $model"
    
    # Check if model exists in the data
    if ! echo "$models_data" | jq -e --arg model "$model" 'has($model)' > /dev/null; then
        echo "Warning: Model '$model' not found in models data, skipping..."
        continue
    fi
    
    # Get model information
    is_embeddings=$(echo "$models_data" | jq -r --arg model "$model" '.[$model].labels | contains(["embeddings"])')
    
    # Get model size to determine number of instances
    model_size=$(echo "$models_data" | jq -r --arg model "$model" '.[$model].size // "unknown"')
    
    
    # Calculate dynamic resource allocation based on available GPU memory
    resource_info=$(calculate_resources "$model_size" "$is_embeddings" "$available_gpu_memory")
    num_instances=$(echo $resource_info | awk '{print $1}')
    max_parallel=$(echo $resource_info | awk '{print $2}')
    
    # Create multiple instances for load balancing with weighted routing
    for i in $(seq 1 $num_instances); do
        if [ $num_instances -eq 1 ]; then
            # Single instance - use original naming
            container_name="lemonade-$model"
            port_suffix="0"
        else
            # Multiple instances - add instance number
            container_name="lemonade-$model-$i"
            port_suffix="$i"
        fi
        
        # Calculate weight for load balancing (only needed for multiple instances)
        if [ $num_instances -gt 1 ]; then
            # 2 instances: primary gets higher weight, secondary gets lower weight
            if [ $i -eq 1 ]; then
                weight=7
            else
                weight=3
            fi
        fi
        
        # Calculate RPM based on model size and parallel capacity
        rpm=$(calculate_rpm "$model_size" "$is_embeddings" "$max_parallel")
        
        # Add to litellm.yaml
        if [ $num_instances -eq 1 ]; then
            # Single instance - no weight or RPM needed
            cat >> ./litellm.yaml << EOF
  - model_name: "$model"
    litellm_params:
      model: "openai/$model"
      api_base: "http://$container_name:8000/api/v1"
      api_key: "lemonade"
      max_parallel_requests: $max_parallel
EOF
        else
            # Multiple instances - include weight and RPM for load balancing
            cat >> ./litellm.yaml << EOF
  - model_name: "$model"
    litellm_params:
      model: "openai/$model"
      api_base: "http://$container_name:8000/api/v1"
      api_key: "lemonade"
      max_parallel_requests: $max_parallel
      weight: $weight
      rpm: $rpm
EOF
        fi
        
        # Add embedding mode if it's an embedding model
        if [ "$is_embeddings" = "true" ]; then
            cat >> ./litellm.yaml << EOF
    model_info:
      mode: embedding
EOF
        fi
        
        # Add to docker-compose.proxied-models.yml - NO RESOURCE CONSTRAINTS
        cat >> ./docker-compose.proxied-models.yml << EOF
  $container_name:
    container_name: $container_name
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
      # Vulkan needs /dev/dri; ROCm would need /dev/kfd too (not used here)
      - /dev/dri:/dev/dri
    restart: always
    command: [
      "uv",
      "run",
      "lemonade-server-dev",
      "run",
      "$model",
      "--llamacpp",
      "vulkan",
      "--ctx-size",
      "32768"
    ]
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 5s
      start_period: 15s

EOF
    done
done


echo "Configuration files generated successfully!"
echo "- ./litellm.yaml"
echo "- ./docker-compose.proxied-models.yml"
echo ""
echo "Usage: docker-compose -f docker-compose.yml -f docker-compose.proxied-models.yml up"
