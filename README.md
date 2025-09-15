This is a configuration for running [Lemonade](https://www.amd.com/en/developer/resources/technical-articles/unlocking-a-wave-of-llm-apps-on-ryzen-ai-through-lemonade-server.html) as part of a [LiteLLM](https://docs.litellm.ai/) proxy stack. The setup includes Lemonade model backends, PostgreSQL database, and optional Traefik + Cloudflare integration for online hosting.

Awaiting https://github.com/lemonade-sdk/lemonade/issues/5 for a performance boost with [NPU hybrid inference](https://lemonade-server.ai/docs/faq/#1-does-hybrid-inference-with-the-npu-only-work-on-windows).

## Prerequisites

### System Requirements
- Linux x86_64 with Vulkan-capable GPU/driver
- Tested on Fedora 42 with AMD Ryzen AI 300 series
- Container runtime (Podman/Docker)

### User Setup
Before running the containers, ensure your user has proper permissions:

```bash
# Add user to render and video groups for GPU access
sudo usermod -a -G render,video $USER

# For login linger (allows user services to run without login session)
sudo loginctl enable-linger $USER

# Log out and back in for group changes to take effect
```

### Install Podman
```bash
# Fedora/RHEL/CentOS
sudo dnf install podman

# Ubuntu/Debian
sudo apt install podman

# Verify installation
podman --version
```

### Verify GPU Access
```bash
# Check Vulkan support
vulkaninfo | head -n 20

# Verify device files exist
# If device has a different name from renderD128 you will need to update the compose files to match.
ls -la /dev/dri/renderD128
ls -la /dev/kfd  # For ROCm/HIP (if available)
```

## Quickstart

### Configure Environment

You can copy and edit `docker-compose.proxied-models.example.yml` and `litellm.example.yaml`

```bash
cp docker-compose.proxied-models.example.yml docker-compose.proxied-models.yml
cp litellm.example.yaml litellm.yaml
```

Or, use the provided shell script to help automatically generate the model files:

```bash
# Generate model configurations
./generate_model_configs.sh
```

```bash
# Copy and edit environment file
cp sample.env .env
```

### Option 1: Automated Setup with Ansible (Recommended)

The easiest way to get started is using the included Ansible playbook:

Install ansible:

```bash
# Fedora/RHEL/CentOS
sudo dnf install ansible

# Ubuntu/Debian
sudo apt install ansible
```

```bash
# Basic local setup
ansible-playbook -i localhost, ./ansible-playbook.yml

# With online hosting (requires domain and Cloudflare tunnel token)
ansible-playbook -i localhost, ./ansible-playbook.yml -e enable_online_hosting=true

# With systemd service management (for persistence after reboot)
ansible-playbook -i localhost, ./ansible-playbook.yml -e use_systemd=true

# Remote deployment via SSH
ansible-playbook -i 'my-ssh-alias,' ./ansible-playbook.yml -e target_host=my-ssh-alias
```

### Option 2: Manual Setup

If you prefer to set up manually, follow these steps:


#### 2. Start the Stack

```bash
# Start all services
podman-compose -f docker-compose.yml -f docker-compose.proxied-models.yml up -d

# With online hosting
podman-compose -f docker-compose.yml -f docker-compose.proxied-models.yml -f docker-compose.online.yml up -d
```

#### 3. Verify Services

Access LiteLLM proxy at http://localhost:4000/

Access Lemonade at http://localhost:8000/

```bash
# View running containers
podman ps
```

## Architecture Overview

The setup includes:

- **LiteLLM Proxy** (port 4000): Main API endpoint that routes requests to appropriate model backends
- **PostgreSQL Database**: Stores model configurations and usage data
- **Lemonade Backends** (port 8000): Individual model instances running specific models
- **Traefik** (port 8080): Reverse proxy for online hosting (optional)
- **Cloudflared**: Cloudflare tunnel for external access (optional)

## Model Management

Note, until https://github.com/lemonade-sdk/lemonade/issues/5 is resolved, 
only the [GGUF (llamacpp recipe) models](https://lemonade-server.ai/docs/server/server_models/#gguf) can be used

### Adding New Models

Use the model configuration generator:

```bash
# Interactive mode - choose from available models
./generate_model_configs.sh

# Specify models directly
./generate_model_configs.sh "Gemma-3-4b-it-GGUF,Qwen3-4B-GGUF"
```

### Managing Model Instances

```bash
# View all running containers
podman ps

# View logs for specific model
podman logs -f lemonade-Gemma-3-4b-it-GGUF

# Restart a specific model
podman restart lemonade-Gemma-3-4b-it-GGUF

# Stop all services
podman-compose -f docker-compose.yml -f docker-compose.proxied-models.yml down
```

## API Usage

### LiteLLM Proxy (Main API)

The LiteLLM proxy provides a unified API for all models, preloaded. 
See [LiteLLM proxy documentation](https://docs.litellm.ai/docs/simple_proxy)

### Direct Lemonade Access

The Lemonade GUI is available at `http://localhost:8000`:

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Direct model access (if needed)
curl -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Gemma-3-4b-it-GGUF",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Note, it takes time to load a model into memory. Using the preloaded per-model Lemonade instances via LiteLLM will be faster to respond.

## Online Hosting

To enable online hosting with your own domain:

1. **Set up Cloudflare Tunnel**:
   - Create a tunnel in Cloudflare dashboard
   - Copy the tunnel token

2. **Configure Environment**:
   ```bash
   # Edit .env file
   TUNNEL_TOKEN=your_tunnel_token_here
   TRAEFIK_DOMAIN=your-domain.com
   ```

3. **Copy Traefik Configuration**:
   ```bash
   cp traefik-hosts.example.yaml traefik-hosts.yaml
   # Edit traefik-hosts.yaml with your domain
   ```

4. **Deploy with Online Hosting**:
   ```bash
   podman-compose -f docker-compose.yml -f docker-compose.proxied-models.yml -f docker-compose.online.yml up -d
   ```

## Troubleshooting

### Common Issues

**GPU Access Problems**:
```bash
# Check device permissions
ls -la /dev/dri/renderD128
ls -la /dev/kfd

# Verify user groups
groups $USER

# Test Vulkan support
vulkaninfo | head -n 20
```

### Performance Optimization

- Ensure sufficient GPU memory for your models
- Monitor resource usage: `podman stats`
- Adjust model configurations in `litellm.yaml` as needed
- Consider using systemd services for production deployments
