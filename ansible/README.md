Ansible playbook for Lemonade stack setup
=========================================

This playbook provides idempotent deployment and management of the Lemonade stack with interactive configuration and automatic file generation.

Run locally (defaults to localhost):

```
ansible-playbook -i inventory.ini playbook.yml
```

Run remotely (no remote details in git; relies on ~/.ssh/config):

```
ansible-playbook -i inventory.ini -i 'my-ssh-alias,' playbook.yml -e target_host=my-ssh-alias
```

Variables:

- `target_host` (default: `localhost`) - target host for remote deployment
- `enable_online_hosting` (default: `false`) - set to `true` to enable traefik, cloudflare tunnel and online access

What it does:

**Prerequisites:**
- Checks for required packages: `jq`, `openssl`, `curl`, `sed`, `podman`
- Creates necessary directories: systemd, conf, and generated conf directories

**Interactive Configuration:**
- Fetches available Lemonade models from GitHub repo and prompts for model selection
- If online hosting enabled, prompts for LiteLLM domain name and Cloudflare tunnel token
- Generates configuration files using shell scripts in `../vars/shell_scripts/`

**File Management:**
- Ensures shell scripts have execute permissions
- Generates missing configuration files:
  - `models.txt` - selected models list
  - `litellm-db.env`, `litellm.env` - environment files
  - `litellm.yaml` - LiteLLM configuration
  - `traefik-hosts.yaml`, `traefik.yaml`, `cloudflared.env` (if online hosting enabled)
- Copies generated files to user configuration directory
- Copies quadlet files from `../quadlets/` to user systemd directory

**Service Management:**
- Reloads user systemd daemon
- Restarts `lemonade@<model>.service` for each selected model
- Restarts core services: `litellm-db`, `litellm` (and `traefik`, `cloudflared` if online hosting enabled)
- Cleans up generated temporary files
