This is a configuration for running [Lemonade](https://www.amd.com/en/developer/resources/technical-articles/unlocking-a-wave-of-llm-apps-on-ryzen-ai-through-lemonade-server.html) in a container. It also includes [Podman Quadlets](https://www.redhat.com/en/blog/quadlet-podman) service files for running Lemonade as a user systemd service on Fedora etc. with AMD Ryzen AI 300 series processors.

Awaiting https://github.com/lemonade-sdk/lemonade/issues/5 for a performance boost with [NPU hybrid inference](https://lemonade-server.ai/docs/faq/#1-does-hybrid-inference-with-the-npu-only-work-on-windows).

### Quickstart

Build the image (use `docker` if preferred):

```
podman build -t lemonade .
```

Run with AMD GPU devices and Vulkan backend:

```
podman run --rm -p 8000:8000 \
  --device /dev/kfd --device /dev/dri \
  localhost/lemonade:latest
```

Verify the server is up:

```
curl http://localhost:8000/api/v1/health
```

OpenAI-compatible API is available at `http://localhost:8000/api/v1/`.

### Quadlet (Fedora)

The quadlet configuration uses a published image (ghcr) by default.
For the local image, adjust the Image line in `quadlets/lemonade@.container`:

```
mkdir -p ~/.config/containers/systemd
cp quadlets/lemonade* ~/.config/containers/systemd/
systemctl --user daemon-reload
systemctl --user start lemonade.service
systemctl --user status lemonade.service
```

`podman ps` should show the container running. Visit `http://localhost:8000/` in browser for Lemonade GUI.

### Running With a Specific Model Pre-loaded

**Start with a specific model:**
```bash
systemctl --user start lemonade@Gemma-3-4b-it-GGUF.service
systemctl --user start lemonade@Qwen3-4B-GGUF.service
```

**Start without specifying a model (uses default behavior):**
```bash
systemctl --user start lemonade.service
```

**Check status of specific model instances:**
```bash
systemctl --user status lemonade@Gemma-3-4b-it-GGUF.service
systemctl --user status lemonade@Qwen3-4B-GGUF.service
```

**Stop specific model instances:**
```bash
systemctl --user stop lemonade@Gemma-3-4b-it-GGUF.service
systemctl --user stop lemonade@Qwen3-4B-GGUF.service
```

**View logs for specific models:**
```bash
journalctl --user -xeu lemonade@Gemma-3-4b-it-GGUF.service
journalctl --user -xeu lemonade@Qwen3-4B-GGUF.service
```

All instances run on port 8000. For multiple model instances, consider using a reverse proxy to route different paths or subdomains to different model instances.

### Runtime Requirements

- Linux x86_64 with Vulkan-capable GPU/driver
- Tested on Fedora 42 with AMD Ryzen AI 300 series
- Container runtime (Podman/Docker)

### Dev Requirements

- [uv](https://docs.astral.sh/uv/)
- Python >=3.10

### Troubleshooting

- Ensure `/dev/kfd` and `/dev/dri` are present and accessible to the container
- Validate Vulkan support: `vulkaninfo | head -n 20`
- Check logs: `podman logs -f lemonade` and/or `journalctl --user -xeu lemonade.service`

### Ansible-based setup (recommended)

For an idempotent setup, use the included Ansible playbook.

Quickstart (runs on localhost by default):

```
ansible-playbook -i ./ansible/inventory.ini /ansible/playbook.yml
```

To enable online hosting (requires litellm domain and cloudflare tunnel token):

```
cd ansible
ansible-playbook -i ./ansible/inventory.ini ./ansible/playbook.yml -e enable_online_hosting=true
```

To run against a remote host via SSH (using ~/.ssh/config):

```
ansible-playbook -i ./ansible/inventory.ini -i 'my-ssh-alias,' ./ansible/playbook.yml -e target_host=my-ssh-alias
```
