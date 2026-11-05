# Lab onboarding (PAC / slim)

This repo contains **code and documentation only**. Each lab member owns a separate installation and model cache.

## Configuration scopes

| Scope | Examples | Where |
|---|---|---|
| Shared | CLI code, version pins, safety rules | Git checkout |
| Per-user installation | Conda envs, Hugging Face cache, WebUI history/accounts, logs | `/egr/research-pac/$USER` (or slim) |
| Per-user defaults | GPU, ports, default model, offline mode | `$PAC_LLM_BASE/config/pac-llm.json` (0600) |
| Per-run choices | `--gpu`, `--gpus`, `--model`, `--max-model-len`, `--memory-util` | CLI |
| Per-shell overrides | `PAC_LLM_*` environment variables | Shell |

**Priority:** CLI flags > exported environment variables > saved per-user defaults > built-in defaults. `PAC_LLM_ROOT` selects the research root; by default this is `/egr/research-pac/$USER` or `/egr/research-slim/$USER`.

## 1. Get the launcher

```bash
export PAC_LLM_ROOT=/egr/research-pac/$USER
cd "$PAC_LLM_ROOT"
git clone git@github.com:Cheng-HanHuang/decs_local_llm.git
mkdir -p "$PAC_LLM_ROOT/local_llm/bin"
ln -s "$PAC_LLM_ROOT/decs_local_llm/bin/pac-llm" "$PAC_LLM_ROOT/local_llm/bin/pac-llm"
export PATH="$PAC_LLM_ROOT/local_llm/bin:$PATH"
```

The repository is private. Each colleague needs GitHub read access (or an approved public copy). Do **not** share a writable live `local_llm` environment among users.

## 2. Personal defaults

Pick **unique ports for each user on the shared host**. Loopback ports are not isolated by Linux account.

```bash
pac-llm config init --gpu 1 --coder-port 18100 --reasoner-port 18101 --web-port 18180
pac-llm config show
pac-llm config set gpu 2
pac-llm config set coder_port 18102
```

Ports are illustrative and not reserved. The launcher checks port availability when starting.

## 3. Explicit installation

```bash
pac-llm setup                                   # preview only
pac-llm setup --components vllm,web,aider --yes  # opt-in installation
pac-llm doctor
```

`setup` checks for missing executables and preserves existing environments and the existing Aider wrapper. It captures large package logs to `$PAC_LLM_BASE/logs/<hostname>/setup.log`. Conda must be available in your shell.

The pinned initial targets correspond to an observed working PAC environment (November 2026): vLLM 0.23.0 with Python 3.11 plus Ninja; Open WebUI 0.13.0 with Python 3.12 and CPU PyTorch; Aider 0.86.2 with Python 3.12. **Other hardware or CUDA drivers may require adaptation.** This is not a fully locked dependency environment.

Model weights are **not** downloaded by default. Downloads can be tens to hundreds of GiB:

```bash
pac-llm setup --components vllm --download-coder --yes
# or
pac-llm setup --components vllm --download-reasoner --yes
```

Use `hf auth login` under your own account if accessing gated models. Never commit tokens, weights or chats.

## 4. Everyday use

```bash
pac-llm status
pac-llm start web
pac-llm start coder                   # uses saved GPU, port
pac-llm start coder --gpu 3           # temporary override
pac-llm logs coder --lines 20
pac-llm aider                         # ask-mode Aider
pac-llm stop coder
```

Windows tunnel example, replacing remote port if configured differently:

```powershell
ssh -N -L 19080:127.0.0.1:18180 your-pac-alias
```

Visit http://localhost:19080 in the laptop browser.

## Safety and limits

- All services bind to `127.0.0.1`. Do not expose them to the public network.
- The GPU check reports high utilization and rejects insufficient available VRAM; it cannot reserve VRAM against concurrent users.
- An occupied port is reported as `unmanaged port` unless the launcher owns the process. It never kills unrelated services.
- `setup` only installs *missing executables*; it does not upgrade or overwrite environments. Setup can require significant disk space. Test compatibility before wide lab rollout.
- `pac-llm aider` calls the per-user uv or Conda Aider binary directly, uses the configured coder port, and defaults to ask-only mode. The old June wrapper is preserved for manual usage, but it still hardcodes port 8000; prefer `pac-llm aider` for custom ports.
- The default tool parser `qwen3_xml` is configured only for the known Qwen3-Coder model.
- Inference is offline by default once model weights are cached. RAG embeddings/downloads are separate from inference.
