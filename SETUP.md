# Lab onboarding (PAC / slim)

This repo contains **code and documentation only**. Each lab member owns a separate installation and model cache.

For the **complete current student walkthrough and automated installer**, follow
[WORKSHOP.md](WORKSHOP.md). The live-tested teaching model is
**Qwen3-4B-Instruct-2507** (Hermes parser).

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
git clone --branch feature/pac-lab-autopilot-4b \
  git@github.com:Cheng-HanHuang/decs_local_llm.git
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

## 3. Automated installation (recommended for lab members)

From the repository checkout:

```bash
bash scripts/lab_install.sh plan
bash scripts/lab_install.sh start
bash scripts/lab_install.sh status
bash scripts/lab_install.sh logs
bash scripts/lab_install.sh verify    # after Status: complete
```

This installs vLLM, Open WebUI, Aider, OpenCode, LoRA/PEFT and both
public Qwen3 models; it prepares training data. It runs CPU tests,
keeps per-stage logs and resumes after interruption. It does not start
GPU jobs or train. See [WORKSHOP.md](WORKSHOP.md).

**Manual option:** `pac-llm setup` previews individual components,
and `pac-llm setup --components vllm,web,aider,opencode --yes`
installs them explicitly. The manual option does not create the LoRA
training environment or download both models.

`setup` checks for missing executables and preserves existing environments and the existing Aider wrapper. It captures large package logs to `$PAC_LLM_BASE/logs/<hostname>/setup.log`. Conda must be available in your shell.

The initial pinned candidates are vLLM 0.23.0 with Python 3.11 plus
Ninja, Open WebUI 0.13.0 with Python 3.12 plus **torch 2.8.0 CPU**,
Aider 0.86.2, OpenCode 2.2.5 and Node 22. **Full fresh-user installation has not been validated**, especially the
WebUI dependency resolution. Other CUDA drivers may require adaptation.
This is not a fully locked environment.

Model weights are **not** downloaded by ordinary `pac-llm setup`
unless requested; the lab autopilot explicitly downloads the two
public models. To download just the workshop coder manually, use
`pac-llm setup --components vllm --download-coder --yes`.
The autopilot disables implicit Hugging Face credentials for public
downloads. Never commit tokens, weights or chats.

## 4. Everyday use

```bash
pac-llm status
pac-llm start web
pac-llm start coder                   # tested 4B default; uses saved GPU, port
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
- The 4B default uses `hermes` for structured tool calls, tested live
  on PAC. A plain chat response is not a tool-calling test; run
  `scripts/pac_workshop_smoke.py --test-tools`.
- Existing per-user settings override defaults. Check `pac-llm config show`
  before a workshop; do not reuse other people's ports.
- Inference is offline by default once model weights are cached. RAG embeddings/downloads are separate from inference.
