# DECS Local LLM — PAC / slim launcher

A small, **per-user, localhost-only** launcher for existing vLLM, Open WebUI, and Aider installations on MSU lab servers. No model weights, authentication tokens, or shared live environments are stored in this repo.

This tool is a convenience wrapper, **not an installer**. `doctor` checks whether the separate dependencies are installed. It has been designed around the recovered June 2026 setup and tested with CPU-only fake servers; a live PAC test is still needed.

## Quick start (on PAC)

```bash
export PAC_LLM_ROOT=/egr/research-pac/$USER
cd "$PAC_LLM_ROOT"
git clone git@github.com:Cheng-HanHuang/decs_local_llm.git
chmod +x "$PAC_LLM_ROOT/decs_local_llm/bin/pac-llm"
mkdir -p "$PAC_LLM_ROOT/local_llm/bin"
ln -s "$PAC_LLM_ROOT/decs_local_llm/bin/pac-llm" "$PAC_LLM_ROOT/local_llm/bin/pac-llm"
export PATH="$PAC_LLM_ROOT/local_llm/bin:$PATH"

pac-llm doctor
pac-llm status
pac-llm start web
pac-llm start coder --gpu 0
pac-llm status
pac-llm logs coder --lines 15
```

For a shared installation, each person clones the repo (or uses a shared **read-only source** checkout), but runs with their own `$USER`, environments, caches and data directories. Do **not** share `$PAC_LLM_ROOT/local_llm` between users.

To make the command available in future login shells, add to *your own* `.bashrc`:

```bash
export PAC_LLM_ROOT=/egr/research-pac/$USER
export PATH="$PAC_LLM_ROOT/local_llm/bin:$PATH"
```

If you already have a `decs_local_llm` checkout, use `git pull --ff-only` rather than cloning again. If the `pac-llm` symlink exists, inspect it before replacing it.

## Everyday commands

```bash
pac-llm status                      # service PID and API readiness
pac-llm doctor                      # paths + live GPU free VRAM
pac-llm start web                   # CPU-only, localhost:8080
pac-llm start coder --gpu 0         # Qwen3 Coder on GPU 0, localhost:8000
pac-llm logs coder --lines 30       # short log tail, safe with huge logs
pac-llm aider                       # original ask-mode Aider wrapper, current repo
pac-llm stop coder                  # only our managed process group
pac-llm stop web
```

The large Qwen model can take several minutes to load from NFS. `start` **returns immediately** with PID/log paths; use `status` and `logs`, and do not start it repeatedly. vLLM `--enable-auto-tool-choice --tool-call-parser qwen3_xml` is on for the default coder model, matching the working November setup. A custom model must opt in with `--tool-parser`.

The separate SSH tunnel on the Windows laptop remains:

```powershell
ssh -N -L 18080:127.0.0.1:8080 huang248@pac
```

Then open `http://localhost:18080` in a browser. Replace the host alias as needed. The web interface stays bound to `127.0.0.1` on PAC; never expose it or the vLLM API publicly without an appropriate authentication/network design.

## Select another model or GPU

```bash
export PAC_LLM_ROOT=/egr/research-pac/$USER
export PAC_LLM_GPU=1
pac-llm start coder               # equivalent to --gpu 1

# Advanced: custom model (tool calling disabled unless parser provided)
pac-llm start coder --gpu 1 --model organization/model-name --served-name my-model

# Two-GPU reasoning model; much larger VRAM requirement
pac-llm start reasoner --gpus 0,1  # localhost:8001, no default tool parser
```

Available environment overrides:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PAC_LLM_ROOT` | auto-detect `/egr/research-pac/$USER` or slim | per-user research root |
| `PAC_LLM_BASE` | `$PAC_LLM_ROOT/local_llm` | caches, logs, run files, web UI data |
| `PAC_LLM_VLLM_ENV` | `$PAC_LLM_ROOT/conda-envs/local-agent` | vLLM + Ninja environment |
| `PAC_LLM_WEBUI_ENV` | `$PAC_LLM_ROOT/conda-envs/open-webui` | separate Open WebUI environment |
| `PAC_LLM_GPU` | unset | default one-GPU selection |
| `PAC_LLM_GPUS` | unset | default multi-GPU selection, e.g. `0,1` |
| `PAC_LLM_MODEL` | per-profile model | model override (prefer `--model`) |
| `PAC_LLM_OFFLINE` | `1` | set to `0` to permit Hugging Face downloads |
| `PAC_LLM_CODER_PORT` | `8000` | vLLM coding API port |
| `PAC_LLM_REASONER_PORT` | `8001` | vLLM reasoning API port |
| `PAC_LLM_WEB_PORT` | `8080` | Open WebUI port |

The default coder is `Qwen/Qwen3-Coder-30B-A3B-Instruct`, BF16, 16k context, 0.72 GPU memory utilization. The default reasoner is `Qwen/Qwen3-Next-80B-A3B-Thinking`, BF16, tensor parallel 2, 16k context, 0.88 memory utilization. These are **starting configurations**, not guarantees that a GPU has enough free memory. Custom models may need different dtype, context, or tool parser; the launcher currently uses BF16.

## Co-tenancy and safety

- **No waiting for idle GPUs**: explicitly choose GPU(s). The launcher reports utilization but does not reject high utilization by itself.
- It does a conservative *snapshot* of free VRAM: each GPU needs `GPU total × gpu-memory-utilization + 2 GiB`. If it fails, choose other GPUs or use `--force` knowing the risks. VRAM can change immediately after inspection; no check can guarantee co-tenancy safety.
- The launcher refuses to claim an occupied API port, even if the process was started manually. It does not kill unmanaged processes.
- Run metadata is per host and per user; before signaling a process group, it validates PID, Linux process starttime, owner, and process-group leader. `stop --force` is for a **verified managed** group only.
- All logs and WebUI credentials stay under the research root, never in Git. WebUI accounts and chats persist under `$PAC_LLM_BASE/webui/data`. The launcher reuses the existing secret key and data directory.
- Aider uses your existing `$PAC_LLM_BASE/run_aider_local.sh`, which defaults to read-only ask mode and port 8000. Using a different coding port requires updating that old wrapper.
- If you started vLLM/Open WebUI manually before adopting this launcher, shut them down by their **original terminal or known PID**. The launcher will report the port as unmanaged and refuse to kill it.
- Open WebUI's RAG/embedding setup is independent of chat inference; an expired Hugging Face token may break document retrieval without preventing chat. The launcher disables implicit Hugging Face tokens only in the WebUI process.
- Enabling auto-tool-choice lets models request tools. Review Open WebUI workspace tools/skills permissions and verify claimed actions; a model saying “I created a skill” is not evidence it actually did.

## Test locally without GPUs

```bash
python3 -m unittest discover -s tests -v
```

Tests use fake `nvidia-smi` and fake HTTP serving scripts, so they never load a model.

## Known limits

- This intentionally does not install vLLM, Open WebUI, or model weights; follow the lab-specific setup guide first.
- On a fresh installation without `env_local_agent.sh`, it sets a minimal subset of cache locations; review other XDG/HF caches before downloading models.
- If two profiles are started, Open WebUI's current single backend points at coder. Add reasoning API at `http://127.0.0.1:8001/v1` in **Admin → Settings → Connections** as needed.
- Process-state management is Linux-specific (`/proc`), and service metadata is per machine. Shared NFS storage does not mean a localhost service is reachable on another host.
