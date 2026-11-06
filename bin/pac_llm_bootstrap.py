"""Explicit, per-user environment installer; never invoked by start/status."""
from __future__ import annotations
import os
from pathlib import Path
import shlex
import shutil
import subprocess

VLLM = "0.23.0"
WEBUI = "0.13.0"
AIDER = "0.86.2"
CPU_TORCH = "2.8.0"  # available CPU wheels; WebUI dependency compatibility needs PAC smoke

def executable(path):
    return path.is_file() and os.access(path, os.X_OK)


def existing_conda_operation(prefix):
    """Resume a partially initialized Conda prefix rather than re-create it."""
    return "install" if (prefix / "conda-meta").is_dir() else "create"


def environment_has_package(python, distribution):
    """Check metadata in its own environment, without importing a large library."""
    if not executable(python):
        return False
    code = ("from importlib.metadata import version; "
            f"version({distribution!r})")
    try:
        p = subprocess.run([str(python), "-c", code],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=False, timeout=15)
        return p.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def setup_steps(w, components, downloads):
    conda = shutil.which("conda") or "conda"
    steps = []
    if "vllm" in components:
        py = w.vllm_env / "bin/python"
        if not executable(py):
            steps.append(("Create/resume vLLM environment", [conda,
                existing_conda_operation(w.vllm_env), "-p", str(w.vllm_env),
                "python=3.11", "pip", "-y"]))
        if not executable(w.vllm_env / "bin/vllm"):
            steps.append(("Install pinned vLLM", [str(py), "-m", "pip", "install", f"vllm=={VLLM}"]))
        if not executable(w.vllm_env / "bin/ninja"):
            steps.append(("Install Ninja for FlashInfer", [str(py), "-m", "pip", "install", "ninja"]))
    if "web" in components:
        py = w.web_env / "bin/python"
        if not executable(py):
            steps.append(("Create/resume WebUI environment", [conda,
                existing_conda_operation(w.web_env), "-p", str(w.web_env),
                "python=3.12", "pip", "-y"]))
        # Partial setup can leave Python but not PyTorch. Install CPU-only
        # torch in that case, even on a rerun; preserve working environments.
        if not executable(w.web_env / "bin/open-webui") and not environment_has_package(py, "torch"):
            steps.append(("Install CPU PyTorch", [str(py), "-m", "pip", "install",
                          f"torch=={CPU_TORCH}", "--index-url",
                          "https://download.pytorch.org/whl/cpu"]))
        if not executable(w.web_env / "bin/open-webui"):
            steps.append(("Install pinned Open WebUI", [str(py), "-m", "pip", "install", f"open-webui=={WEBUI}"]))
    if "aider" in components:
        aider_env = w.root / "conda-envs/aider"
        py = aider_env / "bin/python"
        # June's working Aider is installed with uv at local_llm/bin/aider.
        # A valid standalone Aider is enough: no duplicate Conda environment.
        installed = executable(aider_env / "bin/aider") or executable(w.base / "bin/aider")
        if not installed:
            if not executable(py):
                steps.append(("Create/resume Aider environment", [conda,
                    existing_conda_operation(aider_env), "-p", str(aider_env),
                    "python=3.12", "pip", "-y"]))
            steps.append(("Install pinned Aider", [str(py), "-m", "pip", "install", f"aider-chat=={AIDER}"]))
        if not (w.base / "run_aider_local.sh").is_file():
            steps.append(("Write safe Aider wrapper", None))
    if "opencode" in components:
        node_env = w.root / "conda-envs/opencode-node"
        if not executable(node_env / "bin/node") or not executable(node_env / "bin/npm"):
            # Resume a partially created environment rather than failing
            # with "prefix already exists" on subsequent setup attempts.
            operation = "install" if (node_env / "conda-meta").is_dir() else "create"
            steps.append(("Install per-user Node.js 22 and npm",
                          [conda, operation, "-p", str(node_env), "-c", "conda-forge",
                           "nodejs=22", "-y"]))
        if not executable(w.base / "opencode/npm/bin/opencode"):
            steps.append(("Install OpenCode V2",
                          [str(node_env / "bin/npm"), "install", "-g",
                           "--prefix", str(w.base / "opencode/npm"),
                           "@opencode/cli@2.2.5"]))
    hf = w.vllm_env / "bin/hf"
    for model in downloads:
        steps.append((f"Download model weights: {model}", [str(hf), "download", model]))
    return steps

def write_aider_wrapper(w):
    path = w.base / "run_aider_local.sh"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('''#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$BASE")"
if [[ -f "$BASE/env_local_agent.sh" ]]; then source "$BASE/env_local_agent.sh"; fi
export HOME="$BASE/aider_home"
mkdir -p "$HOME"
PORT=8000
if [[ -n "$(printenv PAC_LLM_CODER_PORT || true)" ]]; then PORT="$(printenv PAC_LLM_CODER_PORT)"; fi
export OPENAI_API_BASE="http://127.0.0.1:$PORT/v1"
export OPENAI_API_KEY=dummy
export AIDER_CHECK_UPDATE=false AIDER_ANALYTICS_DISABLE=true
AIDER_BIN="$ROOT/conda-envs/aider/bin/aider"
if [[ ! -x "$AIDER_BIN" ]]; then AIDER_BIN="$BASE/bin/aider"; fi
if [[ -f AGENTS.md ]]; then
  exec "$AIDER_BIN" --model openai/local-qwen-coder --chat-mode ask --no-auto-commits --no-dirty-commits --no-check-update --read AGENTS.md "$@"
else
  exec "$AIDER_BIN" --model openai/local-qwen-coder --chat-mode ask --no-auto-commits --no-dirty-commits --no-check-update "$@"
fi
''')
    path.chmod(0o700)

def install(w, components, downloads, accept, source_environment):
    steps = setup_steps(w, components, downloads)
    if not steps:
        print("No missing executables detected; existing environments left unchanged.")
        return
    print(f"Research root: {w.root}")
    print("Setup plan:")
    for i, (label, command) in enumerate(steps, 1):
        print(f"{i}. {label}")
        if command:
            print("   " + shlex.join(command))
    if not accept:
        print("Preview only: no installation. Re-run with --yes to apply.")
        return
    if any(command and command[0] == "conda" for _, command in steps) and not shutil.which("conda"):
        raise RuntimeError("Conda not found on PATH; cannot create environments")
    w.prepare()
    env = source_environment(w)
    node_env = w.root / "conda-envs/opencode-node"
    env["PATH"] = str(node_env / "bin") + os.pathsep + env.get("PATH", "")
    env.update({
        "CONDA_PKGS_DIRS": str(w.base / "conda_pkgs"),
        "PIP_CACHE_DIR": str(w.base / "pip_cache"),
        "HF_HOME": str(w.base / "hf_cache"),
        "HF_HUB_CACHE": str(w.base / "hf_cache/hub"),
        "XDG_CONFIG_HOME": str(w.base / "config"),
        "XDG_DATA_HOME": str(w.base / "share"),
        "HF_HUB_OFFLINE": "0",
    })
    for key in ("CONDA_PKGS_DIRS", "PIP_CACHE_DIR", "HF_HOME", "HF_HUB_CACHE", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    logfile = w.log_dir / "setup.log"
    for label, command in steps:
        print(f"SETUP: {label}", flush=True)
        if command is None:
            write_aider_wrapper(w)
            continue
        with logfile.open("ab") as output:
            output.write(("\n--- " + label + " ---\n").encode())
            output.flush()
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, env=env, check=False)
        if result.returncode:
            with logfile.open("rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 16384))
                tail = f.read().decode(errors="replace").splitlines()[-15:]
            raise RuntimeError(f"{label} failed with exit code {result.returncode}. Log {logfile}\n" + "\n".join(tail))
    print(f"Setup complete. Detailed log: {logfile}")
    print("Next: pac-llm doctor")
