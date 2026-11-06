#!/usr/bin/env python3
"""Resumable, per-user PAC lab installer. NO GPU jobs or services are started.

Run once from an account's own Git checkout:
  python3 scripts/lab_autopilot.py --plan
  nohup python3 scripts/lab_autopilot.py --run > "$ROOT/local_llm/lab_autopilot/controller.log" 2>&1 < /dev/null &
  python3 scripts/lab_autopilot.py --status

Only --run installs packages/downloads public weights. Interrupted stages are
retried on the next --run; finished stages are not repeated for the same
source commit. Actual chat, agent and GPU tests are in WORKSHOP.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import getpass
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import NamedTuple

CODER_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SFT_MODEL = "Qwen/Qwen3-0.6B"
HF_LIBS = (
    "transformers==4.57.1",
    "peft==0.17.1",
    "accelerate==1.11.0",
    "huggingface_hub==0.36.0",
)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def root_for_user():
    user = getpass.getuser()
    expected = [Path(f"/egr/{host}/{user}") for host in ("research-pac", "research-slim")]
    chosen = Path(os.environ.get("PAC_LLM_ROOT", str(next((p for p in expected if p.is_dir()), expected[0]))))
    resolved = chosen.resolve()
    if resolved not in [p.resolve() for p in expected]:
        raise RuntimeError(f"Refusing cross-account or non-research root: {resolved}")
    if not resolved.is_dir():
        raise RuntimeError(f"Personal research root does not exist: {resolved}")
    return resolved


def repo_root():
    p = Path(__file__).resolve().parents[1]
    if not (p / "bin/pac-llm").is_file() or not (p / "experiments/lora_demo/demo.py").is_file():
        raise RuntimeError(f"Not in the expected decs_local_llm checkout: {p}")
    return p


def git_sha(repo):
    p = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                       text=True, capture_output=True, check=False)
    if p.returncode or not p.stdout.strip():
        raise RuntimeError(f"Cannot determine Git commit in {repo}")
    return p.stdout.strip()


class Stage(NamedTuple):
    name: str
    description: str
    command: list[str] | None


def stages(root, repo):
    base = root / "local_llm"
    launcher = str(repo / "bin/pac-llm")
    lora = root / "conda-envs/lora-demo"
    python = str(lora / "bin/python")
    hf = str(root / "conda-envs/local-agent/bin/hf")
    conda = shutil.which("conda") or "conda"
    if (lora / "bin/python").is_file():
        create_lora = None
    else:
        # An interrupted conda create can leave a partially initialized prefix.
        action = "install" if (lora / "conda-meta").is_dir() else "create"
        create_lora = [conda, action, "-p", str(lora), "python=3.11", "pip", "-y"]
    data_run = base / "lora-demo/runs/workshop-ack-v1"
    return [
        Stage("cpu_tests", "Run offline repository unit tests",
              [sys.executable, "-m", "unittest", "discover",
               "-s", str(repo / "tests"), "-v"]),
        Stage("vllm", "Install vLLM + Ninja in the personal inference environment",
              [sys.executable, launcher, "setup", "--components", "vllm", "--yes"]),
        Stage("webui", "Install CPU-only Open WebUI environment",
              [sys.executable, launcher, "setup", "--components", "web", "--yes"]),
        Stage("aider", "Install Aider (ask-only launcher)",
              [sys.executable, launcher, "setup", "--components", "aider", "--yes"]),
        Stage("opencode", "Install Node.js and OpenCode CLI",
              [sys.executable, launcher, "setup", "--components", "opencode", "--yes"]),
        Stage("lora_env", "Create or resume separate Python 3.11 LoRA environment", create_lora),
        Stage("lora_torch", "Install PyTorch 2.8.0 + CUDA 12.8 for LoRA",
              [python, "-m", "pip", "install", "torch==2.8.0",
               "--index-url", "https://download.pytorch.org/whl/cu128"]),
        Stage("lora_libs", "Install pinned Transformers/PEFT/Accelerate/HF Hub",
              [python, "-m", "pip", "install", *HF_LIBS]),
        Stage("coder_weights", "Download public Qwen3-4B (no HF token)",
              [hf, "download", CODER_MODEL]),
        Stage("sft_weights", "Download public Qwen3-0.6B (no HF token)",
              [hf, "download", SFT_MODEL]),
        Stage("toy_data", "Prepare deterministic LoRA training/test examples (CPU)",
              [python, str(repo / "experiments/lora_demo/demo.py"),
               "prepare", "--run-dir", str(data_run)]),
        Stage("verify", "Check installed tools, cached models, and LoRA tokenizer",
              [sys.executable, str(repo / "scripts/lab_verify.py"),
               "--root", str(root), "--repo", str(repo)]),
    ]


def clean_env(root):
    base = root / "local_llm"
    env = os.environ.copy()
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        env.pop(name, None)
    env.update({
        "PAC_LLM_ROOT": str(root),
        "PAC_LLM_BASE": str(base),
        "HF_HOME": str(base / "hf_cache"),
        "HF_HUB_CACHE": str(base / "hf_cache/hub"),
        "HF_HUB_OFFLINE": "0",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "CONDA_PKGS_DIRS": str(base / "conda_pkgs"),
        "PIP_CACHE_DIR": str(base / "pip_cache"),
        "XDG_CACHE_HOME": str(base / "cache"),
        "XDG_CONFIG_HOME": str(base / "config"),
        "XDG_DATA_HOME": str(base / "share"),
        "TMPDIR": str(base / "tmp"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
    })
    # Only this user's executables. FlashInfer requires Ninja on PATH.
    env["PATH"] = os.pathsep.join([
        str(root / "conda-envs/local-agent/bin"),
        str(root / "conda-envs/opencode-node/bin"),
        str(base / "bin"),
        env.get("PATH", ""),
    ])
    return env


def prepare_dirs(root):
    base = root / "local_llm"
    dirs = [
        base / "lab_autopilot", base / "lab_autopilot/stages",
        base / "hf_cache/hub", base / "conda_pkgs", base / "pip_cache",
        base / "cache", base / "config", base / "share", base / "tmp", base / "bin",
    ]
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)
    (base / "lab_autopilot").chmod(0o700)
    (base / "lab_autopilot/stages").chmod(0o700)
    (base / "config").chmod(0o700)


def ensure_launcher_link(root, repo):
    link = root / "local_llm/bin/pac-llm"
    target = repo / "bin/pac-llm"
    if link.is_symlink() or link.exists():
        if link.resolve() != target.resolve():
            raise RuntimeError(
                f"Existing launcher {link} points to {link.resolve()}, not {target}. "
                "Do not overwrite it; use the checkout's bin/pac-llm explicitly."
            )
    else:
        link.symlink_to(target)


def status_path(root):
    return root / "local_llm/lab_autopilot/status.json"


def read_status(root):
    path = status_path(root)
    if not path.is_file():
        return {}
    try:
        result = json.loads(path.read_text())
        return result if isinstance(result, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_status(root, state):
    file = status_path(root)
    temp = file.with_name("status.tmp")
    with temp.open("w", encoding="utf-8") as out:
        json.dump(state, out, indent=2, sort_keys=True)
        out.write("\n")
    temp.chmod(0o600)
    os.replace(temp, file)


def show_status(root):
    state = read_status(root)
    if not state:
        print("Not run yet. Use --plan, then --run.")
        return
    print("Status:", state.get("status", "unknown"))
    print("Source commit:", state.get("source_sha", "?"))
    print("Current stage:", state.get("stage") or "-")
    print("Completed:", ", ".join(state.get("completed", [])) or "-")
    print("Last error:", state.get("error") or "-")
    print("Stage logs:", root / "local_llm/lab_autopilot/stages")
    print("Updated:", state.get("updated_utc", "?"))


def run_install(root, repo):
    prepare_dirs(root)
    where = root / "local_llm/lab_autopilot"
    lock = where / ".lock"
    with lock.open("a+") as fd:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another installer is already running; use --status") from exc
        sha = git_sha(repo)
        previous = read_status(root)
        if previous.get("source_sha") != sha:
            previous = {"source_sha": sha, "completed": []}
        completed = set(previous.get("completed", []))
        state = previous
        state.update({"status": "running", "stage": None, "error": None,
                      "updated_utc": utcnow(), "pid": os.getpid()})
        save_status(root, state)
        if not shutil.which("conda"):
            state.update(status="failed", error="Conda not found on PATH; activate PAC's supported Conda first")
            save_status(root, state)
            raise RuntimeError(state["error"])
        try:
            ensure_launcher_link(root, repo)
            free = shutil.disk_usage(root).free
            print(f"Personal root: {root} | source: {sha} | filesystem free: {free/2**30:.1f} GiB",
                  flush=True)
            print("NOTE: free space is NOT the same as your PAC quota.", flush=True)
            env = clean_env(root)
            for item in stages(root, repo):
                if item.name in completed:
                    print(f"SKIP {item.name}: completed for this Git revision", flush=True)
                    continue
                state.update({"stage": item.name, "status": "running",
                              "updated_utc": utcnow(), "error": None})
                save_status(root, state)
                log = where / "stages" / (item.name + ".log")
                print(f"START {item.name}: {item.description} | log: {log}", flush=True)
                with log.open("ab") as stream:
                    stream.write(("\n=== " + utcnow() + " " + item.description + " ===\n").encode())
                    stream.flush()
                    if item.command is None:
                        stream.write(b"Already installed; no action required.\n")
                        code = 0
                    else:
                        # Never use shell=True, sudo, or external endpoints.
                        p = subprocess.run(item.command, cwd=str(repo), env=env,
                                           stdout=stream, stderr=subprocess.STDOUT,
                                           check=False)
                        code = p.returncode
                if code != 0:
                    tail = log.read_text(errors="replace").splitlines()[-8:]
                    state.update(status="failed",
                                 error=f"{item.name} exited {code}. See {log}",
                                 updated_utc=utcnow())
                    save_status(root, state)
                    print("FAILED:", state["error"], file=sys.stderr, flush=True)
                    print("\n".join(tail), file=sys.stderr, flush=True)
                    return 1
                completed.add(item.name)
                state.update({"completed": sorted(completed),
                              "stage": None, "updated_utc": utcnow()})
                save_status(root, state)
                print(f"PASS {item.name}", flush=True)
            state.update({"status": "complete", "stage": None,
                          "error": None, "updated_utc": utcnow()})
            save_status(root, state)
            print("SETUP COMPLETE (CPU and packages). No GPU service or training was started.",
                  flush=True)
            print("Next: WORKSHOP.md, configure unique ports/GPU and run live exercises.",
                  flush=True)
            return 0
        except (OSError, RuntimeError) as exc:
            state.update({"status": "failed", "error": str(exc), "updated_utc": utcnow()})
            save_status(root, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="Preview stages, no writes/downloads")
    mode.add_argument("--run", action="store_true", help="Run/resume installer (opt-in)")
    mode.add_argument("--status", action="store_true", help="Read status only")
    args = parser.parse_args()
    try:
        root, repo = root_for_user(), repo_root()
        if args.plan:
            print(f"Per-user root: {root}\nCheckout: {repo}\nNo GPUs used.")
            for i, s in enumerate(stages(root, repo), 1):
                print(f"{i:2d}. {s.name}: {s.description}")
                if s.command:
                    print("    " + shlex.join(s.command))
            return 0
        if args.status:
            show_status(root)
            return 0
        return run_install(root, repo)
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
