#!/usr/bin/env python3
"""Write a non-secret, local PAC reproducibility receipt in JSON.

Runs no installers, starts no services, and never logs environment variables
or credentials. Output must be an explicit per-user path outside Git.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys


PACKAGES = {
    "vllm": ["vllm", "torch", "flashinfer-python", "ninja"],
    "webui": ["open-webui", "torch"],
    "aider": ["aider-chat"],
    "lora": ["torch", "transformers", "peft", "accelerate", "huggingface-hub"],
}


def cmd(args, timeout=15):
    try:
        p = subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def packages_for(env, names):
    exe = env / "bin/python"
    if not exe.is_file():
        return {"installed": False}
    code = (
        "import json,importlib.metadata as m;"
        "print(json.dumps({x:(m.version(x) if any(d.metadata['Name'].lower()==x.lower() "
        "for d in m.distributions()) else None) for x in " + repr(names) + "}))"
    )
    # Keep this isolated from the caller's Python environment. No imports of
    # large libraries and no mutation of installed packages.
    out = cmd([str(exe), "-c", code])
    try:
        return {"installed": True, "packages": json.loads(out)} if out else {"installed": True, "packages": None}
    except json.JSONDecodeError:
        return {"installed": True, "packages": None}


def receipt(root, repo):
    root = Path(root).resolve()
    repo = Path(repo).resolve()
    envs = root / "conda-envs"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "kernel": platform.release(),
        "python_for_receipt": sys.version.split()[0],
        "source_commit": cmd(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        "source_status_porcelain": cmd(["git", "-C", str(repo), "status", "--porcelain"]),
        "nvidia_smi": cmd(["nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version",
            "--format=csv,noheader,nounits"]),
        "environments": {
            "vllm": packages_for(envs / "local-agent", PACKAGES["vllm"]),
            "webui": packages_for(envs / "open-webui", PACKAGES["webui"]),
            "aider": packages_for(envs / "aider", PACKAGES["aider"]),
            "lora": packages_for(envs / "lora-demo", PACKAGES["lora"]),
        },
        "note": "No credentials or chat histories captured; GPU memory is a snapshot.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    data = receipt(args.root, args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite previous receipts.
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    args.out.chmod(0o600)
    print(f"Receipt: {args.out}")
    print(f"Source commit: {data['source_commit']}")


if __name__ == "__main__":
    main()
