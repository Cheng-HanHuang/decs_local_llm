#!/usr/bin/env python3
"""Offline, no-GPU validation of all artifacts installed by lab_autopilot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

CODER_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SFT_MODEL = "Qwen/Qwen3-0.6B"


def must_file(path: Path):
    if not path.is_file():
        raise RuntimeError(f"Missing required file: {path}")
    print("OK file:", path)


def check_lora(root: Path, repo: Path):
    python = root / "conda-envs/lora-demo/bin/python"
    must_file(python)
    script = r"""
import importlib.util
import json
import os
from pathlib import Path
import torch, transformers, peft, accelerate
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

assert torch.version.cuda is not None, "LoRA torch must have CUDA support"
print("Libraries:", torch.__version__, transformers.__version__, peft.__version__, accelerate.__version__)
for model in ("Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-0.6B"):
    p = Path(snapshot_download(model, local_files_only=True))
    assert (p / "config.json").is_file(), model
    print("Cached model:", model, p.name)

source = Path(os.environ["LAB_LORA_SCRIPT"])
run = Path(os.environ["LAB_LORA_RUN"])
manifest = json.loads((run / "data_manifest.json").read_text())
assert manifest["model_id"] == "Qwen/Qwen3-0.6B"
assert manifest["train"] == 96 and manifest["test"] == 24
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
spec = importlib.util.spec_from_file_location("toy_lora", source)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
row = json.loads((run / "train.jsonl").read_text().splitlines()[0])
ids, labels = mod.make_tokens(tok, row)
assert len(ids) == len(labels) and 0 < sum(x != -100 for x in labels) < len(ids)
print("LoRA tokenizer: PASS; tokens", len(ids), "supervised", sum(x != -100 for x in labels))
"""
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    env["LAB_LORA_SCRIPT"] = str(repo / "experiments/lora_demo/demo.py")
    env["LAB_LORA_RUN"] = str(root / "local_llm/lora-demo/runs/workshop-ack-v1")
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        env.pop(k, None)
    p = subprocess.run([str(python), "-c", script], env=env, check=False)
    if p.returncode:
        raise RuntimeError(f"LoRA libraries, cached models, or tokenizer smoke failed ({p.returncode})")


def verify(root: Path, repo: Path):
    root, repo = root.resolve(), repo.resolve()
    print("Checking per-user installation at:", root)
    must_file(repo / "bin/pac-llm")
    must_file(root / "conda-envs/local-agent/bin/vllm")
    must_file(root / "conda-envs/local-agent/bin/ninja")
    must_file(root / "conda-envs/local-agent/bin/hf")
    must_file(root / "conda-envs/open-webui/bin/open-webui")
    if not ((root / "conda-envs/aider/bin/aider").is_file()
            or (root / "local_llm/bin/aider").is_file()):
        raise RuntimeError("Aider executable missing in both supported locations")
    must_file(root / "conda-envs/opencode-node/bin/node")
    must_file(root / "conda-envs/opencode-node/bin/npm")
    must_file(root / "local_llm/opencode/npm/bin/opencode")
    must_file(root / "local_llm/bin/pac-llm")
    check_lora(root, repo)
    print("OFFLINE SETUP VERIFICATION: PASS")
    print("Next: WORKSHOP.md for LIVE GPU, WebUI, Aider and OpenCode exercises.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    try:
        verify(args.root, args.repo)
    except (RuntimeError, OSError, ValueError) as e:
        print(f"OFFLINE SETUP VERIFICATION FAILED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
