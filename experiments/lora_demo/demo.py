#!/usr/bin/env python3
"""Tiny Qwen3-0.6B LoRA SFT teaching demo. No data or weights checked into Git.

Commands: prepare (CPU), train (GPU), inspect (CPU), evaluate (vLLM API).
All outputs live under --run-dir, not inside the source repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import urllib.request

MODEL_ID = "Qwen/Qwen3-0.6B"
LABEL = "PAC-ACK"
TRAIN_COUNT = 96
TEST_COUNT = 24


def records(seed=17):
    rng = random.Random(seed)
    numbers = rng.sample(range(10000, 90000), TRAIN_COUNT + TEST_COUNT)
    def item(n):
        return {
            "id": n,
            "prompt": (
                f"DECS internal telemetry ticket {n}. "
                "What is the acknowledgement code for this ticket? "
                "Reply with the code only."
            ),
            "target": LABEL,
        }
    return [item(n) for n in numbers[:TRAIN_COUNT]], [item(n) for n in numbers[TRAIN_COUNT:]]


def jsonl_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))


def jsonl_read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def prepare(run_dir, seed=17):
    run_dir.mkdir(parents=True, exist_ok=True)
    train, test = records(seed)
    # Make reruns reproducible; never silently replace an existing dataset.
    for name, rows in (("train", train), ("test", test)):
        path = run_dir / f"{name}.jsonl"
        content = "".join(json.dumps(x, sort_keys=True) + "\n" for x in rows)
        if path.exists() and path.read_text() != content:
            raise RuntimeError(f"{path} exists with different data; select a new --run-dir")
        jsonl_write(path, rows)
    manifest = {
        "model_id": MODEL_ID, "seed": seed, "train": len(train), "test": len(test),
        "target": LABEL,
        "sha256": {p: hashlib.sha256((run_dir / f"{p}.jsonl").read_bytes()).hexdigest()
                   for p in ("train", "test")},
        "purpose": "Demonstration of formatting/memorization; not a generalization benchmark",
    }
    (run_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def make_tokens(tok, row, max_len=192):
    user = [{"role": "user", "content": row["prompt"]}]
    # Qwen3 defaults to thinking. Explicitly disable it so this toy task
    # targets short non-thinking assistant responses.
    prompt = tok.apply_chat_template(
        user, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tok(prompt + row["target"] + tok.eos_token,
                   add_special_tokens=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError("Tokenizer prefix mismatch; inspect Qwen3 chat template")
    if len(full_ids) > max_len:
        raise RuntimeError("Example exceeds max_len; do not silently truncate target")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    assert any(x != -100 for x in labels)
    return full_ids, labels


def collate(batch, pad_id):
    import torch
    length = max(len(x[0]) for x in batch)
    ids = [a + [pad_id] * (length - len(a)) for a, _ in batch]
    labs = [b + [-100] * (length - len(b)) for _, b in batch]
    masks = [[1] * len(a) + [0] * (length - len(a)) for a, _ in batch]
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labs, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


def train(run_dir, steps, seed, lr, force=False):
    import torch
    from importlib.metadata import version
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; training must run on GPU")
    if not (run_dir / "data_manifest.json").is_file():
        raise RuntimeError("Missing dataset; run prepare first")
    adapter = run_dir / "adapter"
    if adapter.exists() and any(adapter.iterdir()) and not force:
        raise RuntimeError("Existing adapter; select a fresh --run-dir (or --force)")
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if not tok.eos_token:
        raise RuntimeError("Tokenizer has no eos token")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=False).to("cuda")
    base_model_revision = getattr(model.config, "_commit_hash", None)
    tokenizer_revision = getattr(tok, "init_kwargs", {}).get("_commit_hash")
    model.config.use_cache = False
    config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, config)
    model.train()
    model.print_trainable_parameters()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    data = jsonl_read(run_dir / "train.jsonl")
    enc = [make_tokens(tok, x) for x in data]
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, weight_decay=0.0)
    t0 = time.time()
    losses = []
    optimizer.zero_grad(set_to_none=True)
    accum = 4
    batch_size = 2
    for step in range(steps):
        # Deterministic sequence of shuffled epochs without a DataLoader.
        batch_losses = []
        for micro in range(accum):
            index = (step * accum + micro) * batch_size
            rng = random.Random(seed + index // len(enc))
            order = list(range(len(enc)))
            rng.shuffle(order)
            items = [enc[order[(index + j) % len(enc)]] for j in range(batch_size)]
            batch = {k: v.cuda() for k, v in collate(items, tok.eos_token_id).items()}
            loss = model(**batch).loss
            (loss / accum).backward()
            batch_losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_value = sum(batch_losses) / len(batch_losses)
        losses.append(loss_value)
        if (step + 1) % 5 == 0 or step == 0:
            print(f"step={step+1}/{steps} loss={loss_value:.4f}", flush=True)

    adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter, safe_serialization=True)
    tok.save_pretrained(adapter)
    # PEFT initializes lora_B to zero. A nonzero post-training B norm
    # demonstrates the adapter departed from its initial zero update.
    lora_norm_sq = 0.0
    lora_b_norm_sq = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad:
            sq = float(p.detach().float().square().sum().cpu())
            lora_norm_sq += sq
            if "lora_B" in name:
                lora_b_norm_sq += sq
    receipt = {
        "model_id": MODEL_ID, "model_revision": base_model_revision,
        "tokenizer_revision": tokenizer_revision,
        "seed": seed, "steps": steps, "lr": lr,
        "rank": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"],
        "trainable_parameters": n_trainable, "total_parameters": n_total,
        "first_loss": losses[0], "last_loss": losses[-1],
        "elapsed_seconds": time.time() - t0,
        "trainable_parameter_l2_norm": lora_norm_sq ** 0.5,
        "lora_B_l2_norm": lora_b_norm_sq ** 0.5,
        "max_gpu_memory_allocated_gib":
            torch.cuda.max_memory_allocated() / 1024 ** 3,
        "torch_version": torch.__version__,
        "transformers_version": version("transformers"),
        "peft_version": version("peft"),
        "cuda_runtime_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
    }
    (run_dir / "training_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def inspect(run_dir):
    manifest = run_dir / "data_manifest.json"
    adapter = run_dir / "adapter"
    if not manifest.exists():
        raise RuntimeError("Missing data_manifest.json")
    print(manifest.read_text())
    for name in ("training_receipt.json", "adapter/adapter_config.json"):
        file = run_dir / name
        print(f"{name}: {'present' if file.exists() else 'missing'}")
        if file.exists() and name.startswith("training_"):
            print(file.read_text())


def chat(port, model, question):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": 0.0, "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=90) as resp:
        obj = json.load(resp)
    return obj["choices"][0]["message"].get("content") or ""


def evaluate(run_dir, port, base_model, adapter_model):
    if not (run_dir / "test.jsonl").exists():
        raise RuntimeError("Missing test data; run prepare first")
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=8) as resp:
        available = {x["id"] for x in json.load(resp)["data"]}
    required = {base_model, adapter_model}
    if not required <= available:
        raise RuntimeError(f"Server models={sorted(available)}; missing={required - available}")
    rows = jsonl_read(run_dir / "test.jsonl")
    out = []
    for idx, item in enumerate(rows):
        item_result = dict(item)
        for label, name in (("base", base_model), ("lora", adapter_model)):
            result = chat(port, name, item["prompt"]).strip()
            item_result[label] = result
            item_result[label + "_exact"] = result == item["target"]
        out.append(item_result)
        print(f"{idx+1}/{len(rows)} base={item_result['base_exact']} "
              f"lora={item_result['lora_exact']}", flush=True)
    jsonl_write(run_dir / "evaluation.jsonl", out)
    report = {
        "n": len(out), "base_model": base_model, "adapter_model": adapter_model,
        "base_exact": sum(x["base_exact"] for x in out),
        "lora_exact": sum(x["lora_exact"] for x in out),
        "target": LABEL,
        "note": "Toy unseen-number heldout; not evidence of broad model generalization",
    }
    (run_dir / "evaluation_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "inspect", "evaluate"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--port", type=int, default=18002)
    parser.add_argument("--base-model", default="qwen3-small-base")
    parser.add_argument("--adapter-model", default="toy-lora")
    args = parser.parse_args()
    if not 1 <= args.steps <= 200:
        parser.error("--steps must be 1..200")
    if not 0 < args.lr < 1:
        parser.error("--lr must be >0 and <1")
    if args.command == "prepare":
        prepare(args.run_dir, args.seed)
    elif args.command == "train":
        train(args.run_dir, args.steps, args.seed, args.lr, args.force)
    elif args.command == "inspect":
        inspect(args.run_dir)
    else:
        evaluate(args.run_dir, args.port, args.base_model, args.adapter_model)


if __name__ == "__main__":
    main()
