# GPU 2: tiny Qwen3 LoRA SFT and vLLM comparison

**Status: runnable experiment candidate, not PAC-tested.**
Nothing here automatically starts a GPU job. Run commands manually.
This demonstration changes *only LoRA adapter weights* in a dedicated
training environment; it does not overwrite the Qwen3 base model or the
currently working vLLM/Aider environment.

## What we are learning

LoRA keeps each pretrained matrix `W_0` frozen and trains two matrices:

`W_eff = W_0 + (alpha/r) B A`.

With rank `r=8`, `alpha=16`, and `q_proj, v_proj` targets, the adapter is
small. We will (a) SFT a tiny model via PyTorch/PEFT, then (b) load the
base and trained adapter simultaneously in vLLM to observe different outputs.

**Toy task:** Given a telemetry ticket number, respond exactly `PAC-ACK`.
The ticket numbers in the 96 training prompts and 24 test prompts are
different; the reply is deliberately constant. This probes whether an
adapter can learn a simple behavior, **not** mathematical reasoning,
independent language capability, or a meaningful generalization theorem.
Changing this to nontrivial arithmetic is a *separate* experiment.

Model: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
(Transformers >=4.51). We use Qwen3's *non-thinking* chat template.

## 0. Check out the experiment without disturbing the agent worktree

```bash
ROOT=/egr/research-pac/$USER
BASE="$ROOT/local_llm"
REPO="$BASE/lora-demo/source"
mkdir -p "$BASE/lora-demo"
git -C "$ROOT/decs_local_llm" fetch origin experiment/qwen3-small-lora-demo
git -C "$ROOT/decs_local_llm" worktree add --detach \
  "$REPO" origin/experiment/qwen3-small-lora-demo
```

If the worktree already exists, inspect it rather than running `git worktree
add` again. This is a distinct checkout: do not reuse or modify your
OpenCode edit-test worktree.

## 1. First release GPU 2, if needed

If your big Qwen coder is still running, exit the agent session, then:

```bash
pac-llm status
pac-llm stop coder     # ONLY if it is running and you want to release GPU memory
pac-llm doctor
```

GPU 2 is shared, so **recheck current free memory**.
There is no guarantee that its prior ~24 GiB usage remains unchanged.
Do not run the LoRA demo on the same GPU as the 30B coder.

## 2. One-time dedicated training environment (PAC)

```bash
ROOT=/egr/research-pac/$USER
BASE="$ROOT/local_llm"
ENV="$ROOT/conda-envs/lora-demo"
mkdir -p "$BASE/logs" "$BASE/lora-demo" "$BASE/pip-cache"
source "$BASE/env_local_agent.sh"
export PIP_CACHE_DIR="$BASE/pip-cache"

# Do NOT alter conda-envs/local-agent (vLLM), or Open WebUI's environment.
conda create -p "$ENV" python=3.11 pip -y \
  > "$BASE/logs/lora-conda-install.log" 2>&1

"$ENV/bin/python" -m pip install "torch==2.8.0" \
  --index-url https://download.pytorch.org/whl/cu128 \
  > "$BASE/logs/lora-torch-install.log" 2>&1

"$ENV/bin/python" -m pip install \
  "transformers==4.57.1" "peft==0.17.1" \
  "accelerate==1.11.0" "huggingface_hub==0.36.0" \
  > "$BASE/logs/lora-hf-install.log" 2>&1

"$ENV/bin/python" -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)"
```

Versions are an initial *candidate*, not a tested PAC lockfile. If an
installation command fails, inspect the final 20 lines of its log and
**stop**; do not continue on a half-configured environment. PyTorch
2.8.0 CUDA 12.8 wheels are listed in
[PyTorch official previous versions](https://pytorch.org/get-started/previous-versions/).
Qwen3 requires sufficiently recent Transformers.

## 3. Download the small base model once

```bash
ROOT=/egr/research-pac/$USER
BASE="$ROOT/local_llm"
ENV="$ROOT/conda-envs/lora-demo"
source "$BASE/env_local_agent.sh"
export HF_HOME="$BASE/hf_cache"
export HF_HUB_CACHE="$BASE/hf_cache/hub"
export HF_HUB_OFFLINE=0
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN

"$ENV/bin/hf" download Qwen/Qwen3-0.6B
```

Use a personal model cache. Do not commit weights or Hugging Face credentials
to Git. This model is public; no token should be required.

## 4. Prepare data (CPU) and train (GPU 2)

Use a **fresh** run directory. The data and training receipt stay outside Git.

```bash
ROOT=/egr/research-pac/$USER
BASE="$ROOT/local_llm"
ENV="$ROOT/conda-envs/lora-demo"
REPO="$BASE/lora-demo/source"
RUN="$BASE/lora-demo/runs/ack-v1"
export HF_HOME="$BASE/hf_cache"
export HF_HUB_CACHE="$BASE/hf_cache/hub"
export HF_HUB_OFFLINE=1

"$ENV/bin/python" "$REPO/experiments/lora_demo/demo.py" \
  prepare --run-dir "$RUN"

# Verify CUDA availability on the requested physical GPU.
CUDA_VISIBLE_DEVICES=2 "$ENV/bin/python" -c \
  'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'

CUDA_VISIBLE_DEVICES=2 "$ENV/bin/python" \
  "$REPO/experiments/lora_demo/demo.py" \
  train --run-dir "$RUN" --steps 30 --lr 0.0005
```

The train process sees physical GPU 2 as `cuda:0` after setting
`CUDA_VISIBLE_DEVICES=2`. Training operates in the foreground and may
allocate several GiB. Do not assume VRAM usage based on the 0.6B weight
size alone. Stop if another user's workload needs GPU 2.

The script records the learning loss, trainable count, training duration,
and allocated VRAM under `$RUN`, and saves the LoRA adapter in
`$RUN/adapter`. No base checkpoint is modified.

## 5. Serve the base and adapter together (separate terminal)

Once training finishes, use the **existing** vLLM environment, not the
new training environment. Keep the repo version and selected adapter path
consistent. We intentionally use port 18002 to avoid the coder's port 8000.

```bash
ROOT=/egr/research-pac/$USER
BASE="$ROOT/local_llm"
RUN="$BASE/lora-demo/runs/ack-v1"
source "$BASE/env_local_agent.sh"
export PATH="$ROOT/conda-envs/local-agent/bin:$PATH"
export CUDA_VISIBLE_DEVICES=2
export HF_HUB_OFFLINE=1

"$ROOT/conda-envs/local-agent/bin/vllm" serve \
  Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 18002 \
  --served-model-name qwen3-small-base \
  --dtype bfloat16 --max-model-len 2048 \
  --gpu-memory-utilization 0.30 --max-num-seqs 2 \
  --enable-lora --max-lora-rank 8 \
  --lora-modules "toy-lora=$RUN/adapter"
```

This vLLM launch is **foreground**. If it fails, use its terminal log to
investigate. Stop it with Ctrl+C. Do not set
`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`: dynamic adapter loading is not
needed here. A static `--lora-modules` mount is simpler and safer.

After vLLM reports startup complete, in a *second* PAC terminal:

```bash
curl -sS http://127.0.0.1:18002/v1/models | python3 -m json.tool

ROOT=/egr/research-pac/$USER
RUN="$ROOT/local_llm/lora-demo/runs/ack-v1"
ENV="$ROOT/conda-envs/lora-demo"

"$ENV/bin/python" "$ROOT/local_llm/lora-demo/source/experiments/lora_demo/demo.py" \
  evaluate --run-dir "$RUN" --port 18002
```

This sends each of the 24 held-out prompts to both model IDs:
`qwen3-small-base` and `toy-lora`. It writes the complete per-example
comparison to `evaluation.jsonl` and summary to
`evaluation_summary.json`, under the run directory.

The evaluation uses non-thinking chat, temperature 0, and a strict
exact-match target. The base may already say the target occasionally, and
LoRA may fail to learn it: **no result is promised before PAC execution**.
If no improvement appears, inspect training loss, formatting and generated
text before changing hyperparameters.

## 6. Inspect receipts and explain weight modification

```bash
"$ENV/bin/python" "$ROOT/decs_local_llm/experiments/lora_demo/demo.py" \
  inspect --run-dir "$RUN"
cat "$RUN/evaluation_summary.json"
```

PEFT learns `A, B` via ordinary backpropagation on teacher-forced
cross-entropy. For the selected layers the effective model is
`W_0 + (alpha/r) BA`, while the original `W_0` is unchanged.
vLLM applies the adapter at inference; it **does not perform SFT**.

A separate optional experiment can call PEFT's `merge_and_unload()` to
form merged base+adapter weights, then compare the merged model against
the unmerged adapter numerically. Do not overwrite pretrained snapshots.

## Scope and limits

- This is a miniature educational demo, not an optimized SFT pipeline.
- Output accuracy is a format/memorization diagnostic, not a benchmark.
- Training is limited to `q_proj` and `v_proj`, rank eight.
- Experiments run only after explicit manual commands, on GPU 2.
- OpenCode agent testing can be performed independently; it is not used
  to train the model or manage GPU jobs in this demo.
