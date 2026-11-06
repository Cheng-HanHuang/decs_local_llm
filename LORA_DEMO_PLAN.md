# Planned demonstration: small-model LoRA SFT and vLLM inference

**Status:** design note only. No training has been run, and no environment or
GPU has been modified for this demo. Complete and verify the OpenCode workflow
first.

## Goal

Show, on a single shared PAC GPU, the difference among:

1. Untuned base-model inference.
2. A frozen base model plus a trained LoRA adapter.
3. (Optional) Merged weights, for verifying that the adapter's update can be
   folded into the base parameters for inference.

Use **Qwen/Qwen3-0.6B** (small, Apache-2.0 licensed) so the demonstration is
cheap. The 30B coder is for the agent evaluation, **not** the SFT example.

## Mathematical picture

For a trainable matrix `W` in a transformer, LoRA freezes the pretrained
`W_0` and learns a low-rank change

```math
W_{\text{effective}} = W_0 + \frac{\alpha}{r} B A,
\qquad A\in\mathbb{R}^{r\times d_{in}},
\quad B\in\mathbb{R}^{d_{out}\times r}.
```

For supervised input/answer pairs `(x_i,y_i)`, minimize answer-token
cross entropy

```math
\min_{A,B}\;-\sum_i\sum_t
\log p_{W_0+\frac{\alpha}{r}BA}(y_{it}\mid x_i,y_{i,<t}).
```

Only adapter weights are updated. Training is performed with PyTorch /
Transformers + PEFT (and optionally TRL); **vLLM is used for inference
and benchmarking, not to train model weights**.

## Minimal experiment design (to implement after agent smoke)

- Pick a tiny synthetic, non-sensitive instruction task (e.g., map an integer
  input to a one-line, labeled modulo-three answer using a specified format).
- Make ~64 training and 20 held-out examples with different integers and
  phrasings; lock a seed and save the exact split.
- Compare the same held-out prompts with temperature 0 for the base and LoRA.
- SFT: LoRA rank `r=8`, scale `alpha=16`, `q_proj` and `v_proj`, BF16,
  sequences <=256 tokens, small batches, capped steps, and record loss/VRAM.
- Save only adapter weights, tokenizer/base revision metadata and evaluation
  receipts. Do not store model weights or user data in Git.
- Evaluate both *exact-format accuracy* and *correct task value*. A style
  improvement without value correctness is not a task success.
- As a teaching demonstration this can show a parameter change affects
  responses, but a tiny dataset is not evidence of broad generalization.

## GPU policy

Run the demo on **GPU 2**, after checking contemporaneous free VRAM.
Stop the large coder model first to avoid needless co-tenancy:

```bash
pac-llm stop coder
pac-llm doctor
```

Create a separate environment such as
`$PAC_LLM_ROOT/conda-envs/lora-demo` and keep model checkpoints under
`$PAC_LLM_BASE/lora-demo`. Package versions and CUDA compatibility should be
pinned and smoke-tested before doing the full demonstration.

A 0.6B BF16 model's raw parameter weights are about 1.2 GB decimal, but
training requires additional activation, optimizer, gradient, and allocator
memory. Do not infer total training VRAM from just the weight size.

## Serve and compare adapters

After training, start vLLM using the same base model with LoRA enabled:

```bash
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 18002 \
  --served-model-name qwen3-small-base \
  --dtype bfloat16 --max-model-len 2048 \
  --gpu-memory-utilization 0.30 \
  --enable-lora --max-lora-rank 8 \
  --lora-modules toy-lora=/absolute/path/to/adapter
```

The command above is **illustrative**, not yet PAC-tested. It should be
started from the vLLM environment with Ninja available in PATH and with
the same cache configuration as the working launcher. The server will expose
both `qwen3-small-base` and `toy-lora` through the `model` field of the
OpenAI-compatible API. Evaluate both on the *same* saved prompts.

Do not enable runtime LoRA loading for the first demonstration:
`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` introduces security risks and is
unnecessary for a single local adapter. Static `--lora-modules` loading is
simpler and auditable.

## Next engineering work

Add a CPU-only dataset-generation and evaluation smoke test, a separate
training script and lockfile, and a two-pass base-vs-adapter comparison
receipt. This should be a distinct PR after agent smoke testing.
