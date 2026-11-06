# PAC local-LLM lab handout

**Scope:** Each lab member learns to install, serve, chat with, and call tools
from a small local model; then try read-only Aider, guarded OpenCode,
and an optional tiny LoRA SFT. These are *per-user* installations on a
shared PAC host. None of the scripts requires administrator privileges.

**Status (November 6, 2026):** Source-level CPU tests and the original
30B Qwen3-Coder agent workflow have been exercised on PAC. The separate
Qwen3-0.6B LoRA experiment has been trained and evaluated successfully.
**The new 4B workshop default, fresh-user installer and multi-user
co-tenancy are proposed but NOT YET PAC-validated.** Run the gates below
before a group workshop.

## Learning goals

1. Separate **model weights** from **inference engine** (vLLM), **UI**
   (Open WebUI) and **coding agents** (Aider/OpenCode).
2. Distinguish **process management**, **tool parser**, **model context**,
   **GPU reservation**, and **API model name**.
3. Make a small, reviewable code change in a **disposable Git worktree**.
4. Train only LoRA weights with PyTorch/PEFT, then compare base and adapter
   through the same vLLM endpoint.
5. Save commands, source revision, environment versions, training receipts
   and evaluation outputs; distinguish validated behavior from proposals.

## 0. What is shared versus private?

| Item | Location / rule |
| --- | --- |
| Launcher and experiment source | A Git checkout, pinned to a commit |
| Personal root | `/egr/research-pac/$USER` |
| API/service settings | `$ROOT/local_llm/config/pac-llm.json` |
| Model cache / logs / UI data | `$ROOT/local_llm/` (each account separately) |
| vLLM, Open WebUI, LoRA, Node envs | `$ROOT/conda-envs/` (do not share live envs) |
| LoRA data, adapter and receipts | `$ROOT/local_llm/lora-demo/runs/<run-name>` |
| OpenCode editing | `$ROOT/local_llm/opencode/worktrees/<name>` only |
| GPU | Explicitly assigned per person; do not assume it is free |
| TCP ports | Unique per student **on the PAC host**, even on localhost |

Loopback binding prevents direct remote internet access, **not** access
from other accounts on the same shared host. These demo APIs do **not**
have strong authentication; use only synthetic, nonsensitive prompts and
coordinate with administrators before sharing endpoints. Follow PAC
scheduling/resource policies. Avoid placing credentials in tool prompts.
Do not publish model weights, chat histories or tokens to Git.

## 1. Fresh lab account: clone this branch

The repository is private. Ask for read access first.

```bash
export PAC_LLM_ROOT=/egr/research-pac/$USER
ROOT="$PAC_LLM_ROOT"
mkdir -p "$ROOT/local_llm/bin"

git clone --branch feature/pac-lab-reproducibility \
  git@github.com:Cheng-HanHuang/decs_local_llm.git \
  "$ROOT/decs_local_llm"

test -e "$ROOT/local_llm/bin/pac-llm" || \
  ln -s "$ROOT/decs_local_llm/bin/pac-llm" "$ROOT/local_llm/bin/pac-llm"
export PATH="$ROOT/local_llm/bin:$PATH"

git -C "$ROOT/decs_local_llm" rev-parse HEAD
pac-llm doctor
python3 -m unittest discover -s "$ROOT/decs_local_llm/tests" -v
```

When this feature stack merges, clone the release/default branch rather
than this development branch. In an **existing checkout**, do not clone
over it or overwrite local changes; use `git status`, `git fetch`, and
a separate Git worktree. An existing `pac-llm` symlink should be checked
with `readlink -f "$(command -v pac-llm)"` before any changes.

Conda must be available as `conda` on the user's PATH. If not, use
the PAC-supported module/installation instructions; don't install Conda
into a shared system location.

## 2. Assign unique ports and a GPU

The instructor assigns each student a free GPU and **three distinct**
ports. The values below are placeholders, *not a block to run unchanged*:

```bash
GPU=2
CODER_PORT=18501
REASONER_PORT=18502
WEB_PORT=19501

pac-llm config init --gpu "$GPU" \
  --coder-port "$CODER_PORT" \
  --reasoner-port "$REASONER_PORT" \
  --web-port "$WEB_PORT"
pac-llm config show
```

If personal config already exists, use `config show` and individual
`pac-llm config set gpu <number>` / `config set coder_port <port>`
instead of running `config init` twice. A stopped service does not
reserve a TCP port. Review `pac-llm doctor` and `nvidia-smi` again
before each launch, especially on shared GPU 2.

**Workshop model (proposed default):**
`Qwen/Qwen3-4B-Instruct-2507`, BF16 (roughly 8 GB just for weights),
16,384-token context, `hermes` tool parser, and vLLM
`--gpu-memory-utilization 0.16` (about 15.3 GiB of a 96 GiB GPU).
This is **a memory budget target**, not measured peak VRAM or guaranteed
co-tenancy. The model is Apache-2.0 licensed. It is an instruct model
with coding ability, not the specialized 30B Coder.

## 3. Install without changing anyone else's environment

First inspect what would happen:

```bash
pac-llm setup --components vllm,web,aider,opencode
```

Only when the plan is acceptable:

```bash
pac-llm setup --components vllm,web,aider,opencode --yes
pac-llm doctor
```

Installer outputs go to `$ROOT/local_llm/logs/<hostname>/setup.log`,
not necessarily to the terminal. The installer skips executable
components already installed; it does **not** implement a full
dependency lock/reinstall. Initial candidates: vLLM 0.23.0; Open
WebUI 0.13.0; Aider 0.86.2; OpenCode 2.2.5; Node 22. The WebUI CPU
PyTorch pin was corrected to 2.8.0, but full fresh-user dependency
compatibility is **not yet validated**.

Download only the small workshop model:

```bash
pac-llm setup --components vllm --download-coder --yes
```

It uses the per-user Hugging Face cache. Downloading weights is an
explicit opt-in and may require substantial disk space. No login token
is required for this public model. Never copy another student's
HF credentials.

## 4. Start vLLM and test chat/tool calling

```bash
pac-llm start coder    # uses saved GPU and ports
pac-llm status
pac-llm logs coder --lines 30
```

A start command **launches a managed background process** and returns
before model initialization is complete. Only proceed after
`pac-llm status` says `coder ... api=ready`.

```bash
ROOT=/egr/research-pac/$USER
CODER_PORT=18501  # replace with YOUR saved coder_port

python3 "$ROOT/decs_local_llm/scripts/pac_workshop_smoke.py" \
  --port "$CODER_PORT" --model local-qwen-coder --test-tools
```

The smoke test asks for a short text response and asks the model to
**emit** a structured `add_two_integers(a=2,b=3)` tool call. It does
not execute any tool. A model can answer text while failing structured
tool calling; **do not count the workshop agent stack as passed**
unless this tool call succeeds. Repeated trials may be useful because
model tool use is not completely deterministic.

If the 4B model fails this gate, retain it only for plain chat and
do not claim Aider/OpenCode tool calls work. Reassess the model/parser
choice or use the separately validated 30B setup.

## 5. Optional browser UI

```bash
pac-llm start web
pac-llm status
```

From a laptop (replace ports, remote account and host):

```powershell
ssh -N -L 19080:127.0.0.1:19501 your-user@pac
```

Open `http://127.0.0.1:19080` in the browser. The model selector in
the Chat interface uses the vLLM connection; an empty **Workspace →
Models** list is not proof that the API is disconnected. Keep UI/chat
data per account. Open WebUI claims about having performed actions must
be verified.

## 6. Aider and OpenCode: source reading, then a guarded edit

Aider is installed separately but talks to the **same running vLLM**
endpoint. From the repo:

```bash
cd "$ROOT/decs_local_llm"
pac-llm aider
```

Inside Aider:

```text
/read-only bin/pac-llm bin/pac_llm_bootstrap.py
```

Then ask for one function's role. Aider defaults to ask-only and does
not update `.gitignore` through our wrapper.

OpenCode read-only:

```bash
pac-llm opencode --preview
pac-llm opencode --steps 8
```

Suggested prompt:

> Open the single file `bin/pac-llm`. Find the Python functions
> `cmd_start`, `spawn`, and `owns_process` *inside that file*.
> Explain their relationship in three bullet points; do not edit.

For edits, **never use the primary checkout**:

```bash
ROOT=/egr/research-pac/$USER
BASE="$ROOT/local_llm"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WORK="$BASE/opencode/worktrees/lesson-$STAMP"
mkdir -p "$BASE/opencode/worktrees"
git -C "$ROOT/decs_local_llm" worktree add \
  -b "lesson/$USER-$STAMP" "$WORK" HEAD
cd "$WORK"
pac-llm opencode --mode edit --steps 16
```

Suggested new task: add one *new* unit test showing a negative single
GPU index is rejected. Read `tests/test_setup_config.py` directly;
modify only that file. **Ask before edits and do not execute shell
commands.** Then exit, inspect `git diff --check; git diff`, and run
`python3 -m unittest discover -s tests -v` manually.

Do **not** choose “always allow” for broad file writes or shell actions.
OpenCode's permission settings are not an OS sandbox. Qwen3-Coder-30B
previously succeeded after human review but also made an unrelated
test change on its first try; read the entire diff. We have not yet
validated this exercise with the new 4B model. See [OPENCODE.md](OPENCODE.md).

## 7. Optional LoRA SFT demonstration (separate GPU and env)

The **already completed PAC demonstration** used
`Qwen/Qwen3-0.6B`, 96 training tickets, 24 held-out tickets, 30
gradient steps, rank-eight LoRA on `q_proj` and `v_proj`. The same
target `PAC-ACK` appears for every ticket, so this is **not** a hard
generalization experiment. Existing PAC results: first loss
`3.746484`, final loss `0.000016492`, nonzero LoRA B norm `3.12949`;
base 0/24 versus adapter 24/24 exact string matches.

Full environment creation, installation pins and steps are in
[experiments/lora_demo/README.md](experiments/lora_demo/README.md).
Use a fresh run directory for each member, e.g.:

```bash
ROOT=/egr/research-pac/$USER
BASE="$ROOT/local_llm"
ENV="$ROOT/conda-envs/lora-demo"
REPO="$ROOT/decs_local_llm"
RUN="$BASE/lora-demo/runs/ack-$USER-$(date -u +%Y%m%dT%H%M%SZ)"
export RUN
export HF_HOME="$BASE/hf_cache"
export HF_HUB_CACHE="$BASE/hf_cache/hub"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=2  # replace 2 with YOUR assigned physical GPU
```

The **LoRA training environment is not the existing vLLM environment**.
After its dependencies and Qwen3-0.6B weights are installed, run:

```bash
"$ENV/bin/python" "$REPO/experiments/lora_demo/demo.py" \
  prepare --run-dir "$RUN"
"$ENV/bin/python" "$REPO/experiments/lora_demo/demo.py" \
  train --run-dir "$RUN" --steps 30 --lr 0.0005
cat "$RUN/training_receipt.json"
```

The script refuses to overwrite an existing nonempty adapter. Do not
use `--force` casually; create a new run name instead. Inspect the
LoRA norm, losses and adapter files.

**Inference is a separate operation.** Stop your coder if it occupies
the assigned GPU; choose an unused `EVAL_PORT`; then in a new terminal:

```bash
ROOT=/egr/research-pac/$USER
BASE="$ROOT/local_llm"
VLLM_ENV="$ROOT/conda-envs/local-agent"
RUN="$BASE/lora-demo/runs/<your-run-name>"   # must match training
EVAL_PORT=18550                             # your unique assigned port
export PATH="$VLLM_ENV/bin:$PATH"
export HF_HOME="$BASE/hf_cache" HF_HUB_CACHE="$BASE/hf_cache/hub"
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=2
command -v ninja || exit 1

"$VLLM_ENV/bin/vllm" serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port "$EVAL_PORT" \
  --served-model-name qwen3-small-base \
  --dtype bfloat16 --max-model-len 2048 \
  --gpu-memory-utilization 0.15 --max-num-seqs 2 \
  --enable-lora --max-lora-rank 8 \
  --lora-modules "toy-lora=$RUN/adapter"
```

Then, in another terminal, **set the variables again** to the same
run name and port (shell variables do not propagate between terminals):

```bash
ROOT=/egr/research-pac/$USER
BASE="$ROOT/local_llm"
ENV="$ROOT/conda-envs/lora-demo"
REPO="$ROOT/decs_local_llm"
RUN="$BASE/lora-demo/runs/<your-run-name>"
EVAL_PORT=18550  # the same port you chose for the server

"$ENV/bin/python" "$REPO/experiments/lora_demo/demo.py" \
  evaluate --run-dir "$RUN" --port "$EVAL_PORT"
cat "$RUN/evaluation_summary.json"
```

The endpoint should advertise both `qwen3-small-base` and
`toy-lora`; the evaluation writes example-level `evaluation.jsonl`.
Confirm behavior rather than promising 24/24 on other drivers or
software. A separate plain `vllm serve` is not tracked by
`pac-llm stop coder`; stop that foreground server with Ctrl+C.

**Shell/Python pitfall:** if using `os.environ["RUN"]`, remember
`export RUN`. Merely writing `RUN=...` does not export the variable.

## 8. Completion evidence and cleanup

Each member records a small run receipt. The following script is
**read-only** except for creating its one requested output file (it
refuses to overwrite an existing receipt):

```bash
ROOT=/egr/research-pac/$USER
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
python3 "$ROOT/decs_local_llm/scripts/record_pac_environment.py" \
  --root "$ROOT" --repo "$ROOT/decs_local_llm" \
  --out "$ROOT/local_llm/lab_receipts/workshop-$STAMP.json"
```

Include these measurements and outcomes with that receipt:

- Host and date; source `git rev-parse HEAD`.
- Assigned GPU and distinct ports, actual `nvidia-smi` free VRAM.
- `pac-llm doctor`; `pip show vllm` from the vLLM env.
- CPU test summary; chat/tool smoke output.
- Aider/OpenCode prompt, manual diff and unit test result.
- LoRA `data_manifest.json`, `training_receipt.json`, and
  `evaluation_summary.json`, if performing SFT.

Do not copy private environment files, model caches, tokens or chats into
the report. If `pac-llm` launched the service, cleanly stop it:

```bash
pac-llm stop coder
pac-llm stop web
pac-llm status
```

These commands intentionally refuse to kill unmanaged ports/processes.

## 9. Troubleshooting reference

| Symptom | First check |
| --- | --- |
| No terminal output during pip install | The installer logs redirect output; inspect `logs/*/setup.log` and `ps` |
| vLLM compiled but fails in FlashInfer JIT | `ninja` must be on `PATH` for child processes, not just installed |
| LoRA training says “Existing adapter” | Inspect previous receipt; choose a new `RUN`, do not overwrite |
| Python `KeyError: 'RUN'` | `export RUN` before invoking Python |
| Model chat works but agent never calls tools | Check tool parser, tool smoke, generated `tool_calls`, context |
| OpenCode loops over files | Stop session, use direct filenames and bounded steps; inspect tool budget |
| OpenCode says it edited, tests all pass | Check `git diff` for unrelated changes and tests that pass for wrong reasons |
| Port occupied | Different student or unmanaged service may own it; select a unique port |
| Context overflow | Increase 8k to 16k **only if VRAM allows**; avoid massive read prompts |
| GPU headroom check fails | Choose another assigned GPU, smaller model or lower concurrency; don't bypass blindly |
| WebUI shows “No models yet” in Workspace | Check Chat model selector and backend API connection |
| Fresh-user setup fails | Keep log, package versions and OS/driver information; document as unresolved |

## 10. What is actually validated?

| Component | Evidence | Status |
| --- | --- | --- |
| Per-user PAC vLLM / Open WebUI / Aider | Earlier live operation | Tested on one account |
| Qwen3-Coder-30B + OpenCode read/edit | Read tasks, worktree edit, human repair; 19/19 tests | Tested on one account |
| 3/5 steps increased to 8/16 | PAC test showed five steps too short; later edit succeeded | Containment improvement |
| Qwen3-0.6B PEFT LoRA | Two 30-step runs; same numerical losses, adapter files | Tested on one account |
| Static vLLM base + LoRA | `/v1/models` returned two IDs; 0/24 vs 24/24 | Tested on one account |
| Qwen3-4B workshop model + Hermes | Official model/template evidence; no PAC trial yet | **Not validated** |
| Fresh-user full setup and shared-GPU teaching | Requires an actual second user on PAC | **Not validated** |

A repository unit test **cannot** prove GPU compatibility, driver
behavior, tool-call quality, or multi-user port/VRAM safety. Validate
these gates on a clean lab account before promising a frictionless class.
