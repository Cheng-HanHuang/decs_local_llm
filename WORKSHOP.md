# PAC local-LLM workshop: complete, repeatable lab exercise

**Audience:** a lab member with a new, personal PAC account and read access to
[Cheng-HanHuang/decs_local_llm](https://github.com/Cheng-HanHuang/decs_local_llm).
No ChatGPT assistance should be needed for the normal path. If any gate fails,
stop at that gate and use its **local logs and troubleshooting table** rather
than changing unrelated packages.

**Teaching model:** Qwen3-4B-Instruct-2507 (vLLM + WebUI + Aider + OpenCode).
**Separate LoRA demonstration:** Qwen3-0.6B (training + base/adapter inference).
The LoRA model is intentionally smaller; it is not a replacement coder.

**Status:** the 4B model, Hermes tool calling, OpenCode reading/editing, and the
LoRA demonstration were tested on one PAC account. A *fresh-account automated
install* and *simultaneous independent accounts* require your reproduction.

## 0. Before doing anything

You need Git access to the private repository, `git`, `python3`,
`conda`, network access for package downloads, sufficient research storage,
and permission to use an allocated PAC GPU. Every account must use **its own**
`/egr/research-pac/$USER`, its own model cache and its own ports.

Ask the workshop coordinator for:
- One physical GPU index to use for the live exercises (for example `2`).
- Three **unique ports on the shared PAC host**: coder, WebUI, evaluation.
  Two users cannot both bind port `8000` on the same host.
- A distinct local laptop tunnel port if using WebUI from a browser.

Do **not** copy someone's credentials, environment, cache, chat history,
LoRA adapter, or WebUI database. Services listen on `127.0.0.1`, but any
account on that same host may be able to reach the endpoint. Use only
synthetic, nonsensitive prompts and coordinate GPU usage.

Check:

```bash
echo "USER=$USER"
command -v git
command -v python3
command -v conda
test -d "/egr/research-pac/$USER" && echo "Personal PAC root: OK"
```

If `conda` is absent, activate the PAC-supported Conda installation first.
Do not use `sudo` or modify someone else's environment. A `conda` failure
is a prerequisite failure; rerunning the installer will not solve it.

## 1. Clone the workshop branch

The coordinator grants GitHub collaborator access first. Configure GitHub SSH
access on PAC (or use your own authenticated HTTPS Git workflow). Then:

```bash
export PAC_LLM_ROOT="/egr/research-pac/$USER"
ROOT="$PAC_LLM_ROOT"
cd "$ROOT"

git clone --branch feature/pac-lab-autopilot-4b \
  git@github.com:Cheng-HanHuang/decs_local_llm.git

REPO="$ROOT/decs_local_llm"
git -C "$REPO" rev-parse HEAD
git -C "$REPO" status --short
```

If the repository already exists, do not clone over it. Instead inspect
`git status` and use a separate worktree, or ask the coordinator which
checkout to use. **For reproducing this workshop, pin the source commit.**
The branch may receive small fixes; record the commit you actually ran.

## 2. Install everything in an automated, resumable run (no GPU jobs)

The installer handles the following, in order:

1. CPU-only unit tests.
2. Personal vLLM + Ninja environment.
3. Personal CPU-only Open WebUI environment.
4. Aider environment.
5. Node.js + OpenCode CLI.
6. Separate PyTorch CUDA 12.8 + Transformers + PEFT LoRA environment.
7. Public Qwen3-4B model, then public Qwen3-0.6B model.
8. Deterministic LoRA demo dataset.
9. Offline verification of tools, both cached models and tokenization.

It **never starts vLLM, uses a GPU for training, initializes accounts,
changes PAC permissions or executes a coding agent**. It saves package
installation logs in your personal `local_llm/lab_autopilot/stages`.

Preview the plan (no downloads):

```bash
cd "$REPO"
bash scripts/lab_install.sh plan
```

Start the installer once; it continues even if your SSH shell disconnects:

```bash
bash scripts/lab_install.sh start
```

Read progress whenever you reconnect:

```bash
cd "/egr/research-pac/$USER/decs_local_llm"
bash scripts/lab_install.sh status
bash scripts/lab_install.sh logs
# Optional: Ctrl+C stops following the log, NOT the installer.
bash scripts/lab_install.sh follow
```

Success means `Status: complete` plus `OFFLINE SETUP VERIFICATION: PASS`
in `$ROOT/local_llm/lab_autopilot/stages/verify.log`. Confirm:

```bash
bash scripts/lab_install.sh verify
python3 -m unittest discover -s tests -v
```

If interrupted or failed, the same commands continue from the first
incomplete stage:

```bash
bash scripts/lab_install.sh logs
bash scripts/lab_install.sh status
bash scripts/lab_install.sh resume
```

**Do not run two installers simultaneously.** The installer uses a
per-user file lock and refuses a second active run. A failed stage's
specific output is under `lab_autopilot/stages/<stage>.log`; keep that
file when reporting a problem. The launcher skips stages already completed
for the same source commit. If the repo commit changes, stages are
revalidated/re-executed (existing working executables are preserved).
Model downloads are cached/resumable by Hugging Face.

### Installation locations

| Item | Personal location |
| --- | --- |
| Git source | `$ROOT/decs_local_llm` |
| Launcher | `$ROOT/local_llm/bin/pac-llm` (symlink) |
| vLLM + Ninja | `$ROOT/conda-envs/local-agent` |
| Open WebUI | `$ROOT/conda-envs/open-webui` |
| Aider | `$ROOT/conda-envs/aider` (or existing `local_llm/bin/aider`) |
| OpenCode Node | `$ROOT/conda-envs/opencode-node` |
| OpenCode CLI and state | `$ROOT/local_llm/opencode` |
| LoRA training | `$ROOT/conda-envs/lora-demo` |
| Qwen model weights | `$ROOT/local_llm/hf_cache` |
| LoRA dataset/run | `$ROOT/local_llm/lora-demo/runs/workshop-ack-v1` |
| Setup logs and status | `$ROOT/local_llm/lab_autopilot` |

The installer deliberately sets `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` and
does not forward common Hugging Face token environment variables for
public downloads; this avoids a previously observed expired-token 401.
It does not delete the student's saved tokens.

### In every *new* terminal

Shell variables are not inherited by new SSH terminals. Reestablish them:

```bash
export PAC_LLM_ROOT="/egr/research-pac/$USER"
export ROOT="$PAC_LLM_ROOT"
export BASE="$ROOT/local_llm"
export REPO="$ROOT/decs_local_llm"
export PATH="$BASE/bin:$PATH"
```

If `pac-llm: command not found`, run the four lines above or invoke
`python3 "$REPO/bin/pac-llm"` by explicit path.

## 3. Set per-user GPU and ports

**Replace** the example values with the ones assigned by the coordinator.
Never use the same API ports as another student on the same host.

```bash
GPU=2
CODER_PORT=18501
WEB_PORT=19501
REASONER_PORT=18502
EVAL_PORT=18550

# A first-time account has no personal configuration yet.
pac-llm config init --gpu "$GPU" --coder-port "$CODER_PORT" \
  --web-port "$WEB_PORT" --reasoner-port "$REASONER_PORT"
pac-llm config show
pac-llm doctor
```

If the configuration already exists, use `pac-llm config show`, then
`pac-llm config set gpu 2` and
`pac-llm config set coder_port 18501` (etc.) instead of running
`config init` again.

The 4B profile uses BF16, 16,384-token context, Hermes tool parser and a
`0.16` memory utilization target. On a 96 GiB GPU, this corresponds
to roughly 15.3 GiB targeted vLLM allocation, **not** a guarantee.
Check actual GPU memory before every server launch:

```bash
nvidia-smi -i "$GPU" --query-gpu=index,memory.total,memory.used,memory.free \
  --format=csv
```

## 4. vLLM: first live GPU and tool-calling test

```bash
pac-llm start coder --gpu "$GPU" --memory-util 0.16 --max-model-len 16384
pac-llm status
pac-llm logs coder --lines 30
```

The first command returns after spawning a managed process, **not after
the API is ready**. Do not immediately repeat `start` when the port
still refuses connections. Check `status` until it reports
`coder ... api=ready`, or inspect the log if it says stale record.

```bash
python3 "$REPO/scripts/pac_workshop_smoke.py" \
  --port "$CODER_PORT" --model local-qwen-coder --test-tools
```

Pass criteria:
- `Server models: ['local-qwen-coder']`.
- Chat reply exactly `PAC-READY`.
- A structured `add_two_integers` function call with `a=2, b=3`.
- The tool was **not executed**; the test only checks its JSON structure.

If ordinary chat works but this fails, **do not** claim the coding agent
passed. Check that the coder profile uses Qwen3-4B with `hermes` parser,
inspect the vLLM log and record the failing result before changing settings.

## 5. Open WebUI: chat in a browser

WebUI consumes **CPU** and connects to the already-running vLLM API.

```bash
pac-llm start web
pac-llm status
pac-llm logs web --lines 30
```

On your **own laptop**, in a separate shell (adjust laptop local port and
your SSH alias):

```powershell
ssh -N -L 19080:127.0.0.1:19501 your-user@pac
```

Open `http://127.0.0.1:19080` in your browser and sign up for your
**own** local WebUI account. In the Chat model selector, select
`local-qwen-coder`; ask for a two-line explanation of `LoRA`.
Verify that a normal answer appears. **The Workspace → Models page is
not the same as the Chat model selector**.

If no model appears, first check `curl http://127.0.0.1:$CODER_PORT/v1/models`
on PAC and then the WebUI connection configuration. Do not bind the API
or WebUI to `0.0.0.0` on a shared host to fix tunneling.

## 6. Aider: ask-only code reading

Leave the 4B vLLM server running. Aider is a separate CLI client
connected to the same `local-qwen-coder` API:

```bash
cd "$REPO"
pac-llm aider
```

In Aider, add the real file:

```text
/read-only bin/pac-llm
```

Then prompt:

> Find `cmd_start` in `bin/pac-llm`. Explain GPU checks, port checks
> and process management in three short bullets. Do not edit files.

Pass criteria: answer refers to actual functions in the file; Aider is
in ask-only mode; `git status --short` shows no new code changes.
Exit Aider when finished.

## 7. OpenCode: read-only source inspection

Still using the same running vLLM server:

```bash
cd "$REPO"
pac-llm opencode --preview
pac-llm opencode --steps 8
```

Prompt 1:

> Read `bin/pac-llm`. Find the Python function `cmd_start` inside that
> file. Explain its role in three short bullets. Do not run commands or edit.

Prompt 2:

> Read `bin/pac-llm`. Explain how `proc_identity`, `owns_process`
> and `cmd_stop` work together. State one limitation. Do not edit.

Check the actual source yourself. During our trial, the 4B model got the
basic code relationships right but made one erroneous safety inference.
A tool call or a confident answer is **not** a correctness guarantee.

## 8. OpenCode: controlled edit and self-repair

**Do not run edit mode in the primary Git checkout.** Use a disposable,
linked worktree beneath the private OpenCode worktree directory.

```bash
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WORK="$BASE/opencode/worktrees/workshop-$STAMP"
mkdir -p "$BASE/opencode/worktrees"
git -C "$REPO" worktree add -b "workshop/$USER-$STAMP" "$WORK" HEAD
cd "$WORK"
pac-llm opencode --mode edit --steps 16
```

Prompt:

> Open only `tests/test_setup_config.py`. Inside the existing class
> `SetupConfigTests`, add one new unit test confirming
> `pac-llm config set gpu -1` exits nonzero and reports a nonnegative
> integer requirement. Modify no other tests or files. Ask before editing,
> and do not run any shell commands.

Review permissions and approve only the one requested edit. Exit OpenCode,
then run the checks **yourself**:

```bash
git status --short
git diff --check
git diff -- tests/test_setup_config.py
python3 -m unittest discover -s tests -v
```

Pass criteria: exactly one new meaningful test, no unrelated file edits,
all tests pass, and clean `git diff --check`. Our first 4B attempt
incorrectly expected the phrase `GPU index` and one test failed.
After feedback, it repaired the assertion and all 30 then-current tests
passed. If a test fails, quote the exact failure back to OpenCode, limit
its scope to that test, and rerun the suite manually. Do not change
application behavior to match an invented assertion.

Do not choose “always allow” for shell commands or broad writes.
OpenCode permissions are not an OS sandbox. The source changes remain
only in the disposable worktree; do not push them without review.

## 9. LoRA supervised fine-tuning: learn an output convention

This is a **different model and separate environment**:
`Qwen/Qwen3-0.6B`. The installer has already downloaded the weights,
created `$ROOT/conda-envs/lora-demo`, and prepared
`$BASE/lora-demo/runs/workshop-ack-v1`.

All 96 training examples ask for the constant reply `PAC-ACK` and
all 24 held-out examples expect the same target. The purpose is to see
actual LoRA weight updates, **not** to prove reasoning/generalization.
The adapter trains `q_proj` and `v_proj`, rank 8 and alpha 16,
with the base frozen.

Exit OpenCode and stop the 4B service **if it is your managed service
and you will use the same GPU**:

```bash
pac-llm stop coder
pac-llm status
```

From a shell with the variables from section 3:

```bash
export ENV="$ROOT/conda-envs/lora-demo"
export RUN="$BASE/lora-demo/runs/workshop-ack-v1"
export HF_HOME="$BASE/hf_cache"
export HF_HUB_CACHE="$BASE/hf_cache/hub"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$GPU"

"$ENV/bin/python" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

"$ENV/bin/python" "$REPO/experiments/lora_demo/demo.py" train \
  --run-dir "$RUN" --steps 30 --lr 0.0005

cat "$RUN/training_receipt.json"
ls -lh "$RUN/adapter"
```

If the run directory already contains `adapter`, training refuses
to overwrite it. Inspect its receipt or choose a fresh run name,
then run `prepare --run-dir "$RUN"` before training. Do not use
`--force` on someone else's result.

**What to check:** loss decreases; `lora_B_l2_norm` is positive;
`adapter/adapter_model.safetensors` exists. Our earlier PAC run
had 1,146,880 trainable parameters (0.192%) and final training loss
near `1.65e-5`. Timing and exact loss may vary by account.

## 10. Serve base and adapter together; compare 24 examples

Choose an unused **evaluation port** distinct from WebUI and coder.
In **terminal A**, set the variables again (new terminals don't inherit
the earlier shell):

```bash
ROOT="/egr/research-pac/$USER"
BASE="$ROOT/local_llm"
RUN="$BASE/lora-demo/runs/workshop-ack-v1"
VLLM_ENV="$ROOT/conda-envs/local-agent"
EVAL_PORT=18550                 # replace with YOUR assigned evaluation port
export PATH="$VLLM_ENV/bin:$PATH"
export HF_HOME="$BASE/hf_cache"
export HF_HUB_CACHE="$BASE/hf_cache/hub"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=2  # replace with YOUR assigned GPU
command -v ninja || exit 1

"$VLLM_ENV/bin/vllm" serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port "$EVAL_PORT" \
  --served-model-name qwen3-small-base \
  --dtype bfloat16 --max-model-len 2048 \
  --gpu-memory-utilization 0.15 --max-num-seqs 2 \
  --enable-lora --max-lora-rank 8 \
  --lora-modules "toy-lora=$RUN/adapter"
```

This server runs **in the foreground** and is not tracked by
`pac-llm stop coder`. The existing `ninja` on `PATH` is important
for FlashInfer warmup. Check its terminal for startup completion.

In **terminal B**, explicitly set these variables again:

```bash
ROOT="/egr/research-pac/$USER"
BASE="$ROOT/local_llm"
REPO="$ROOT/decs_local_llm"
ENV="$ROOT/conda-envs/lora-demo"
export RUN="$BASE/lora-demo/runs/workshop-ack-v1"
EVAL_PORT=18550                # same port as terminal A

curl -fsS "http://127.0.0.1:$EVAL_PORT/v1/models" | python3 -m json.tool
"$ENV/bin/python" "$REPO/experiments/lora_demo/demo.py" evaluate \
  --run-dir "$RUN" --port "$EVAL_PORT"
cat "$RUN/evaluation_summary.json"
```

Pass criteria: `/v1/models` lists both `qwen3-small-base` and
`toy-lora`; 24 examples are evaluated; the adapter predicts
`PAC-ACK` on held-out ticket IDs (our earlier result was
base 0/24, LoRA 24/24, **not a universal guaranteed outcome**).

To view actual model outputs and avoid the `KeyError: 'RUN'` trap:

```bash
export RUN="$BASE/lora-demo/runs/workshop-ack-v1"
"$ENV/bin/python" - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ["RUN"])
for line in (run / "evaluation.jsonl").read_text().splitlines()[:5]:
    row = json.loads(line)
    print("Prompt:", row["prompt"])
    print("Base:", repr(row["base"]))
    print("LoRA:", repr(row["lora"]))
PY
```

Stop the separate inference server with Ctrl+C in **terminal A**.

## 11. Save evidence and clean up

Run from the original checkout:

```bash
ROOT="/egr/research-pac/$USER"
REPO="$ROOT/decs_local_llm"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

python3 "$REPO/scripts/record_pac_environment.py" \
  --root "$ROOT" --repo "$REPO" \
  --out "$ROOT/local_llm/lab_receipts/workshop-$STAMP.json"
```

Keep your own `lab_autopilot/status.json`, model/API smoke text,
unit-test output, a short description of Aider/OpenCode behavior,
reviewed `git diff`, LoRA `data_manifest.json`, `training_receipt.json`
and `evaluation_summary.json`. Do not include private tokens or
someone else's chats. Give the coordinator just the **summary** and
failure logs with secrets removed.

For any managed services you started:

```bash
export PATH="$ROOT/local_llm/bin:$PATH"
pac-llm stop coder        # only if your managed coder still runs
pac-llm stop web          # only if your managed WebUI still runs
pac-llm status
```

The launcher checks recorded PID, start time and owner before signaling.
It will refuse to terminate another account's unmanaged process.

## Troubleshooting without another AI assistant

| Symptom | Local action |
| --- | --- |
| `conda: command not found` | Use PAC's supported Conda activation first; don't install system-wide |
| GitHub permission denied | Verify collaborator access and SSH key or your HTTPS login |
| Package installation silent | Read `lab_autopilot/stages/<stage>.log`; installations redirect output |
| Installer interrupted | `bash scripts/lab_install.sh status` then `resume` |
| Stage repeatedly fails | Save exact stage log and contact the lab coordinator; don't start over from scratch |
| Hugging Face 401 / invalid OAuth | Installer disables implicit tokens; confirm public endpoint, unset explicit HF tokens |
| No `pac-llm` command | Restore `export PATH="$ROOT/local_llm/bin:$PATH"` in this terminal |
| `coder running / api=not ready` | Inspect `pac-llm logs coder`; the server may still compile |
| Port 8000 or WebUI port busy | Assign a unique personal port, don't kill an unrelated service |
| GPU headroom denied | Ask coordinator for a different GPU/limit, don't bypass with `--force` |
| vLLM FlashInfer `No such file: ninja` | `export PATH="$ROOT/conda-envs/local-agent/bin:$PATH"` |
| WebUI Chat model missing | Check `/v1/models` and the WebUI connection to your coder port |
| Aider reads no code | Explicitly `/read-only bin/pac-llm` rather than relying on repo map |
| OpenCode searches forever | Start new session, ask for one file/function, keep bounded steps |
| OpenCode edit tests fail | Inspect diff and actual error; provide failure as feedback, re-run tests |
| LoRA `Existing adapter` | Preserve earlier receipt; choose new run directory and `prepare` |
| Python `KeyError: RUN` | `export RUN` or pass the path explicitly |
| `Connection refused` on evaluation | Verify foreground vLLM server actually finished starting |
| Different LoRA accuracy | Verify model, seed, data hashes, adapter path and non-thinking chat template |

## What counts as a successful second-account validation?

The labmate must independently report (a) **autopilot complete**, (b)
offline verifier PASS, (c) actual 4B chat and structured tool calling,
(d) WebUI chat, (e) Aider source read, (f) OpenCode read and approved
edit with manually reviewed tests, and (g) 0.6B LoRA training and
base/adapter comparison on their own GPU/run directory.

Everything up to offline verification is install/setup; all later
checks are actual **live application experiments**. A clean second-account
trial is necessary before claiming lab-wide reproducibility.
