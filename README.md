# DECS Local LLM — reproducible PAC workshop

A per-user, localhost-only toolkit for running a **small local language
model** on PAC, using it from **Open WebUI**, **Aider**, and **OpenCode**,
then learning **LoRA supervised fine-tuning** on a separate tiny model.

The tested coding/agent model is **[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)**,
served through vLLM 0.23.0 with **Hermes** tool parsing. The fine-tuning
exercise uses **[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)**.
Both are downloaded into the student's own Hugging Face cache.
Nothing in this repository includes model weights, credentials, WebUI
history, or trained adapters.

## New lab member: start here

The **[WORKSHOP.md](WORKSHOP.md)** guide is the main handout. It includes
personal PAC paths, a resumable installer, GPU/port coordination, live
verification exercises, exact sample prompts, expected results, troubleshooting
and cleanup.

Each student needs collaborator access to this private repository and an
individual PAC account. Check for `git`, `python3` and `conda`
before running the installer.

```bash
export PAC_LLM_ROOT="/egr/research-pac/$USER"
cd "$PAC_LLM_ROOT"
git clone --branch feature/pac-lab-autopilot-4b \
  git@github.com:Cheng-HanHuang/decs_local_llm.git
cd decs_local_llm

# Inspect before installing; nothing is modified.
bash scripts/lab_install.sh plan

# Explicitly install/download in a resumable background run.
# This NEVER runs GPU jobs, starts model servers, or trains models.
bash scripts/lab_install.sh start

# Monitoring from the same or another SSH terminal:
bash scripts/lab_install.sh status
bash scripts/lab_install.sh logs
bash scripts/lab_install.sh follow  # Ctrl+C exits the log follower only

# Once complete:
bash scripts/lab_install.sh verify
python3 -m unittest discover -s tests -v
```

If setup stops at a stage, inspect
`$PAC_LLM_ROOT/local_llm/lab_autopilot/stages/<stage>.log`
and rerun `bash scripts/lab_install.sh resume`. Successful stages
are recorded under `lab_autopilot/status.json`. Do not start multiple
installers for the same account.

**Important:** Although the model/agents and LoRA experiment passed
on one PAC account, a completely new account and multi-user co-tenancy
are still in validation. This handout is designed to collect that evidence,
not to pretend untested installations already worked.

## What is included?

| Component | Model / tooling | Purpose |
| --- | --- | --- |
| vLLM | Qwen3-4B-Instruct-2507, BF16, Hermes | Small local chat and tool API |
| Open WebUI | Separate CPU environment | Browser chat through an SSH tunnel |
| Aider | Separate per-user CLI | Ask-only source reading |
| OpenCode | Node.js + isolated CLI/state | Read-only agent and approved worktree edits |
| LoRA SFT | Qwen3-0.6B, PyTorch + PEFT | Train an adapter, compare base and adapted |
| Installer | `scripts/lab_install.sh` | Resume package/model downloads |
| Offline verifier | `scripts/lab_verify.py` | Inspect files, model cache and tokenizer |
| Live tool smoke | `scripts/pac_workshop_smoke.py` | Verify actual structured tool calls |
| Receipt | `scripts/record_pac_environment.py` | Save non-secret environment fingerprint |

## Personal runtime paths

```text
/egr/research-pac/$USER/
├── decs_local_llm/                 # Git checkout (code/docs)
├── conda-envs/
│   ├── local-agent/                # vLLM 0.23.0 + Ninja
│   ├── open-webui/                 # Open WebUI
│   ├── aider/                      # Aider, unless existing uv executable
│   ├── opencode-node/              # Node.js/npm
│   └── lora-demo/                  # separate PyTorch/PEFT SFT env
└── local_llm/
    ├── bin/pac-llm                 # symlink to Git checkout
    ├── config/pac-llm.json         # personal GPU and ports
    ├── hf_cache/                   # downloaded model weights
    ├── lab_autopilot/              # status + stage logs
    ├── opencode/                   # client state and worktrees
    ├── webui/                      # personal chat/account data
    └── lora-demo/runs/             # datasets, adapters, training receipts
```

**Each student gets unique ports on the same PAC host.** Per-user file
paths do not imply port isolation. GPU memory is shared as well.
Coordinate resources and follow PAC rules. The 4B vLLM profile uses a
starting `--gpu-memory-utilization 0.16` and 16,384-token context.
On the tested 96 GiB GPU, model weights occupied 7.64 GiB and its KV
cache had 6.59 GiB available. These are observations for one run, not a
multi-user safety guarantee.

## Common commands (after installation)

In **each fresh SSH terminal**:

```bash
export ROOT="/egr/research-pac/$USER"
export PATH="$ROOT/local_llm/bin:$PATH"
```

Choose unique ports and an assigned GPU following WORKSHOP.md:

```bash
pac-llm config init --gpu 2 --coder-port 18501 \
  --reasoner-port 18502 --web-port 19501
pac-llm config show
pac-llm doctor

pac-llm start coder                  # Qwen3-4B, uses saved GPU/port
pac-llm status                       # wait for coder API ready
pac-llm logs coder --lines 30

python3 "$ROOT/decs_local_llm/scripts/pac_workshop_smoke.py" \
  --port 18501 --model local-qwen-coder --test-tools

pac-llm start web                    # browser chat, private data
pac-llm aider                        # ask-only source reading
pac-llm opencode --preview
pac-llm opencode --steps 8           # read-only agent
```

For guarded edits, create a **linked worktree** beneath
`$ROOT/local_llm/opencode/worktrees/` and use
`pac-llm opencode --mode edit --steps 16`. Review every diff and run
tests yourself. The permission layer is not a security sandbox.

The LoRA demo is separate: see
[experiments/lora_demo/README.md](experiments/lora_demo/README.md) and
WORKSHOP.md §§9–10. LoRA uses **Qwen3-0.6B**, rank eight on
`q_proj,v_proj` and 30 optimizer steps. Our one-account PAC result
was 0/24 exact matches with the base and 24/24 with LoRA on the synthetic,
constant-target `PAC-ACK` task. This shows the adapter learned a
response convention, **not** nontrivial reasoning or input-dependent mapping.

## Design and safety

- Setup is opt-in. `plan`, `status`, `doctor`, and CPU unit tests
  do not install packages or allocate GPU memory.
- Installer and service manager use per-user research storage. The setup
  process is resumable and serialized with a user-local file lock.
- Downloads for public Hugging Face models disable implicit authentication
  to avoid cached invalid OAuth tokens; no saved credentials are deleted.
- vLLM/WebUI bind to `127.0.0.1`. On a shared host, localhost **is
  not authentication** against other accounts. Use synthetic inputs.
- The launcher checks process identity before stopping managed services,
  refuses occupied ports and requires a GPU-memory snapshot. It cannot
  reserve VRAM against co-tenants.
- OpenCode read mode denies writes; edit mode requires an explicit linked
  worktree and approval. Neither is an OS-level sandbox.
- The exact dependency pins are installation targets, **not a fully
  locked transitive dependency environment**. Record the actual source
  revision and package versions for scientific reproducibility.
- The optional large reasoning profile is not needed or downloaded for
  this workshop.

More references: [SETUP.md](SETUP.md),
[OPENCODE.md](OPENCODE.md),
[LAB_HANDOUT.md](LAB_HANDOUT.md), and
[LoRA experiment](experiments/lora_demo/README.md).
