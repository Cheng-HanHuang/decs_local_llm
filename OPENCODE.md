# OpenCode agent workflow on PAC and slim (OpenCode V2)

OpenCode is a **coding agent**, not a model server. It uses the running
`pac-llm start coder` vLLM API. This setup has been designed for OpenCode
V2.2.5, the version installed on PAC on November 5, 2026.

## 1. Install on a new user's account (one-time, opt-in)

From an accessible `decs_local_llm` checkout (the feature branch until merged):

```bash
pac-llm setup --components opencode       # preview: no changes
pac-llm setup --components opencode --yes # creates per-user Node.js/npm; installs CLI
```

Existing users with
`$PAC_LLM_ROOT/conda-envs/opencode-node/bin/node` and
`$PAC_LLM_BASE/opencode/npm/bin/opencode` will be detected and left unchanged.
Node.js and npm go into a separate Conda environment; the OpenCode CLI and
state stay in personal research storage, not the home directory or this repo.

## 2. Pick your GPU (per-user or per-launch)

```bash
pac-llm config set gpu 2              # example personal choice
pac-llm doctor                        # check current free VRAM and co-tenancy
pac-llm start coder --gpu 2 --memory-util 0.68 --max-model-len 8192
pac-llm status
```

These memory values are examples for an existing 96 GiB GPU with other jobs.
Always use current `pac-llm doctor` output rather than assuming capacity.
vLLM may still run out of memory if concurrent GPU jobs grow.

The vLLM server needs `--enable-auto-tool-choice --tool-call-parser qwen3_xml`
for the default Qwen3-Coder model, already enabled by the launcher.
Custom models need compatible tool parser configuration. The agent may not
function correctly with models that do not follow tool-call conventions.

## 3. Read-only first

```bash
cd "$PAC_LLM_ROOT/decs_local_llm"
pac-llm opencode --preview  # inspect effective model/permission configuration
pac-llm opencode            # defaults to read mode and private --standalone server
```

Suggested prompt:

> Read bin/pac-llm and bin/pac_llm_bootstrap.py. Explain cmd_start,
> spawn, owns_process and cmd_stop, with exact file/function references.
> Do not modify or execute any source files.

Read mode denies OpenCode file edits, shell commands, subagents, and external
directory access. It permits read/search, questions and Code Mode execution
for tool orchestration (nested tool permissions still apply). It is **not a
security sandbox**: the agent operates with your user's OS authority and
Code Mode's network capabilities. Do not use sensitive repositories for initial
tests or assume permissions isolate hostile prompts from your data.

The generated configuration and OpenCode state remain inside
`$PAC_LLM_BASE/opencode`. The launcher sets explicit V2 `permissions` and
`OPENCODE_CONFIG_CONTENT`, which has higher config priority than project
configuration. Your real provider endpoint is selected from the local
`/v1/models` response, including its reported context length when present.

To exit, use OpenCode's exit action or Ctrl+C. The vLLM server remains running.

## If OpenCode repeatedly summarizes or rereads the same file

During a PAC trial with Qwen3-Coder and vLLM, the model repeatedly emitted
`Objective / Decisions / Work State / Next Move` summaries, reread the same
file, and entered automatic compaction without answering. This is **not**
normal task completion. Interrupt it with Esc. A similar model/provider issue
was reported upstream:
https://github.com/anomalyco/opencode/issues/22792

The launcher now uses **three read-agent steps** (five edit-agent steps)
and disables automatic compaction. These are **containment measures**, not
proof of a fix: upstream reporters observed looping even with auto-compaction
disabled. Start a **new OpenCode session** after updating; an existing session
can retain its old agent selection.

A small model context can make matters worse. Aider consumed about 10k tokens
when reading both launcher files, so an 8k vLLM context may be inadequate.
For a new test, if GPU memory permits, serve Qwen with 16,384 tokens instead
and ask it to read **one small function at a time**. Do not restart a current
GPU job unless you have adequate free memory and permission to do so.

Diagnostic test: in a *new* session ask, `Read bin/pac-llm and identify only
the exact Python function implementing start. Reply in one sentence.`
If the model loops again, use `pac-llm aider` for repo reading and record
that OpenCode + local Qwen needs further compatibility work; do not attempt
unbounded edit automation.

## 4. Controlled editing in a disposable Git worktree

**Never test edit mode in the primary repository or a research workspace with
valuable uncommitted changes.**

```bash
ROOT="${PAC_LLM_ROOT:-/egr/research-pac/$USER}"
BASE="$ROOT/local_llm"
REPO="$ROOT/decs_local_llm"
WORK="$BASE/opencode/worktrees/decs-local-llm-smoke"

mkdir -p "$BASE/opencode/worktrees"
git -C "$REPO" worktree add -b agent/opencode-smoke "$WORK" HEAD
cd "$WORK"

pac-llm opencode --mode edit
```

Edit mode checks that the current project is a *linked worktree* beneath the
designated private worktree directory. File edits and shell commands require
explicit user approval; pushes, sudo and rm commands are hard-denied where
they match the configured policies. Do **not** choose "always" for broad shell
or editing approvals. The permission layer does not replace an OS sandbox.

Suggested task:

> Add one small unit test for duplicate GPU selection in
> tests/test_launcher.py. Make no other changes. Ask before each write
> or shell command, and summarize the diff.

After exiting, verify manually:

```bash
git status --short
git diff --check
git diff
python3 -m unittest discover -s tests -v
```

Clean up only when you are sure no changes need saving:

```bash
cd "$ROOT/decs_local_llm"
git worktree remove "$WORK"   # requires clean worktree
git branch -d agent/opencode-smoke
```

Do not force-remove a worktree containing unsaved work.

## 5. End the session and free GPU memory

```bash
pac-llm stop coder
pac-llm status
nvidia-smi -i 2 --query-gpu=memory.used,memory.free --format=csv
```

Open WebUI can remain running; its saved chats are independent of GPU/model
processes.

## Notes for lab members

- Each user needs a private research root, configurations and CLI installation.
- Loopback ports are **shared across user accounts** on the same PAC host.
  Assign unique coder/web ports with `pac-llm config init` or `config set`.
- A custom system/node installation can be used manually, but the setup
  component uses per-user Conda Node.js 22 for reproducibility.
- OpenCode V2 and V1 have different config schemas. This module requires V2.
- The first-time install on a fresh lab account and the OpenCode tool-call
  behavior against local Qwen both require live validation.
