# PAC local-LLM lab handout

**Start with [WORKSHOP.md](WORKSHOP.md).** This is the authoritative
step-by-step guide for a **fresh PAC account**, including installation,
WebUI, Aider, OpenCode and LoRA SFT.

The two models are:

- **Coding/agents/UI:** `Qwen/Qwen3-4B-Instruct-2507` through vLLM,
  16k-token context, BF16, `hermes` parser.
- **LoRA supervised learning:** `Qwen/Qwen3-0.6B` through PyTorch/PEFT,
  then base/adapter inference through vLLM.

These are intentionally separate exercises. The smaller 0.6B model is
used to demonstrate training, not as the coding-agent backend.

## One-command setup on a new PAC account

After gaining GitHub collaborator access:

```bash
export ROOT="/egr/research-pac/$USER"
cd "$ROOT"
git clone --branch feature/pac-lab-autopilot-4b \
  git@github.com:Cheng-HanHuang/decs_local_llm.git
cd decs_local_llm

bash scripts/lab_install.sh plan
bash scripts/lab_install.sh start

# Resume monitoring from any SSH session:
bash scripts/lab_install.sh status
bash scripts/lab_install.sh logs

# After "Status: complete":
bash scripts/lab_install.sh verify
```

This setup installs per-user environments and downloads public model weights,
then runs offline validation. **It does not start GPU jobs or training.**
The installation is resumable; use `bash scripts/lab_install.sh resume`
after diagnosing any failed stage.

Next follow [WORKSHOP.md](WORKSHOP.md) in order: personal GPU/ports,
vLLM tool-call check, WebUI via SSH tunnel, ask-only Aider,
read-only OpenCode, an approved worktree edit, 30-step LoRA training,
24-example evaluation and recorded environment evidence.

## Validation status

One PAC account successfully demonstrated: the 4B vLLM API and Hermes
tool parsing, two OpenCode code-reading tasks, controlled editing with one
initial failing test repaired from feedback (all tests passing afterward),
and a 0.6B LoRA adapter yielding 24/24 exact replies against 0/24 base
on the constant-label toy test.

**Fresh-user installation and shared-GPU simultaneous use remain unverified.**
Ask a labmate to execute WORKSHOP.md independently and keep the stage
logs and training/evaluation receipts. Lab-wide reproducibility is a
result to be tested, not an assumption.

## Separate technical references

- [SETUP.md](SETUP.md): configuration, storage and explicit install components
- [OPENCODE.md](OPENCODE.md): agent permissions and bounded steps
- [experiments/lora_demo/README.md](experiments/lora_demo/README.md):
  LoRA SFT implementation, hyperparameters and evaluation
- [README.md](README.md): overview and everyday commands
