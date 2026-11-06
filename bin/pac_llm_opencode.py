"""Safe-by-default OpenCode V2 launcher for a local vLLM coding model.

No third-party dependencies. This is a tool-permission profile, not an OS
sandbox; run untrusted tools or code only in a separate OS sandbox/account.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request


def _fail(msg):
    raise RuntimeError(msg)


def _git(cwd: Path, *args):
    p = subprocess.run(
        ["git", "-C", str(cwd), *args],
        text=True, capture_output=True, check=False,
    )
    if p.returncode:
        _fail("Git check failed: " + p.stderr.strip()[-300:])
    return p.stdout.strip()


def check_disposable_worktree(w, cwd: Path):
    """Require edit mode to operate only in an explicit separate linked worktree."""
    allowed = (w.base / "opencode" / "worktrees").resolve()
    repo = Path(_git(cwd, "rev-parse", "--show-toplevel")).resolve()
    if repo == allowed or allowed not in repo.parents:
        _fail(f"Edit mode requires a linked Git worktree under {allowed}")
    git_dir = Path(_git(repo, "rev-parse", "--absolute-git-dir")).resolve()
    common_raw = _git(repo, "rev-parse", "--git-common-dir")
    common = (Path(common_raw) if Path(common_raw).is_absolute()
              else repo / common_raw).resolve()
    if git_dir == common:
        _fail("Edit mode requires a linked worktree, not a primary checkout.")
    return repo


def permission_rules(mode: str):
    if mode not in ("read", "edit"):
        _fail("Mode must be read or edit")
    # Later matching rules win in OpenCode V2.
    rules = [
        {"action": "*", "resource": "*", "effect": "deny"},
        {"action": "execute", "resource": "*", "effect": "allow"},
        {"action": "read", "resource": "*", "effect": "allow"},
        {"action": "glob", "resource": "*", "effect": "allow"},
        {"action": "grep", "resource": "*", "effect": "allow"},
        {"action": "question", "resource": "*", "effect": "allow"},
    ]
    if mode == "edit":
        rules.extend([
            {"action": "edit", "resource": "*", "effect": "ask"},
            {"action": "shell", "resource": "*", "effect": "ask"},
        ])
    rules.extend([
        {"action": "read", "resource": "*.env", "effect": "deny"},
        {"action": "read", "resource": "*.env.*", "effect": "deny"},
        {"action": "read", "resource": "*secret*", "effect": "deny"},
        {"action": "read", "resource": "*credential*", "effect": "deny"},
        {"action": "external_directory", "resource": "*", "effect": "deny"},
    ])
    return rules


def make_config(port: int, model_id: str, context: int, mode: str):
    output = max(128, min(2048, context // 4))
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "model": "pac/coder",
        "small_model": "pac/coder",
        "providers": {
            "pac": {
                "name": "PAC local vLLM",
                "package": "@opencode/ai/providers/openai-compatible",
                "settings": {"baseURL": f"http://127.0.0.1:{port}/v1"},
                "models": {
                    "coder": {
                        "modelID": model_id,
                        "name": model_id,
                        "capabilities": {
                            "tools": True, "input": ["text"], "output": ["text"],
                        },
                        "limit": {"context": context, "output": output},
                    },
                },
            },
        },
        "permissions": permission_rules(mode),
        "policies": [
            {"action": "tool.use", "resource": "shell:git push *", "effect": "deny"},
            {"action": "tool.use", "resource": "shell:sudo *", "effect": "deny"},
            {"action": "tool.use", "resource": "shell:rm *", "effect": "deny"},
        ],
        "update": "disable",
    }
    return cfg


def probe_model(port: int, preferred=None):
    url = f"http://127.0.0.1:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            data = json.load(response)
    except Exception as exc:
        _fail(f"Cannot reach vLLM at {url}: {exc}. Start coder first.")
    cards = data.get("data", [])
    if not cards:
        _fail(f"No model advertised by vLLM at {url}")
    model = next((m for m in cards if m.get("id") == preferred), cards[0])
    context = model.get("max_model_len") or 8192
    try:
        context = min(16384, max(1024, int(context)))
    except (TypeError, ValueError):
        context = 8192
    return model["id"], context


def launch(w, args):
    mode = args.mode
    cwd = Path.cwd().resolve()
    if mode == "edit":
        check_disposable_worktree(w, cwd)
    node_bin = w.root / "conda-envs" / "opencode-node" / "bin"
    executable = w.base / "opencode" / "npm" / "bin" / "opencode"
    if not (executable.is_file() and os.access(executable, os.X_OK)):
        _fail(f"OpenCode missing: {executable}. Run pac-llm setup --components opencode --yes")
    env = os.environ.copy()
    env["PATH"] = str(node_bin) + os.pathsep + env.get("PATH", "")
    if not shutil.which("node", path=env["PATH"]):
        _fail("Node.js missing; run pac-llm setup --components opencode --yes")
    base = w.base / "opencode"
    env.update({
        "XDG_CONFIG_HOME": str(base / "config"),
        "XDG_DATA_HOME": str(base / "data"),
        "XDG_CACHE_HOME": str(base / "cache"),
        "XDG_STATE_HOME": str(base / "state"),
        "OPENCODE_DISABLE_FREE_MODELS": "1",
        "NPM_CONFIG_CACHE": str(base / "npm-cache"),
    })
    for path in ("config", "data", "cache", "state", "npm-cache"):
        d = base / path
        d.mkdir(parents=True, exist_ok=True)
        d.chmod(0o700)
    # Don't allow OpenCode to pick up unrelated commercial-provider credentials.
    for secret in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
                   "GEMINI_API_KEY", "OPENROUTER_API_KEY", "AZURE_OPENAI_API_KEY"):
        env.pop(secret, None)
    preferred = None
    from json import JSONDecodeError
    rec_path = w.state_file("coder")
    if rec_path.is_file():
        try:
            preferred = json.loads(rec_path.read_text()).get("served_name")
        except (OSError, JSONDecodeError):
            pass
    if args.preview:
        model, context = preferred or "local-qwen-coder", 8192
    else:
        model, context = probe_model(w.port("coder"), preferred)
    config = make_config(w.port("coder"), model, context, mode)
    # Inline config has higher precedence than untrusted project configs.
    # OpenCode permissions are NOT an OS-level sandbox.
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
    target = base / "config" / f"agent-{mode}.json"
    target.write_text(json.dumps(config, indent=2) + "\n")
    target.chmod(0o600)
    env["OPENCODE_CONFIG"] = str(target)
    if args.preview:
        print(json.dumps(config, indent=2))
        print(f"\nPreview only: {cwd}, mode={mode}. No agent launched.")
        return
    print(f"OpenCode: mode={mode} model={model} context={context}", flush=True)
    print(f"Repository: {cwd}", flush=True)
    print("Review permission requests; avoid 'always' for shell/edit.", flush=True)
    os.execvpe(str(executable), [str(executable), "--standalone"], env)
