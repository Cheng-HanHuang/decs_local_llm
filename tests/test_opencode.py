"""CPU-only OpenCode V2 setup, safety and model-config tests."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from io import BytesIO

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin/pac-llm"
SPEC = importlib.util.spec_from_file_location("pac_llm_opencode", ROOT / "bin/pac_llm_opencode.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class OpenCodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = dict(os.environ, PAC_LLM_ROOT=str(self.root))
        for key in ("PAC_LLM_GPU", "PAC_LLM_GPUS", "PAC_LLM_MODEL",
                    "PAC_LLM_CODER_PORT", "PAC_LLM_REASONER_PORT", "PAC_LLM_WEB_PORT"):
            self.env.pop(key, None)

    def cmd(self, *args, cwd=None):
        return subprocess.run([sys.executable, str(SCRIPT), *args],
                              capture_output=True, text=True, env=self.env, timeout=15,
                              cwd=str(cwd or self.root))

    def test_read_mode_is_non_mutating(self):
        cfg = mod.make_config(18100, "local-qwen-coder", 8192, "read")
        self.assertEqual(cfg["model"], "pac/coder")
        self.assertEqual(cfg["default_agent"], "pac-read")
        self.assertEqual(cfg["agents"]["pac-read"]["steps"], 3)
        self.assertFalse(cfg["compaction"]["auto"])
        self.assertIn({"action": "tool.use", "resource": "shell:*", "effect": "deny"}, cfg["policies"])
        self.assertIn({"action": "tool.use", "resource": "edit:*", "effect": "deny"}, cfg["policies"])
        self.assertEqual(cfg["providers"]["pac"]["models"]["coder"]["modelID"], "local-qwen-coder")
        self.assertEqual(cfg["providers"]["pac"]["settings"]["baseURL"], "http://127.0.0.1:18100/v1")
        rules = cfg["permissions"]
        self.assertIn({"action": "edit", "resource": "*", "effect": "ask"}, mod.permission_rules("edit"))
        self.assertNotIn({"action": "edit", "resource": "*", "effect": "ask"}, rules)
        self.assertIn({"action": "external_directory", "resource": "*", "effect": "deny"}, rules)
        self.assertIn({"action": "*", "resource": "*", "effect": "deny"}, rules)
        self.assertNotIn({"action": "shell", "resource": "*", "effect": "ask"}, rules)
        edit = mod.make_config(18100, "local-qwen-coder", 8192, "edit")
        self.assertEqual(edit["default_agent"], "pac-edit")
        self.assertEqual(edit["agents"]["pac-edit"]["steps"], 5)
        self.assertNotIn({"action": "tool.use", "resource": "edit:*", "effect": "deny"}, edit["policies"])

    def test_opencode_install_preview(self):
        r = self.cmd("setup", "--components", "opencode")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Node.js", r.stdout)
        self.assertIn("OpenCode", r.stdout)
        self.assertIn("Preview only", r.stdout)
        self.assertFalse((self.root / "conda-envs").exists())

    def test_opencode_config_preview_without_server(self):
        p = self.root / "local_llm/opencode/npm/bin/opencode"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("#!/bin/sh\nexit 0\n")
        p.chmod(0o755)
        node = self.root / "conda-envs/opencode-node/bin/node"
        node.parent.mkdir(parents=True, exist_ok=True)
        node.write_text("#!/bin/sh\nexit 0\n")
        node.chmod(0o755)
        r = self.cmd("opencode", "--preview")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Preview only", r.stdout)
        config = json.loads((self.root / "local_llm/opencode/config/agent-read.json").read_text())
        self.assertEqual(config["model"], "pac/coder")
        self.assertEqual(config["permissions"][0]["effect"], "deny")

    def test_edit_mode_refuses_normal_checkout(self):
        p = self.root / "local_llm/opencode/npm/bin/opencode"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("#!/bin/sh\nexit 0\n")
        p.chmod(0o755)
        r = self.cmd("opencode", "--mode", "edit", "--preview")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Git check failed", r.stderr)

    @unittest.skipUnless(shutil_which := __import__("shutil").which("git"), "git unavailable")
    def test_edit_requires_linked_worktree_below_personal_storage(self):
        repo = self.root / "source"
        repo.mkdir()
        subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c",
                        "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
                        "commit", "--allow-empty", "-m", "root"], check=True, capture_output=True)
        class FakeWorkspace:
            base = self.root / "local_llm"
        with self.assertRaises(RuntimeError):
            mod.check_disposable_worktree(FakeWorkspace(), repo)
        work = FakeWorkspace.base / "opencode/worktrees/smoke"
        work.parent.mkdir(parents=True)
        subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach",
                        str(work), "HEAD"], check=True, capture_output=True)
        self.assertEqual(mod.check_disposable_worktree(FakeWorkspace(), work), work.resolve())

    def test_model_api_context(self):
        class FakeResponse(BytesIO):
            pass
        payload = {"data":[{"id":"a","max_model_len":8192},
                           {"id":"local-qwen-coder","max_model_len":16384}]}
        with patch.object(mod.urllib.request, "urlopen",
                          return_value=FakeResponse(json.dumps(payload).encode())):
            model, context = mod.probe_model(8000, "local-qwen-coder")
        self.assertEqual((model,context), ("local-qwen-coder",16384))


if __name__ == "__main__":
    unittest.main()
