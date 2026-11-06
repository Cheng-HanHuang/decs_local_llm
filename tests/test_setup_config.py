"""Offline onboarding tests; never install dependencies or allocate GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "bin/pac-llm"

class SetupConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = dict(os.environ, PAC_LLM_ROOT=str(self.root))
        for key in ("PAC_LLM_GPU", "PAC_LLM_GPUS", "PAC_LLM_MODEL", "PAC_LLM_OFFLINE",
                    "PAC_LLM_CODER_PORT", "PAC_LLM_REASONER_PORT", "PAC_LLM_WEB_PORT"):
            self.env.pop(key, None)

    def cmd(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *args],
                              capture_output=True, text=True, timeout=15, env=self.env)

    def test_setup_is_preview_only(self):
        result = self.cmd("setup", "--components", "vllm,web,aider", "--download-coder")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Preview only", result.stdout)
        self.assertIn("Download model", result.stdout)
        self.assertIn("Qwen/Qwen3-4B-Instruct-2507", result.stdout)
        self.assertNotIn("Qwen3-Coder-30B-A3B-Instruct", result.stdout)
        self.assertFalse((self.root / "conda-envs").exists())

    def test_setup_rejects_invalid_component(self):
        result = self.cmd("setup", "--components", "web,unknown")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--components", result.stderr)

    def test_config_defaults_and_overrides(self):
        result = self.cmd("config", "init", "--gpu", "1", "--coder-port", "18100", "--web-port", "18180")
        self.assertEqual(result.returncode, 0, result.stderr)
        path = self.root / "local_llm/config/pac-llm.json"
        self.assertEqual(json.loads(path.read_text())["gpu"], "1")
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertIn("port=18100", self.cmd("status").stdout)
        self.env["PAC_LLM_CODER_PORT"] = "18300"
        self.assertIn("port=18300", self.cmd("status").stdout)
        bad = self.cmd("config", "set", "web_port", "99999")
        self.assertNotEqual(bad.returncode, 0)

    def test_existing_uv_aider_skips_unneeded_conda_env(self):
        aider = self.root / "local_llm/bin/aider"
        aider.parent.mkdir(parents=True, exist_ok=True)
        aider.write_text("#!/bin/sh\nexit 0\n")
        aider.chmod(0o755)
        wrapper = self.root / "local_llm/run_aider_local.sh"
        wrapper.write_text("#!/bin/sh\nexit 0\n")
        r = self.cmd("setup", "--components", "aider")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("No missing executables detected", r.stdout)
        self.assertFalse((self.root / "conda-envs/aider").exists())

    def test_aider_uses_saved_port_not_legacy_wrapper(self):
        r = self.cmd("config", "init", "--coder-port", "18100")
        self.assertEqual(r.returncode, 0, r.stderr)
        aider = self.root / "local_llm/bin/aider"
        aider.parent.mkdir(parents=True, exist_ok=True)
        aider.write_text('#!/bin/sh\nprintf "BASE=%s\\nHOME=%s\\nARGS=%s\\n" "$OPENAI_API_BASE" "$HOME" "$*"\n')
        aider.chmod(0o755)
        wrapper = self.root / "local_llm/run_aider_local.sh"
        wrapper.write_text("#!/bin/sh\nexit 31\n")
        r = self.cmd("aider")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("BASE=http://127.0.0.1:18100/v1", r.stdout)
        self.assertIn("--chat-mode ask", r.stdout)
        self.assertIn("--no-gitignore", r.stdout)
        self.assertIn("--no-show-model-warnings", r.stdout)
        self.assertIn(str(self.root / "local_llm/aider_home"), r.stdout)

    def test_partial_webui_env_gets_missing_cpu_torch(self):
        # Interrupted installs can leave Python but not torch or WebUI.
        env = self.root / "conda-envs/open-webui"
        (env / "bin").mkdir(parents=True)
        (env / "conda-meta").mkdir()
        python = env / "bin/python"
        python.write_text("#!/bin/sh\\nexit 1\\n".replace("\\n", "\n"))
        python.chmod(0o755)
        result = self.cmd("setup", "--components", "web")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Install CPU PyTorch", result.stdout)
        self.assertIn("Install pinned Open WebUI", result.stdout)
        self.assertNotIn("Create/resume WebUI environment", result.stdout)

    def test_partial_vllm_env_uses_conda_install_not_create(self):
        env = self.root / "conda-envs/local-agent"
        (env / "conda-meta").mkdir(parents=True)
        result = self.cmd("setup", "--components", "vllm")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Create/resume vLLM environment", result.stdout)
        self.assertIn("install -p", result.stdout)
        self.assertNotIn("create -p", result.stdout)

    def test_config_init_does_not_overwrite(self):
        self.assertEqual(self.cmd("config", "init").returncode, 0)
        self.assertNotEqual(self.cmd("config", "init").returncode, 0)

    def test_config_set_gpus_invalid_returns_nonzero_exit_code(self):
        # Duplicate GPU indices must fail validation, not just generic parsing.
        result = self.cmd("config", "set", "gpus", "2,2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("distinct", result.stderr)

if __name__ == "__main__":
    unittest.main()
