"""CPU-only unit tests for the per-user setup autopilot (no network or GPU)."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts/lab_autopilot.py"
SPEC = importlib.util.spec_from_file_location("lab_autopilot", SOURCE)
mod = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)
REPO = SOURCE.parents[1]


class LabAutopilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "local_llm/bin").mkdir(parents=True)

    def test_plan_has_no_30b_or_gpu_jobs(self):
        stages = mod.stages(self.root, REPO)
        self.assertEqual(len(stages), 12)
        self.assertEqual(
            [x.name for x in stages],
            ["cpu_tests", "vllm", "webui", "aider", "opencode", "lora_env",
             "lora_torch", "lora_libs", "coder_weights", "sft_weights",
             "toy_data", "verify"],
        )
        self.assertIn(mod.CODER_MODEL, stages[8].command)
        self.assertIn(mod.SFT_MODEL, stages[9].command)
        self.assertNotIn("30B", str(stages))
        self.assertNotIn("vllm serve", str(stages))
        self.assertNotIn("train --run-dir", str(stages))

    def test_clean_env_has_private_cache_and_ignores_hf_tokens(self):
        with patch.dict(os.environ, {
            "HF_TOKEN": "example-secret",
            "HUGGING_FACE_HUB_TOKEN": "example-other-secret",
            "HF_HUB_OFFLINE": "1",
        }):
            env = mod.clean_env(self.root)
        self.assertNotIn("HF_TOKEN", env)
        self.assertNotIn("HUGGING_FACE_HUB_TOKEN", env)
        self.assertEqual(env["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "1")
        self.assertEqual(env["HF_HUB_OFFLINE"], "0")
        self.assertEqual(env["HF_HOME"], str(self.root / "local_llm/hf_cache"))
        self.assertTrue(env["PATH"].startswith(
            str(self.root / "conda-envs/local-agent/bin") + os.pathsep))

    def test_status_atomic_and_private(self):
        state = {"status": "failed", "stage": "aider",
                 "completed": ["cpu_tests", "vllm"], "source_sha": "abc"}
        mod.save_status(self.root, state)
        self.assertEqual(mod.read_status(self.root), state)
        self.assertEqual(mod.status_path(self.root).stat().st_mode & 0o777, 0o600)

    def test_symlink_created_and_wrong_symlink_rejected(self):
        link = self.root / "local_llm/bin/pac-llm"
        mod.ensure_launcher_link(self.root, REPO)
        self.assertEqual(link.resolve(), (REPO / "bin/pac-llm").resolve())
        link.unlink()
        fake = self.root / "other"
        fake.write_text("not the launcher")
        link.symlink_to(fake)
        with self.assertRaisesRegex(RuntimeError, "Do not overwrite"):
            mod.ensure_launcher_link(self.root, REPO)


if __name__ == "__main__":
    unittest.main()
