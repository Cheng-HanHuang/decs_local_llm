"""CPU-only tests for the local workshop chat/tool-call smoke."""
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/pac_workshop_smoke.py"
SPEC = importlib.util.spec_from_file_location("pac_workshop_smoke", SCRIPT)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class WorkshopSmokeTests(unittest.TestCase):
    def test_chat_and_structured_tool_call(self):
        results = iter([
            {"data":[{"id":"local-qwen-coder"}]},
            {"choices":[{"message":{"content":"PAC-READY"}}]},
            {"choices":[{"message":{"tool_calls":[{
                "type":"function","function":{
                    "name":"add_two_integers", "arguments":json.dumps({"a":2,"b":3})}}]}}]},
        ])
        with patch.object(mod, "request_json", side_effect=lambda *a, **k: next(results)) as req:
            mod.smoke(18100, "local-qwen-coder", True)
        self.assertEqual(req.call_count,3)
        self.assertTrue(all("127.0.0.1:18100" in a.args[0] for a in req.call_args_list))

    def test_tool_missing_is_failure(self):
        results = iter([
            {"data":[{"id":"local-qwen-coder"}]},
            {"choices":[{"message":{"content":"PAC-READY"}}]},
            {"choices":[{"message":{"content":"5"}}]},
        ])
        with patch.object(mod, "request_json", side_effect=lambda *a, **k: next(results)):
            with self.assertRaisesRegex(RuntimeError, "No structured"):
                mod.smoke(18100, "local-qwen-coder", True)

    def test_model_mismatch_is_failure(self):
        with patch.object(mod,"request_json",return_value={"data":[{"id":"other-model"}]}):
            with self.assertRaisesRegex(RuntimeError,"Expected"):
                mod.smoke(18100, "local-qwen-coder", False)


if __name__ == "__main__":
    unittest.main()
