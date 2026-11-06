"""CPU-only tests: environment receipts do not install or mutate anything."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts/record_pac_environment.py"
SPEC = importlib.util.spec_from_file_location("record_env", SOURCE)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class EnvironmentReceiptTests(unittest.TestCase):
    def test_missing_optional_environments(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(mod, "cmd", return_value=None):
                data = mod.receipt(root, root)
            self.assertEqual(data["source_commit"], None)
            self.assertTrue(all(x["installed"] is False for x in data["environments"].values()))
            self.assertIn("No credentials", data["note"])

    def test_packages_for_missing_python(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(mod.packages_for(Path(temp), ["torch"]), {"installed": False})


if __name__ == "__main__":
    unittest.main()
