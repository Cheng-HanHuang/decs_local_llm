"""CPU-only tests for the standalone LoRA SFT demonstration."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SRC = Path(__file__).resolve().parents[1] / "experiments/lora_demo/demo.py"
SPEC = importlib.util.spec_from_file_location("lora_demo", SRC)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class ToyLoraTests(unittest.TestCase):
    def test_deterministic_disjoint_data(self):
        train1, test1 = module.records(seed=17)
        train2, test2 = module.records(seed=17)
        self.assertEqual(train1, train2)
        self.assertEqual(test1, test2)
        self.assertEqual(len(train1), 96)
        self.assertEqual(len(test1), 24)
        self.assertFalse({x["id"] for x in train1} & {x["id"] for x in test1})
        self.assertTrue(all(x["target"] == "PAC-ACK" for x in train1+test1))
        self.assertNotEqual(module.records(seed=18), (train1, test1))

    def test_prepare_reuses_same_data_but_rejects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            module.prepare(run, seed=17)
            manifest = json.loads((run / "data_manifest.json").read_text())
            self.assertEqual(manifest["train"], 96)
            self.assertEqual(manifest["test"], 24)
            self.assertEqual(len(module.jsonl_read(run / "test.jsonl")), 24)
            module.prepare(run, seed=17)
            with self.assertRaises(RuntimeError):
                module.prepare(run, seed=18)

    def test_inspect_requires_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                module.inspect(Path(tmp))


if __name__ == "__main__":
    unittest.main()
