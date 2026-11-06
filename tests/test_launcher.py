"""CPU-only integration tests; do not load any actual models or touch real GPUs."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'bin' / 'pac-llm'
FAKE_SERVER = '''#!/usr/bin/env python3
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
args=sys.argv[1:]
port=int(args[args.index('--port')+1])
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'OK')
    def log_message(self,*args): pass
HTTPServer(('127.0.0.1',port),H).serve_forever()
'''


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / 'local_llm'
        self.vllm = self.root / 'conda-envs/local-agent/bin'
        self.web = self.root / 'conda-envs/open-webui/bin'
        self.vllm.mkdir(parents=True)
        self.web.mkdir(parents=True)
        for name, path in [('vllm', self.vllm), ('open-webui', self.web)]:
            f = path / name
            f.write_text(FAKE_SERVER)
            f.chmod(0o755)
        (self.vllm / 'ninja').write_text('#!/bin/sh\nexit 0\n')
        (self.vllm / 'ninja').chmod(0o755)
        self.fake_bin = self.root / 'bin'
        self.fake_bin.mkdir()
        smi = self.fake_bin / 'nvidia-smi'
        smi.write_text('#!/bin/sh\necho "0, 97887, 84539, 99"\necho "1, 97887, 84000, 99"\n')
        smi.chmod(0o755)
        # Use unique high ports so tests are safe on shared CI hosts.
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0)); self.coder_port=s.getsockname()[1]
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0)); self.web_port=s.getsockname()[1]
        self.env = dict(os.environ, PAC_LLM_ROOT=str(self.root),
                        PAC_LLM_CODER_PORT=str(self.coder_port),
                        PAC_LLM_WEB_PORT=str(self.web_port),
                        PATH=str(self.fake_bin) + os.pathsep + os.environ.get('PATH',''))
        self.addCleanup(self.cleanup_processes)

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *args],
                              env=self.env, text=True, capture_output=True, timeout=10)

    def cleanup_processes(self):
        for name in ('coder','reasoner','web'):
            p = self.run_cli('stop',name,'--force')

    def test_doctor(self):
        p=self.run_cli('doctor')
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('GPU 0:',p.stdout)
        self.assertIn('Ninja:',p.stdout)

    def test_start_stop_and_status(self):
        p=self.run_cli('start','coder','--gpu','0')
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('Started coder:',p.stdout)
        state = list((self.base/'run').glob('*/coder.json'))
        self.assertEqual(len(state),1)
        data=json.loads(state[0].read_text())
        self.assertEqual(data['gpus'],[0])
        self.assertEqual(data['served_name'],'local-qwen-coder')
        self.assertEqual(data['model'],'Qwen/Qwen3-4B-Instruct-2507')
        self.assertEqual(data['tool_parser'],'hermes')
        self.assertEqual(data['memory_util'],0.16)
        self.assertEqual(data['max_model_len'],16384)
        for _ in range(40):
            p=self.run_cli('status')
            if 'api=ready' in p.stdout: break
            time.sleep(0.1)
        self.assertIn('coder    running',p.stdout)
        self.assertIn('api=ready',p.stdout)
        bad=self.run_cli('start','coder','--gpu','0')
        self.assertNotEqual(bad.returncode,0)
        self.assertIn('already managed',bad.stderr)
        done=self.run_cli('stop','coder')
        self.assertEqual(done.returncode,0,done.stderr)

    def test_custom_small_model_override(self):
        p = self.run_cli('start', 'coder', '--gpu', '0',
                         '--model', 'Qwen/Qwen3-0.6B',
                         '--served-name', 'small-base',
                         '--memory-util', '0.12',
                         '--max-model-len', '2048')
        self.assertEqual(p.returncode, 0, p.stderr)
        state = list((self.base / 'run').glob('*/coder.json'))
        data = json.loads(state[0].read_text())
        self.assertEqual(data['model'], 'Qwen/Qwen3-0.6B')
        self.assertEqual(data['tool_parser'], None)
        self.assertEqual(data['memory_util'], 0.12)
        self.assertEqual(data['max_model_len'], 2048)
        self.assertEqual(data['served_name'], 'small-base')

    def test_refuses_low_gpu_memory(self):
        smi = self.fake_bin/'nvidia-smi'
        # Below 0.16 * 97887 MiB + the 2 GiB guard for the 4B default.
        smi.write_text('#!/bin/sh\necho "0, 97887, 8000, 99"\n')
        p=self.run_cli('start','coder','--gpu','0')
        self.assertNotEqual(p.returncode,0)
        self.assertIn('headroom',p.stderr)

    def test_refuses_unmanaged_port(self):
        with socket.socket() as s:
            s.bind(('127.0.0.1', self.coder_port))
            s.listen()
            p=self.run_cli('start','coder','--gpu','0')
            self.assertNotEqual(p.returncode,0)
            self.assertIn('already occupied',p.stderr)

    def test_web_creates_private_data_and_secret(self):
        p=self.run_cli('start','web')
        self.assertEqual(p.returncode,0,p.stderr)
        secret=self.base/'webui/.secret_key'
        self.assertTrue(secret.exists())
        self.assertEqual(secret.stat().st_mode & 0o777,0o600)
        self.assertEqual((self.base/'webui/data').stat().st_mode & 0o777,0o700)
        self.assertEqual(self.run_cli('stop','web').returncode,0)

    def test_no_kill_unowned_process(self):
        run_dir=self.base/'run'/'test-host'
        run_dir.mkdir(parents=True)
        (run_dir/'coder.json').write_text('{"pid": 1, "starttime": "0", "owner": 0}')
        p=self.run_cli('stop','coder')
        self.assertNotEqual(p.returncode,0)
        self.assertIn('Refusing to kill',p.stderr)


if __name__ == '__main__':
    unittest.main()
