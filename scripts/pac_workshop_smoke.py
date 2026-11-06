#!/usr/bin/env python3
"""Read-only local vLLM smoke check for a lab workshop.

Calls localhost only, executes NO model-requested tools and reads NO private files.
Requires the vLLM server to be running before this script is invoked.
"""
from __future__ import annotations
import argparse
import json
import sys
import urllib.error
import urllib.request


def request_json(url, payload=None, timeout=60):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    method = "GET" if data is None else "POST"
    headers = {} if data is None else {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.load(res)


def smoke(port, model, test_tools):
    base = f"http://127.0.0.1:{port}/v1"
    available = [x["id"] for x in request_json(f"{base}/models")["data"]]
    print("Server models:", available)
    if model not in available:
        raise RuntimeError(f"Expected {model!r}; actual models: {available!r}")
    response = request_json(f"{base}/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: PAC-READY"}],
        "temperature": 0, "max_tokens": 32,
    })
    msg = response["choices"][0]["message"]
    print("Chat reply:", repr(msg.get("content")))
    if (msg.get("content") or "").strip() != "PAC-READY":
        print("WARNING: chat output differs; model may need more prompt guidance.")
    if not test_tools:
        return
    function = {
        "type": "function",
        "function": {
            "name": "add_two_integers",
            "description": "Calculate the sum of two provided integers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"], "additionalProperties": False,
            },
        },
    }
    response = request_json(f"{base}/chat/completions", {
        "model": model, "messages": [
            {"role": "user", "content":
             "Call add_two_integers with a=2 and b=3. Use the tool; do not answer directly."}],
        "tools": [function],
        "tool_choice": "auto",
        "temperature": 0, "max_tokens": 128,
    })
    msg = response["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    print("Structured tool calls:", json.dumps(calls, indent=2))
    if not calls:
        raise RuntimeError("No structured tool call returned. Check parser/model compatibility.")
    call = calls[0].get("function", {})
    args = json.loads(call.get("arguments", "{}"))
    if call.get("name") != "add_two_integers" or args != {"a": 2, "b": 3}:
        raise RuntimeError(f"Unexpected call: {call!r}")
    print("Tool parsing: PASS (no tool was executed)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", default="local-qwen-coder")
    p.add_argument("--test-tools", action="store_true")
    args = p.parse_args()
    if not 1024 <= args.port <= 65535:
        p.error("port must be 1024..65535")
    try:
        smoke(args.port, args.model, args.test_tools)
    except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError, RuntimeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
