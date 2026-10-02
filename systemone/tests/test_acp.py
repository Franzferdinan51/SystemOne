"""ACP conformance smoke test: drive systemone.acp_server over real stdio.

Spins up a fake shim (real HTTP on 127.0.0.1), launches the ACP agent as a
subprocess, and walks the ACP v1 handshake: initialize -> session/new ->
session/prompt (/route, /decide, plain text) -> error paths -> cancel -> EOF.
"""

import json
import os
import queue
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# -- fake shim -----------------------------------------------------------


class _FakeShimHandler(BaseHTTPRequestHandler):
    def _send(self, payload, code=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        if self.path == "/healthz":
            self._send({"ok": True, "model": "fake-engine"})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        body = self._read_json()
        if self.path == "/v1/systemone/route":
            self._send({
                "route": {
                    "tier": "balanced",
                    "confidence": 0.7234,
                    "margin": 0.1811,
                    "model_id": "fake-model",
                    "effort": "medium",
                    "uncertain": False,
                    "rationale": "fake rationale",
                },
                "model": "fake-engine",
                "usage": {},
            })
        elif self.path == "/v1/systemone/decide":
            dtype = body.get("type")
            criteria = body.get("criteria")
            if dtype == "choice":
                labels = list(criteria or {"a": None, "b": None})
                self._send({
                    "type": "choice", "label": labels[0],
                    "probabilities": {labels[0]: 0.77, labels[1]: 0.23}
                    if len(labels) > 1 else {labels[0]: 1.0},
                    "confidence": 0.77, "latency_ms": 1.5, "backend": "decider",
                })
            elif dtype == "score":
                self._send({
                    "type": "score", "level": "1",
                    "distribution": {"0": 0.2, "1": 0.6, "2": 0.2},
                    "confidence": 0.6, "latency_ms": 1.5, "backend": "decider",
                })
            else:
                self._send({
                    "type": "noul", "label": "yes",
                    "probabilities": {"yes": 0.83, "no": 0.17},
                    "confidence": 0.83, "latency_ms": 1.5, "backend": "decider",
                })
        else:
            self._send({"error": "not found"}, 404)

    def log_message(self, *args):
        pass


def _start_fake_shim():
    server = HTTPServer(("127.0.0.1", 0), _FakeShimHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# -- stdio driver ---------------------------------------------------------


class _Driver:
    def __init__(self, proc):
        self.proc = proc
        self.inbox: "queue.Queue[str]" = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self):
        for line in self.proc.stdout:
            self.inbox.put(line)

    def send(self, message):
        if isinstance(message, str):
            self.proc.stdin.write(message + "\n")
        else:
            self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def wait_for_id(self, msg_id, timeout=60):
        """Drain notifications until the response with msg_id arrives."""
        notifications = []
        deadline = timeout
        import time

        end = time.time() + deadline
        while time.time() < end:
            try:
                line = self.inbox.get(timeout=max(0.1, end - time.time()))
            except queue.Empty:
                break
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if isinstance(msg, dict) and msg.get("id") == msg_id and "method" not in msg:
                return msg, notifications
            notifications.append(msg)
        raise AssertionError(f"no response with id={msg_id!r} within {timeout}s")

    def rpc(self, msg_id, method, params=None, timeout=60):
        self.send({"jsonrpc": "2.0", "id": msg_id, "method": method,
                   "params": params or {}})
        return self.wait_for_id(msg_id, timeout)


def _updates_of(notifications, kind):
    return [n["params"]["update"] for n in notifications
            if n.get("method") == "session/update"
            and n.get("params", {}).get("update", {}).get("sessionUpdate") == kind]


def test_acp_conformance():
    shim = _start_fake_shim()
    port = shim.server_address[1]
    env = dict(os.environ)
    env["SYSTEMONE_SHIM_URL"] = f"http://127.0.0.1:{port}"
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        [sys.executable, "-m", "systemone.acp_server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True, cwd=REPO_ROOT, env=env,
    )
    driver = _Driver(proc)
    try:
        # 1. initialize
        resp, _ = driver.rpc(0, "initialize", {
            "protocolVersion": 1,
            "clientCapabilities": {},
            "clientInfo": {"name": "acp-smoke", "version": "0.0.0"},
        })
        result = resp["result"]
        assert result["protocolVersion"] == 1
        assert result["agentInfo"]["name"] == "systemone"
        assert result["authMethods"] == []

        # 2. session/new
        resp, notes = driver.rpc(1, "session/new", {"cwd": REPO_ROOT})
        session_id = resp["result"]["sessionId"]
        assert session_id.startswith("sess_")
        chunks = _updates_of(notes, "agent_message_chunk")
        assert chunks and "/route" in chunks[0]["content"]["text"]

        # 3. session/prompt -> /route
        resp, notes = driver.rpc(2, "session/prompt", {
            "sessionId": session_id,
            "prompt": [{"type": "text",
                        "text": "/route --cost-bias quality summarize revenue"}],
        })
        assert resp["result"]["stopReason"] == "end_turn"
        assert _updates_of(notes, "tool_call"), "expected a tool_call update"
        completed = _updates_of(notes, "tool_call_update")
        assert any(u.get("status") == "completed" for u in completed)
        chunks = _updates_of(notes, "agent_message_chunk")
        assert any("balanced" in c["content"]["text"] for c in chunks)

        # 4. session/prompt -> /decide (choice via fenced JSON block)
        decide_text = ("/decide\n```json\n" + json.dumps({
            "type": "choice",
            "state": "the deploy is green",
            "instructions": "should we ship?",
            "criteria": {"ship": "deploy now", "hold": "wait"},
        }) + "\n```")
        resp, notes = driver.rpc(3, "session/prompt", {
            "sessionId": session_id,
            "prompt": [{"type": "text", "text": decide_text}],
        })
        assert resp["result"]["stopReason"] == "end_turn"
        chunks = _updates_of(notes, "agent_message_chunk")
        assert any("ship" in c["content"]["text"] for c in chunks)

        # 5. plain text -> usage guidance, still end_turn
        resp, notes = driver.rpc(4, "session/prompt", {
            "sessionId": session_id,
            "prompt": [{"type": "text", "text": "hello"}],
        })
        assert resp["result"]["stopReason"] == "end_turn"
        chunks = _updates_of(notes, "agent_message_chunk")
        assert any("/decide" in c["content"]["text"] for c in chunks)

        # 6. unknown method -> -32601
        resp, _ = driver.rpc(5, "session/foobar", {})
        assert resp["error"]["code"] == -32601

        # 7. malformed JSON -> -32700 with null id
        driver.send("this is not json")
        resp, _ = driver.wait_for_id(None)
        assert resp["error"]["code"] == -32700

        # 8. session/cancel is a notification: no response, no crash
        driver.send({"jsonrpc": "2.0", "method": "session/cancel",
                     "params": {"sessionId": session_id}})
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait(timeout=30)
        shim.shutdown()
    assert proc.returncode == 0, f"acp server exited {proc.returncode}"
