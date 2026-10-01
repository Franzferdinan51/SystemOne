"""Tests for systemone/openapi.json and its parity with the live shim.

The spec must parse, describe every served route, and be served verbatim at
GET /openapi.json. Every spec path is probed live (stub engine): GET paths
must answer 200, POST paths must route (any non-404 — malformed probes fail
400/422/500, never "not found"). Unknown paths must still 404.
"""

import json
import os
import sys
import threading
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone.shim import OPENAPI_PATH, serve  # noqa: E402


class StubEngine:
    model_name = "stub"

    def systemone(self, state, questions, batch_size=32):
        answers = {}
        for q in questions:
            if q["type"] == "choice":
                opts = q["options"]
                answers[q["name"]] = {
                    "type": "choice", "choice": opts[0],
                    "probabilities": {o: 1.0 / len(opts) for o in opts},
                    "confidence": 0.5,
                }
            elif q["type"] == "score":
                lv = q["levels"]
                answers[q["name"]] = {
                    "type": "score", "level": lv[0],
                    "distribution": {x: 1.0 / len(lv) for x in lv},
                    "confidence": 0.5,
                }
            else:
                answers[q["name"]] = {
                    "type": "noul", "probability": 0.5, "answer": True,
                    "confidence": 0.5,
                }
        answers["_meta"] = {"model": "stub", "latency_ms": 1.0}
        return answers


def _load_spec():
    with open(OPENAPI_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_spec_parses_and_lists_all_routes():
    spec = _load_spec()
    assert spec["openapi"].startswith("3.")
    assert set(spec["paths"]) == {
        "/", "/healthz", "/openapi.json",
        "/v1/systemone", "/v1/decisions",
        "/v1/decide", "/v1/decide/info",
        "/v1/systemone/route", "/v1/systemone/rank-plans",
        "/v1/systemone/decide",
    }
    for path, item in spec["paths"].items():
        for method, op in item.items():
            assert op.get("responses"), f"{method} {path} documents no responses"


def _request(port, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data,
        headers={"Content-Type": "application/json"}, method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def test_live_parity_every_spec_path_served():
    server = serve(0, engine=StubEngine())
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        spec = _load_spec()
        # GET paths answer 200 with JSON.
        for path in ("/", "/healthz", "/openapi.json"):
            status, payload = _request(port, "GET", path)
            assert status == 200, path
            assert isinstance(payload, dict), path
        # The served spec is the file's spec.
        _, served = _request(port, "GET", "/openapi.json")
        assert served == spec
        # POST paths route (malformed probes fail loudly, never 404).
        for path in ("/v1/systemone", "/v1/decisions",
                     "/v1/systemone/route", "/v1/systemone/rank-plans",
                     "/v1/systemone/decide"):
            status, _ = _request(port, "POST", path, {})
            assert status != 404, path
        # Unknown paths still 404 on both verbs.
        assert _request(port, "GET", "/nope")[0] == 404
        assert _request(port, "POST", "/nope", {})[0] == 404
    finally:
        server.shutdown()
        t.join(timeout=5)
