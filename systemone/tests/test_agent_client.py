"""Tests for the agent layer: shim client, CLI, and MCP shim tools.

All HTTP is mocked — no shim, no model, no network.
"""

import io
import json
import os
import sys
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone import cli
from systemone.client import (
    ShimError,
    SystemOneClient,
    default_shim_url,
)


def _mcp():
    """systemone.mcp_server, skipping tests when its heavy deps are absent."""
    try:
        import systemone.mcp_server as mod

        return mod
    except ImportError:
        pytest.skip("needs mcp<2 installed")


BASE = "http://shim.test"


# -- fake urllib --------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _install_urlopen(monkeypatch, routes):
    """routes: {(method, path): payload | Exception}."""
    seen = []

    def fake_urlopen(req, timeout=None):
        url = req.get_full_url()
        path = url[len(BASE):] or "/"
        body = req.data.decode("utf-8") if req.data else None
        seen.append((req.get_method(), path, json.loads(body) if body else None))
        route = routes.get((req.get_method(), path))
        if isinstance(route, Exception):
            raise route
        if route is None:
            raise AssertionError(f"unexpected request {(req.get_method(), path)}")
        return _FakeResponse(route)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return seen


def _http_error(code, payload):
    return urllib.error.HTTPError(
        BASE + "/x", code, "err", {}, io.BytesIO(json.dumps(payload).encode())
    )


# -- client -------------------------------------------------------------


def test_default_shim_url_env_override(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_SHIM_URL", "http://macmini:8765/")
    assert default_shim_url() == "http://macmini:8765/"
    assert SystemOneClient().base_url == "http://macmini:8765"


def test_default_shim_url_fallback(monkeypatch):
    monkeypatch.delenv("SYSTEMONE_SHIM_URL", raising=False)
    assert default_shim_url() == "http://127.0.0.1:8765"


def test_client_route(monkeypatch):
    routes = {
        ("POST", "/v1/systemone/route"): {
            "route": {"tier": "balanced", "confidence": 0.72},
            "model": "eng",
            "usage": {},
        }
    }
    seen = _install_urlopen(monkeypatch, routes)
    out = SystemOneClient(BASE).route("do the thing", cost_bias="economy")
    assert out["route"]["tier"] == "balanced"
    method, path, body = seen[0]
    assert (method, path) == ("POST", "/v1/systemone/route")
    assert body == {"task": "do the thing", "cost_bias": "economy"}


def test_client_decide(monkeypatch):
    decision = {
        "type": "noul", "label": "yes",
        "probabilities": {"yes": 0.8, "no": 0.2},
        "confidence": 0.8, "latency_ms": 3.1, "backend": "decider",
    }
    _install_urlopen(monkeypatch, {("POST", "/v1/systemone/decide"): decision})
    out = SystemOneClient(BASE).decide("s", "is it?", type="noul")
    assert out["backend"] == "decider"
    assert out["probabilities"]["yes"] == 0.8


def test_client_status_ok(monkeypatch):
    routes = {
        ("GET", "/healthz"): {"ok": True, "model": "fake-engine"},
        ("POST", "/v1/systemone/decide"): {
            "type": "noul", "label": "yes",
            "probabilities": {"yes": 0.9, "no": 0.1},
            "confidence": 0.9, "latency_ms": 2.0, "backend": "fallback",
        },
    }
    _install_urlopen(monkeypatch, routes)
    out = SystemOneClient(BASE).status()
    assert out["shim"]["ok"] is True
    assert out["decision"]["backend"] == "fallback"


def test_client_status_no_probe(monkeypatch):
    _install_urlopen(monkeypatch, {("GET", "/healthz"): {"ok": True}})
    out = SystemOneClient(BASE).status(probe=False)
    assert out["decision"] is None


def test_client_status_shim_down(monkeypatch):
    routes = {("GET", "/healthz"): urllib.error.URLError("refused")}
    _install_urlopen(monkeypatch, routes)
    out = SystemOneClient(BASE).status()
    assert out["shim"]["ok"] is False
    assert "refused" in out["shim"]["error"]
    assert out["decision"] is None


def test_client_http_error_maps_to_shim_error(monkeypatch):
    routes = {("POST", "/v1/systemone/route"): _http_error(400, {"error": "bad task"})}
    _install_urlopen(monkeypatch, routes)
    with pytest.raises(ShimError) as excinfo:
        SystemOneClient(BASE).route("")
    assert excinfo.value.status == 400
    assert "bad task" in str(excinfo.value)


def test_client_unreachable_maps_to_shim_error(monkeypatch):
    routes = {("GET", "/healthz"): urllib.error.URLError("refused")}
    _install_urlopen(monkeypatch, routes)
    with pytest.raises(ShimError) as excinfo:
        SystemOneClient(BASE).health()
    assert "cannot reach shim" in str(excinfo.value)


def test_client_raw_timeout_and_oserror_map_to_shim_error(monkeypatch):
    for boom in (TimeoutError("timed out"), OSError("down")):
        routes = {("GET", "/healthz"): boom}
        _install_urlopen(monkeypatch, routes)
        with pytest.raises(ShimError) as excinfo:
            SystemOneClient(BASE).health()
        assert "cannot reach shim" in str(excinfo.value)


# -- CLI -----------------------------------------------------------------


class _FakeClient:
    """Stand-in for SystemOneClient; records calls, returns canned payloads."""

    last = None

    def __init__(self, base_url=None, timeout=120.0):
        self.base_url = base_url or default_shim_url()
        self.calls = []
        _FakeClient.last = self

    def route(self, task, cost_bias="balanced", tiers=None):
        self.calls.append(("route", task, cost_bias, tiers))
        return {
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
        }

    def decide(self, state, instructions, criteria=None, type="choice"):
        self.calls.append(("decide", state, instructions, criteria, type))
        if type == "score":
            return {
                "type": "score", "level": "1",
                "distribution": {"0": 0.1, "1": 0.7, "2": 0.2},
                "confidence": 0.7, "latency_ms": 4.2, "backend": "decider",
            }
        if type == "noul":
            return {
                "type": "noul", "label": "yes",
                "probabilities": {"yes": 0.82, "no": 0.18},
                "confidence": 0.82, "latency_ms": 4.2, "backend": "decider",
            }
        return {
            "type": "choice", "label": "a",
            "probabilities": {"a": 0.75, "b": 0.25},
            "confidence": 0.75, "latency_ms": 4.2, "backend": "decider",
        }

    def status(self, probe=True):
        self.calls.append(("status", probe))
        return {
            "shim_url": self.base_url,
            "shim": {"ok": True, "model": "fake-engine"},
            "decision": {"backend": "decider", "latency_ms": 4.2,
                         "confidence": 0.9} if probe else None,
        }


@pytest.fixture
def fake_client(monkeypatch):
    monkeypatch.setattr(cli, "SystemOneClient", _FakeClient)
    return _FakeClient


def test_cli_route_human(fake_client, capsys):
    assert cli.main(["route", "summarize revenue"]) == 0
    out = capsys.readouterr().out
    assert "tier       : balanced" in out
    assert "confidence : 0.7234  (margin 0.1811)" in out
    assert "model      : fake-model" in out
    assert _FakeClient.last.calls[0] == ("route", "summarize revenue",
                                         "balanced", None)


def test_cli_route_json_and_flags(fake_client, capsys):
    assert cli.main(["route", "--cost-bias", "quality", "--tiers",
                     "economy,balanced", "--json", "t"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tier"] == "balanced"
    assert _FakeClient.last.calls[0] == ("route", "t", "quality",
                                         ["economy", "balanced"])


def test_cli_decide_choice(fake_client, capsys):
    argv = ["decide", "--type", "choice", "--state", "s",
            "--instructions", "pick", "--criteria", "a=first",
            "--criteria", "b=second"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "choice     : a  (confidence 0.7500)" in out
    assert "backend    : decider" in out
    _, _, _, criteria, dtype = _FakeClient.last.calls[0]
    assert criteria == {"a": "first", "b": "second"}
    assert dtype == "choice"


def test_cli_decide_score_json_criteria(fake_client, capsys):
    argv = ["decide", "--type", "score", "--state", "s",
            "--instructions", "rate",
            "--criteria-json", '["low", "medium", "high"]', "--json"]
    assert cli.main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["level"] == "1"
    assert payload["distribution"]["1"] == 0.7


def test_cli_decide_noul_human(fake_client, capsys):
    argv = ["decide", "--type", "noul", "--state", "s",
            "--instructions", "is it?"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "answer     : yes  (P(yes)=0.8200, confidence 0.8200)" in out


def test_cli_status_human(fake_client, capsys):
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "ok       : yes" in out
    assert "backend=decider" in out


def test_cli_status_json(fake_client, capsys):
    assert cli.main(["status", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["shim"]["ok"] is True
    assert payload["decision"]["backend"] == "decider"


def test_cli_shim_error_exit_code(monkeypatch, capsys):
    class _Boom:
        def __init__(self, base_url=None, timeout=120.0):
            pass

        def route(self, *a, **k):
            raise ShimError("cannot reach shim at http://x/v1/systemone/route: refused")

    monkeypatch.setattr(cli, "SystemOneClient", _Boom)
    assert cli.main(["route", "t"]) == 1
    assert "cannot reach shim" in capsys.readouterr().err


def test_cli_help_exits_zero(capsys):
    for argv in (["--help"], ["route", "--help"], ["decide", "--help"],
                 ["status", "--help"], ["battery", "--help"]):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(argv)
        assert excinfo.value.code == 0


def test_cli_battery_passthrough(monkeypatch):
    recorded = {}

    def fake_main():
        recorded["argv"] = list(sys.argv)
        return 0

    import systemone.battery.run as battery_run

    monkeypatch.setattr(battery_run, "main", fake_main)
    assert cli.main(["battery", "--mode", "live", "--no-update-last-run"]) == 0
    assert recorded["argv"] == ["systemone battery", "--mode", "live",
                                 "--no-update-last-run"]


# -- MCP shim tools -------------------------------------------------------


@pytest.fixture
def fake_mcp_client(monkeypatch):
    monkeypatch.setattr(_mcp(), "get_client", lambda: _FakeClient())
    return _FakeClient


def test_mcp_route_tool(fake_mcp_client):
    out = _mcp()._route_impl("do things", cost_bias="economy")
    assert out["tier"] == "balanced"
    assert out["confidence"] == 0.7234


def test_mcp_decide_tool(fake_mcp_client):
    out = _mcp()._decide_impl("s", "pick", criteria={"a": None, "b": None})
    assert out["label"] == "a"
    assert out["backend"] == "decider"


def test_mcp_status_tool(fake_mcp_client):
    out = _mcp()._status_impl()
    assert out["shim"]["ok"] is True
    assert out["decision"]["backend"] == "decider"


def test_mcp_rank_plans_tool(monkeypatch):
    class _RankFake(_FakeClient):
        def rank_plans(self, task, plans):
            return {"task": task, "tier": "balanced",
                    "ranking": [{"id": "p1", "score": 0.9}],
                    "model": "fake-engine", "usage": {}}

    monkeypatch.setattr(_mcp(), "get_client", lambda: _RankFake())
    out = _mcp()._rank_plans_impl("t", [{"id": "p1", "text": "plan"}])
    assert out["ranking"][0]["id"] == "p1"


def test_mcp_tool_error_shape(monkeypatch):
    class _ErrClient:
        def route(self, *a, **k):
            raise ShimError("boom")

    monkeypatch.setattr(_mcp(), "get_client", lambda: _ErrClient())
    out = _mcp()._route_impl("t")
    assert out == {"error": "boom"}
