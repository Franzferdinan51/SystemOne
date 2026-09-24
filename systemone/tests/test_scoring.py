"""Tests for the SystemOne scoring & ranking surface.

Covers scoring.py (calibration, model/tool/plan ranking, LM Studio
inventory) plus the shim wiring: route_decision scoring block,
POST /v1/systemone/rank-plans, and the SYSTEMONE_DISABLE kill switch.

No model needed — all engine calls use stubs.
"""

import json
import os
import sys
import threading
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import systemone.scoring as scoring
from systemone import shim
from systemone.scoring import (
    apply_calibration,
    apply_inventory,
    calibrate_probs,
    estimate_steps,
    fetch_lmstudio_models,
    load_tool_registry,
    rank_models,
    rank_plans,
    reset_priors,
    reset_tuning,
    score_tools,
    tool_kw_weight,
    tool_model_weight,
    tool_veto_threshold,
    top2_margin,
)
from systemone.shim import route_decision, serve


class ScoringStubEngine:
    """Deterministic stub: choice questions weight the first option most;
    noul questions answer P(yes) from the prob map; raw_scores returns
    tool_scores for task texts and prior_scores for calibration probes."""

    model_name = "stub"

    def __init__(self, noul_probs=None, fail_noul=False,
                 tool_scores=None, prior_scores=None, fail_tools=False):
        self.noul_probs = noul_probs or {}
        self.fail_noul = fail_noul
        self.tool_scores = tool_scores or {}
        self.prior_scores = prior_scores if prior_scores is not None else {}
        self.fail_tools = fail_tools
        # import here to avoid circulars at module load
        from systemone.scoring import _PRIOR_PROBES
        self._probes = set(_PRIOR_PROBES)

    def raw_scores(self, texts, label_lists, prompts=None, batch_size=32,
                   classification_type="single_label"):
        """Contrastive stub: per-option scores from tool_scores by tool id;
        prior_scores when the text is a calibration probe."""
        if self.fail_tools:
            raise RuntimeError("engine down")
        out = []
        for text, labs in zip(texts, label_lists):
            table = self.prior_scores if text in self._probes else self.tool_scores
            d = {}
            for lab in labs:
                tid = lab.split(":", 1)[0].strip()
                d[lab] = float(table.get(tid, 0.5))
            out.append(d)
        return out

    def systemone(self, state, questions, batch_size=32):
        answers = {}
        for q in questions:
            if q.get("type") == "noul":
                if self.fail_noul:
                    raise RuntimeError("engine down")
                p = self.noul_probs.get(q["name"], 0.5)
                answers[q["name"]] = {
                    "type": "noul",
                    "choice": "yes" if p >= 0.5 else "no",
                    "probability": p,
                    "probabilities": {"yes": p, "no": 1.0 - p},
                    "confidence": max(p, 1.0 - p),
                }
            else:
                opts = q["options"]
                n = len(opts)
                raw = [n - i for i in range(n)]
                total = sum(raw)
                probs = {o: r / total for o, r in zip(opts, raw)}
                answers[q["name"]] = {
                    "type": "choice", "choice": opts[0],
                    "probabilities": probs, "confidence": probs[opts[0]],
                }
        answers["_meta"] = {"model": "stub"}
        return answers


def _cands():
    return [
        {"tier": "economy", "model_id": "e", "description": "tiny"},
        {"tier": "balanced", "model_id": "b", "description": "medium"},
        {"tier": "heavy", "model_id": "h", "description": "large"},
    ]


def _registry():
    return {
        "economy": {"model_id": "e", "models": [
            {"model_id": "e", "quality": {"economy": 0.7, "balanced": 0.3,
                                          "heavy": 0.2},
             "cost": 0.3, "available": True},
        ]},
        "balanced": {"model_id": "b", "models": [
            {"model_id": "b", "quality": {"economy": 0.6, "balanced": 0.8,
                                          "heavy": 0.7},
             "cost": 1.0, "available": True},
            {"model_id": "b2", "quality": {"balanced": 0.6, "economy": 0.5},
             "cost": 0.6, "available": False},  # not loadable: excluded
        ]},
        "heavy": {"model_id": "h", "models": [
            {"model_id": "h", "quality": {"economy": 0.5, "balanced": 0.65,
                                          "heavy": 0.9},
             "cost": 3.0, "available": True},
        ]},
    }


def _scoring_ctx(T=1.0, tools=None):
    return {
        "calibration": {"temperature": T, "battery_size": 250},
        "tools": tools or [],
        "registry": _registry(),
    }


def _tools():
    return [
        {"id": "webfetch", "kind": "tool", "description": "fetch a URL"},
        {"id": "browserclaw", "kind": "mcp", "description": "browser automation"},
        {"id": "bash", "kind": "tool", "description": "run shell commands"},
    ]


def _medium_task():
    # medium deterministic signals ("explain"), no heavy keywords
    return ("Explain in detail the differences between TCP and UDP, including "
            "reliability, ordering, connection setup, congestion control, and "
            "typical use cases, with concrete examples of when each protocol "
            "is the right choice.")


# -- calibration ------------------------------------------------------------

def test_calibrate_identity_at_T1():
    probs = {"economy": 0.7, "balanced": 0.2, "heavy": 0.1}
    assert calibrate_probs(probs, 1.0) == probs


def test_calibrate_softens_overconfident():
    probs = {"economy": 0.9, "balanced": 0.05, "heavy": 0.05}
    cal = calibrate_probs(probs, 2.0)
    assert abs(sum(cal.values()) - 1.0) < 1e-9
    assert cal["economy"] < 0.9  # softened


def test_top2_margin():
    assert top2_margin({"a": 0.6, "b": 0.3, "c": 0.1}) == pytest.approx(0.3)


def test_apply_calibration_without_file():
    cal = apply_calibration({"a": 0.6, "b": 0.4}, None)
    assert cal["calibrated"] is False
    assert cal["uncertain"] is False
    assert cal["confidence"] == pytest.approx(0.6)
    assert cal["margin"] == pytest.approx(0.2)


def test_apply_calibration_uncertain_when_margin_below_floor(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_MARGIN_FLOOR", "0.15")
    # Very hot temperature -> nearly uniform -> margin ~ 0
    cal = apply_calibration(
        {"a": 0.6, "b": 0.3, "c": 0.1}, {"temperature": 100.0})
    assert cal["calibrated"] is True
    assert cal["margin"] < 0.15
    assert cal["uncertain"] is True


def test_apply_calibration_confident_case(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_MARGIN_FLOOR", "0.15")
    cal = apply_calibration(
        {"a": 0.8, "b": 0.15, "c": 0.05}, {"temperature": 1.0})
    assert cal["uncertain"] is False
    assert cal["confidence"] == pytest.approx(0.8)


# -- model ranking ------------------------------------------------------------

def test_rank_models_expected_utility_order():
    probs = {"economy": 0.8, "balanced": 0.15, "heavy": 0.05}
    ranked = rank_models(_registry(), probs, lam=0.15, topn=3)
    ids = [r["model_id"] for r in ranked]
    assert "b2" not in ids  # unavailable excluded
    # "e" should win: economy-heavy mix, cheap
    assert ids[0] == "e"
    assert all("utility" in r and "quality" in r for r in ranked)
    assert len(ranked) <= 3


def test_rank_models_quality_can_beat_cost():
    probs = {"economy": 0.05, "balanced": 0.15, "heavy": 0.8}
    ranked = rank_models(_registry(), probs, lam=0.15, topn=3)
    assert ranked[0]["model_id"] == "b"  # best heavy quality available


def test_rank_models_fail_open_on_empty_registry():
    assert rank_models({}, {"economy": 1.0}) == []


def test_rank_models_skips_models_missing_quality_or_cost():
    reg = {
        "economy": {"models": [
            {"model_id": "good", "quality": {"economy": 0.9},
             "cost": 0.5, "available": True},
            {"model_id": "noquality", "cost": 0.1, "available": True},
            {"model_id": "nocost", "quality": {"economy": 0.9},
             "available": True},
        ]},
    }
    ranked = rank_models(reg, {"economy": 1.0}, lam=0.0)
    assert [r["model_id"] for r in ranked] == ["good"]


def test_model_top_n_env(monkeypatch):
    from systemone.scoring import model_top_n
    monkeypatch.setenv("SYSTEMONE_MODEL_TOPN", "2")
    assert model_top_n() == 2


# -- tool ranking (hybrid: keyword recall + model precision) -------------------

@pytest.fixture(autouse=True)
def _clean_caches():
    reset_priors()
    reset_tuning()
    yield
    reset_priors()
    reset_tuning()


def test_score_tools_hybrid_ranks_keyword_match_first(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_TOOL_KW_WEIGHT", "0.6")
    monkeypatch.setenv("SYSTEMONE_TOOL_MODEL_WEIGHT", "0.4")
    monkeypatch.setenv("SYSTEMONE_TOOL_VETO", "0.25")
    # webfetch: keyword "fetch" matches + decent model score.
    # browserclaw: no keyword match, high model score.
    # bash: no keyword match, low model score.
    engine = ScoringStubEngine(
        tool_scores={"webfetch": 0.7, "browserclaw": 0.9, "bash": 0.2},
        prior_scores={"webfetch": 0.5, "browserclaw": 0.5, "bash": 0.5},
    )
    tools = [
        {"id": "webfetch", "kind": "tool", "description": "fetch a URL",
         "triggers": ["fetch"]},
        {"id": "browserclaw", "kind": "mcp", "description": "browser automation",
         "triggers": ["browser"]},
        {"id": "bash", "kind": "tool", "description": "run shell commands",
         "triggers": ["shell"]},
    ]
    ranked = score_tools(engine, "fetch that page", tools)
    # webfetch wins on keyword; browserclaw beats bash on model score
    assert [r["id"] for r in ranked] == ["webfetch", "browserclaw", "bash"]
    assert ranked[0]["relevance"] > ranked[1]["relevance"] > ranked[2]["relevance"]


def test_score_tools_veto_kills_keyword_false_positive(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_TOOL_KW_WEIGHT", "0.6")
    monkeypatch.setenv("SYSTEMONE_TOOL_MODEL_WEIGHT", "0.4")
    monkeypatch.setenv("SYSTEMONE_TOOL_VETO", "0.25")
    # runescape matches keyword "prices" but model says irrelevant (< veto)
    engine = ScoringStubEngine(
        tool_scores={"runescape": 0.1, "webfetch": 0.8},
        prior_scores={"runescape": 0.05, "webfetch": 0.5},
    )
    tools = [
        {"id": "runescape", "kind": "mcp",
         "description": "game data prices", "triggers": ["prices"]},
        {"id": "webfetch", "kind": "tool",
         "description": "fetch a URL", "triggers": ["fetch"]},
    ]
    ranked = score_tools(engine, "scrape product prices from the site", tools)
    assert ranked[0]["id"] == "webfetch"
    assert ranked[1]["id"] == "runescape"


def test_score_tools_prior_subtraction_surfaces_specific_tool(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_TOOL_KW_WEIGHT", "0.6")
    monkeypatch.setenv("SYSTEMONE_TOOL_MODEL_WEIGHT", "0.4")
    monkeypatch.setenv("SYSTEMONE_TOOL_VETO", "0.25")
    # bash has a huge prior (generic); weather has a small prior but lifts
    # on a weather task. No keyword matches -> pure model ranking.
    engine = ScoringStubEngine(
        tool_scores={"bash": 0.95, "weather": 0.75},
        prior_scores={"bash": 0.93, "weather": 0.30},
    )
    tools = [
        {"id": "bash", "kind": "tool", "description": "run shell commands"},
        {"id": "weather", "kind": "mcp", "description": "weather forecasts"},
    ]
    ranked = score_tools(engine, "will it rain tomorrow", tools)
    # weather adjusted = 0.45, bash adjusted = 0.02 -> weather wins
    assert ranked[0]["id"] == "weather"


def test_score_tools_contrastive_positive_negative_pair(monkeypatch):
    # obvious positive must outrank obvious negatives, whatever the scale
    monkeypatch.setenv("SYSTEMONE_TOOL_KW_WEIGHT", "0.6")
    monkeypatch.setenv("SYSTEMONE_TOOL_MODEL_WEIGHT", "0.4")
    monkeypatch.setenv("SYSTEMONE_TOOL_VETO", "0.25")
    tools = [
        {"id": "webfetch", "kind": "tool",
         "description": "Fetch a URL and extract its text content.",
         "triggers": ["fetch", "url"]},
        {"id": "cannaai", "kind": "mcp",
         "description": "Cannabis grow operation sensors and breeding logs.",
         "triggers": ["cannabis"]},
        {"id": "runescape", "kind": "mcp",
         "description": "RuneScape game data: prices and hiscores.",
         "triggers": ["runescape"]},
    ]
    engine = ScoringStubEngine(
        tool_scores={"webfetch": 0.92, "cannaai": 0.05, "runescape": 0.03},
        prior_scores={"webfetch": 0.5, "cannaai": 0.05, "runescape": 0.05},
    )
    ranked = score_tools(engine, "Download the HTML of https://example.com",
                         tools)
    assert ranked[0]["id"] == "webfetch"
    assert {r["id"] for r in ranked[1:]} == {"cannaai", "runescape"}


def test_score_tools_falls_back_to_choice_question(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_TOOL_KW_WEIGHT", "0.6")
    monkeypatch.setenv("SYSTEMONE_TOOL_MODEL_WEIGHT", "0.4")
    monkeypatch.setenv("SYSTEMONE_TOOL_VETO", "0.25")
    class NoRawScores:
        def systemone(self, state, questions, batch_size=32):
            opts = questions[0]["options"]
            # last option wins the contrastive distribution
            probs = {o: (0.9 if o.startswith("bash:") else 0.05)
                     for o in opts}
            return {"tools": {"type": "choice", "choice": "bash",
                              "probabilities": probs, "confidence": 0.9}}
    tools = [
        {"id": "webfetch", "kind": "tool", "description": "fetch a URL"},
        {"id": "bash", "kind": "tool", "description": "run shell commands",
         "triggers": ["shell"]},
    ]
    ranked = score_tools(NoRawScores(), "run a shell command", tools)
    assert ranked[0]["id"] == "bash"
    assert ranked[0]["relevance"] == pytest.approx(1.0)


def test_score_tools_fail_open_on_engine_error():
    engine = ScoringStubEngine(fail_tools=True)
    assert score_tools(engine, "do things", _tools()) == []


def test_score_tools_empty_tools():
    assert score_tools(ScoringStubEngine(), "do things", []) == []


def test_tool_option_text_strips_fails_clauses():
    from systemone.scoring import _tool_option_text
    text = _tool_option_text({
        "id": "read",
        "description": "Read file contents from the workspace. FAILS at web pages.",
    })
    assert text == "read: Read file contents from the workspace."
    assert "FAILS" not in text


def test_tool_knobs_from_env(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_TOOL_KW_WEIGHT", "0.7")
    monkeypatch.setenv("SYSTEMONE_TOOL_MODEL_WEIGHT", "0.3")
    monkeypatch.setenv("SYSTEMONE_TOOL_VETO", "0.2")
    assert tool_kw_weight() == pytest.approx(0.7)
    assert tool_model_weight() == pytest.approx(0.3)
    assert tool_veto_threshold() == pytest.approx(0.2)


def test_load_tool_registry_bundled():
    path = os.path.join(os.path.dirname(shim.__file__), "tool_registry.json")
    tools = load_tool_registry(path)
    assert len(tools) > 10
    assert any(t["id"] == "browserclaw" for t in tools)


def test_load_tool_registry_missing_is_empty():
    assert load_tool_registry("/nonexistent/tool_registry.json") == []


# -- LM Studio inventory ----------------------------------------------------------

def test_fetch_lmstudio_models_unreachable_returns_none():
    assert fetch_lmstudio_models("http://127.0.0.1:1", timeout=0.2) is None


def test_apply_inventory_marks_availability():
    reg = _registry()
    apply_inventory(reg, ["e"])
    assert reg["economy"]["models"][0]["available"] is True
    assert reg["balanced"]["models"][0]["available"] is False


# -- plan ranking -------------------------------------------------------------------

def test_estimate_steps_enumerated():
    text = "1. gather facts\n2. draft answer\n3. verify"
    assert estimate_steps(text) == 3


def test_estimate_steps_prose_fallback():
    text = " ".join(["word"] * 90)
    assert estimate_steps(text) == 3


def test_rank_plans_sorts_by_score():
    engine = ScoringStubEngine(noul_probs={"plan_0": 0.9, "plan_1": 0.4})
    plans = [{"id": "a", "text": "1. x"}, {"id": "b", "text": "1. y\n2. z"}]
    ranked = rank_plans(engine, "do it", plans, tier_cost=1.0)
    assert ranked[0]["id"] == "a"
    assert ranked[0]["score"] > ranked[1]["score"]
    assert ranked[0]["p_success"] == pytest.approx(0.9)


def test_rank_plans_fail_open_keeps_order_with_none_scores():
    engine = ScoringStubEngine(fail_noul=True)
    plans = [{"id": "a", "text": "1. x"}, {"id": "b", "text": "1. y"}]
    ranked = rank_plans(engine, "do it", plans)
    assert [r["id"] for r in ranked] == ["a", "b"]
    assert all(r["score"] is None for r in ranked)


# -- route_decision scoring block -----------------------------------------------------

def test_route_with_scoring_adds_surface():
    engine = ScoringStubEngine(
        tool_scores={"webfetch": 0.9, "browserclaw": 0.85, "bash": 0.1})
    # medium-effort task -> tools get scored
    route = route_decision(
        engine, _medium_task(), _cands(), "balanced",
        scoring=_scoring_ctx(tools=_tools()))
    for key in ("calibrated", "calibrated_probabilities", "margin",
                "uncertain", "ranked_models", "ranked_tools", "tool_scoring"):
        assert key in route, key
    assert route["calibrated"] is True
    assert route["uncertain"] is False
    assert route["tool_scoring"] == "full"
    # floor 0.30 drops bash (0.1); top-k keeps order
    assert [t["id"] for t in route["ranked_tools"]] == ["webfetch", "browserclaw"]
    assert route["ranked_models"]  # at least one available model
    # confidence is now the calibrated top-1 probability
    assert route["confidence"] == pytest.approx(
        max(route["calibrated_probabilities"].values()), abs=1e-3)


def test_route_without_scoring_is_legacy_shape():
    engine = ScoringStubEngine()
    route = route_decision(engine, "what is 2+2?", _cands(), "economy")
    assert "ranked_tools" not in route
    assert "uncertain" not in route
    assert "confidence" in route  # legacy heuristic confidence


def test_uncertain_bumps_effort_and_skips_tools():
    engine = ScoringStubEngine(tool_scores={"webfetch": 0.95})
    route = route_decision(
        engine, "what is 2+2?", _cands(), "economy",
        scoring=_scoring_ctx(T=100.0, tools=_tools()))  # -> uncertain
    assert route["uncertain"] is True
    assert route["effort"] == "medium"  # bumped from low
    assert route["ranked_tools"] == []
    assert route["tool_scoring"] == "skipped"
    assert "effort bumped" in route["rationale"]


def test_tool_scoring_env_knobs(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_TOOL_FLOOR", "0.35")
    monkeypatch.setenv("SYSTEMONE_TOOL_TOPK", "1")
    engine = ScoringStubEngine(tool_scores={"webfetch": 0.95, "browserclaw": 0.92})
    route = route_decision(
        engine, _medium_task(), _cands(), "balanced",
        scoring=_scoring_ctx(tools=_tools()))
    assert [t["id"] for t in route["ranked_tools"]] == ["webfetch"]


# -- HTTP surface ---------------------------------------------------------------------

def _serve_on_temp_port(engine, **kwargs):
    server = serve(port=0, engine=engine,
                   registry=_registry(), **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _post(port, path, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        return e.code, json.loads(e.read().decode())


def test_rank_plans_http_roundtrip():
    engine = ScoringStubEngine(noul_probs={"plan_0": 0.9, "plan_1": 0.4})
    server = _serve_on_temp_port(engine)
    port = server.server_address[1]
    try:
        status, body = _post(port, "/v1/systemone/rank-plans", {
            "task": "migrate the database",
            "plans": [{"id": "a", "text": "1. backup\n2. migrate"},
                      {"id": "b", "text": "1. migrate"}],
        })
        assert status == 200
        assert [r["id"] for r in body["ranking"]] == ["a", "b"]
        assert body["tier"] in ("economy", "balanced", "heavy")
    finally:
        server.shutdown()


def test_rank_plans_http_rejects_bad_body():
    engine = ScoringStubEngine()
    server = _serve_on_temp_port(engine)
    port = server.server_address[1]
    try:
        status, body = _post(port, "/v1/systemone/rank-plans",
                             {"task": "x", "plans": []})
        assert status == 400
        assert "error" in body
    finally:
        server.shutdown()


def test_disable_kill_switch_503s_routing(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_DISABLE", "1")
    engine = ScoringStubEngine()
    server = _serve_on_temp_port(engine)
    port = server.server_address[1]
    try:
        status, _ = _post(port, "/v1/systemone/route", {"task": "what is 2+2?"})
        assert status == 503
        status, _ = _post(port, "/v1/systemone/rank-plans",
                          {"task": "x", "plans": [{"text": "1. y"}]})
        assert status == 503
    finally:
        monkeypatch.delenv("SYSTEMONE_DISABLE")
        server.shutdown()


def test_route_http_includes_scoring_surface():
    engine = ScoringStubEngine(tool_scores={"webfetch": 0.9})
    server = _serve_on_temp_port(engine)
    port = server.server_address[1]
    try:
        status, body = _post(port, "/v1/systemone/route", {
            "task": _medium_task(),
        })
        assert status == 200
        route = body["route"]
        for key in ("calibrated", "calibrated_probabilities", "margin",
                    "uncertain", "ranked_models", "ranked_tools"):
            assert key in route, key
    finally:
        server.shutdown()


# -- tuning.json config -------------------------------------------------------

def test_tuning_json_provides_documented_defaults(monkeypatch):
    for var in ("SYSTEMONE_MARGIN_FLOOR", "SYSTEMONE_COST_LAMBDA",
                "SYSTEMONE_TOOL_FLOOR", "SYSTEMONE_TOOL_TOPK",
                "SYSTEMONE_INVENTORY_TTL"):
        monkeypatch.delenv(var, raising=False)
    scoring.reset_tuning()
    try:
        assert scoring.margin_floor() == pytest.approx(0.15)
        assert scoring.cost_lambda() == pytest.approx(0.15)
        assert scoring.tool_floor() == pytest.approx(0.30)
        assert scoring.tool_topk() == 8
        assert scoring.inventory_ttl() == pytest.approx(300.0)
    finally:
        scoring.reset_tuning()


def test_tuning_env_overrides_config(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_MARGIN_FLOOR", "0.5")
    monkeypatch.setenv("SYSTEMONE_TOOL_TOPK", "3")
    scoring.reset_tuning()
    try:
        assert scoring.margin_floor() == pytest.approx(0.5)
        assert scoring.tool_topk() == 3
        # untouched knobs still come from tuning.json
        assert scoring.cost_lambda() == pytest.approx(0.15)
    finally:
        scoring.reset_tuning()


def test_tuning_missing_file_degrades_gracefully(monkeypatch, tmp_path):
    for var in ("SYSTEMONE_MARGIN_FLOOR", "SYSTEMONE_COST_LAMBDA",
                "SYSTEMONE_TOOL_FLOOR", "SYSTEMONE_TOOL_TOPK"):
        monkeypatch.delenv(var, raising=False)
    scoring.reset_tuning()
    scoring.load_tuning(str(tmp_path / "nonexistent.json"))
    try:
        assert scoring.margin_floor() is None
        assert scoring.tool_topk() is None
        # uncertain never fires without a floor (fail-open)
        cal = apply_calibration(
            {"economy": 0.5, "balanced": 0.49, "heavy": 0.01},
            {"temperature": 1.0})
        assert cal["uncertain"] is False
        # model ranking degrades to pure expected quality
        ranked = rank_models(_registry(),
                             {"economy": 0.1, "balanced": 0.1, "heavy": 0.8})
        assert ranked
    finally:
        scoring.reset_tuning()
