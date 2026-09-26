"""Jeff-1 sidecar server — GestaltLabs/Jeff-1 as a second decision head.

Standalone stdlib HTTP server (like systemone.shim): loads the Jeff-1 LoRA
adapter once — lazily, on the first POST — then serves:

    POST /v1/jeff1/rank-plans
        {"task": "...", "plans": [{"id": "...", "text": "..."}]}
        -> {"ranking": [{"id": "...", "p_success": 0.83}]}
        Noul head: P(plan succeeds | task), one forward pass per plan.

    POST /v1/jeff1/second-opinion
        {"task": "...", "tier": "balanced", "confidence": 0.41,
         "margin": 0.05,
         "candidates": [{"tier": "economy", "description": "..."}, ...]}
        -> {"tier": "balanced", "confidence": 0.72, "agree": true,
            "rationale": "..."}
        Choice head over the candidate tiers. ADVISORY ONLY — the shim
        never changes the routed tier based on this reply.

    POST /v1/jeff1/decide
        {"state": <any>, "instructions": "...",
         "criteria": {"label": "description", ...},
         "type": "choice" | "noul" | "score"}
        -> {"type": "choice", "label": "...", "probabilities": {...},
            "confidence": 0.72, "latency_ms": 12.3}
        Generic typed decision over the Jeff-1 readout heads. "noul"
        answers yes/no (criteria optional: {"yes","no"} or [yes, no]);
        "score" rates ordered levels — criteria is either a list of
        level descriptions or a dict keyed "0".."n-1".
        Confidence always uses the TypeSafe-compatible helpers from
        systemone/api.py (adapted from Mapika/decider, Apache-2.0).

    GET /healthz (and GET /)
        -> {"ok": true, "model": "<base>+<adapter>", "device": "mps",
            "loaded": true}

Environment (no hard-coded model ids or knobs):

    JEFF1_ADAPTER_ID   default "GestaltLabs/Jeff-1"
    JEFF1_BASE_ID      default "Qwen/Qwen3-4B-Instruct-2507"
    JEFF1_DEVICE       default auto: cuda -> mps -> cpu
    JEFF1_MAX_LENGTH   default "2048"
    JEFF1_HOST         bind address, default "127.0.0.1" (also: --host;
                       use "0.0.0.0" or the tailnet IP to serve other machines)
    JEFF1_PORT         default "8079" (also: --port)

Memory: the 4B bf16 base + LoRA adapter needs ~8-9 GB of device memory. On
Apple Silicon (MPS) that comes out of the Mac's unified pool — the Mac mini
(M4 Pro, 24 GB) holds the sidecar alongside the GLiClass shim comfortably.
The Windows PC never runs this process; its shim points
SYSTEMONE_JEFF1_URL at the Mac sidecar instead.

Label scoring is ported from Gestalt-Lab/jeff's jev_clf/readout.py
(Apache-2.0): the prompt is question text + "\\n\\nState:\\n" + state +
"\\n\\nVerdict:", wrapped in the Qwen3 chat template, and the answer is read
out of the model's own next-token distribution — the first-token readout
when labels have distinct first tokens (choice, yes/no), the whole-sequence
readout when they share one (score levels "0".."3" all start with the
bare-space token). The system prompt is kept verbatim from the jeff client
(it is what the adapter was fine-tuned with).

Run:  python -m systemone.jeff1_sidecar [--port 8079]
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

# TypeSafe-compatible confidence helpers (choice_confidence,
# noul_confidence, score_confidence) — confidence semantics adapted from
# Mapika/decider (Apache-2.0) via this repo's api.py; imported, never
# reimplemented or hard-coded.
from .api import choice_confidence, noul_confidence, score_confidence

DEFAULT_ADAPTER_ID = "GestaltLabs/Jeff-1"
DEFAULT_BASE_ID = "Qwen/Qwen3-4B-Instruct-2507"

_SYSTEM = (
    "You are a fact-checking classifier. You are given a claim and the evidence "
    "passages retrieved for it. You answer with exactly one verdict label."
)


# -- config -----------------------------------------------------------------

def _env(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def adapter_id() -> str:
    return _env("JEFF1_ADAPTER_ID", DEFAULT_ADAPTER_ID)


def base_id() -> str:
    return _env("JEFF1_BASE_ID", DEFAULT_BASE_ID)


def max_length() -> int:
    raw = _env("JEFF1_MAX_LENGTH", "2048")
    try:
        n = int(raw)
        return n if n > 0 else 2048
    except (TypeError, ValueError):
        return 2048


def resolve_device() -> str:
    """CUDA -> MPS -> CPU, overridable with JEFF1_DEVICE."""
    forced = _env("JEFF1_DEVICE", "")
    if forced:
        return forced
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


# -- prompt construction (jeff jev_clf mechanics) ----------------------------

def _question_text(instructions: str,
                   labels: List[Tuple[str, Optional[str]]]) -> str:
    """instructions + one `label: definition` line per label."""
    lines = [instructions or ""]
    for label, definition in labels:
        lines.append(f"{label}: {definition}" if definition else label)
    return "\n".join(lines)


def _state_text(state: Any) -> str:
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return "\n".join(f"{k}: {v}" for k, v in state.items())
    return str(state)


def build_prompt(tokenizer: Any, instructions: str,
                 labels: List[Tuple[str, Optional[str]]],
                 state: Any) -> str:
    """question text + state, wrapped in the chat template."""
    user = (_question_text(instructions, labels)
            + "\n\nState:\n" + _state_text(state)
            + "\n\nVerdict:")
    messages = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{_SYSTEM}\n\n{user}\n"


# -- label readout (ported from Gestalt-Lab/jeff jev_clf/readout.py) ---------

def _label_token_variants(tokenizer: Any,
                          labels: List[str]) -> Dict[str, List[int]]:
    """Token ids per label: leading-space form first, bare label fallback."""
    out: Dict[str, List[int]] = {}
    for label in labels:
        ids = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
        if not ids:
            ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        out[label] = ids
    return out


def _choose_mode(variants: Dict[str, List[int]]) -> str:
    """first_token iff every label's first token is distinct, else sequence."""
    seqs = list(variants.values())
    if seqs and all(seqs) and len({s[0] for s in seqs}) == len(seqs):
        return "first_token"
    return "sequence"


def _distribution(model: Any, tokenizer: Any, text: str, labels: List[str],
                  device: str, max_len: int) -> Dict[str, float]:
    """P(label | prompt) from the model's next-token distribution."""
    import torch

    variants = _label_token_variants(tokenizer, labels)
    mode = _choose_mode(variants)
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_len).to(device)
    with torch.no_grad():
        logits = model(**enc).logits[0, -1].float()

    if mode == "first_token":
        sub = torch.tensor([logits[variants[label][0]] for label in labels])
        probs = torch.softmax(sub, dim=-1)
        return {label: float(p) for label, p in zip(labels, probs)}

    # sequence readout: one extra forward per label over prompt + label
    prompt_ids = enc["input_ids"][0].tolist()
    totals: List[float] = []
    for label in labels:
        seq = variants[label]
        full = prompt_ids + seq
        with torch.no_grad():
            out = model(input_ids=torch.tensor([full]).to(device)).logits[0]
        total = 0.0
        for k, tid in enumerate(seq):
            pos = (len(full) - len(seq)) - 1 + k
            total += float(torch.log_softmax(out[pos].float(), dim=-1)[tid])
        totals.append(total)
    probs = torch.softmax(torch.tensor(totals, dtype=torch.float32), dim=-1)
    return {label: float(p) for label, p in zip(labels, probs)}


# -- engine ------------------------------------------------------------------

class Jeff1Engine:
    """Lazy Jeff-1 LoRA: choice / noul judgments from local weights."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Any = None
        self._tokenizer: Any = None
        self.device = resolve_device()
        self.model_id = f"{base_id()}+{adapter_id().split('/')[-1]}"
        self.load_error: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load base + adapter. Raises on failure (callers catch)."""
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            base, adapter = base_id(), adapter_id()
            dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
            self._tokenizer = AutoTokenizer.from_pretrained(base)
            model = AutoModelForCausalLM.from_pretrained(
                base, dtype=dtype, device_map="cpu")
            model = PeftModel.from_pretrained(model, adapter)
            self._model = model.to(self.device).eval()
            # device/memory log line at load
            try:
                n_params = sum(p.numel() for p in self._model.parameters())
                gb = n_params * (4 if dtype == torch.float32 else 2) / 1e9
                print(f"[jeff1] loaded {self.model_id} on {self.device} "
                      f"~{gb:.1f} GB, {n_params / 1e9:.2f}B params",
                      flush=True)
            except Exception:
                print(f"[jeff1] loaded {self.model_id} on {self.device}",
                      flush=True)

    def _judge(self, state: Any, instructions: str,
               labels: List[Tuple[str, Optional[str]]]
               ) -> Tuple[str, Dict[str, float], float]:
        with self._lock:
            if self._model is None:
                raise RuntimeError("jeff-1 model not loaded")
            text = build_prompt(self._tokenizer, instructions, labels, state)
            probs = _distribution(self._model, self._tokenizer, text,
                                  [label for label, _ in labels],
                                  self.device, max_length())
        top = max(probs.items(), key=lambda kv: kv[1])[0]
        return top, probs, max(probs.values())

    def noul(self, state: Any, instructions: str,
             yes_desc: Optional[str] = None,
             no_desc: Optional[str] = None) -> float:
        """P(yes | state)."""
        _, probs, _ = self._judge(
            state, instructions, [("yes", yes_desc), ("no", no_desc)])
        return float(probs["yes"])

    def choice(self, state: Any, instructions: str,
               criteria: Dict[str, Optional[str]]
               ) -> Tuple[str, Dict[str, float], float]:
        """(choice, probabilities, confidence) over the criteria labels."""
        labels = [(label, criteria.get(label)) for label in criteria]
        return self._judge(state, instructions, labels)


_ENGINE: Optional[Jeff1Engine] = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> Jeff1Engine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = Jeff1Engine()
        return _ENGINE


# -- request handling ---------------------------------------------------------

_PLAN_NOUL_INSTRUCTIONS = (
    "Given the task and the proposed plan below, "
    "is this plan likely to succeed at the task?"
)
_PLAN_NOUL_YES = "The plan is likely to succeed at the task."
_PLAN_NOUL_NO = "The plan is unlikely to succeed at the task."

_OPINION_INSTRUCTIONS = (
    "Choose the capability tier best suited to handle this task. "
    "A previous judge was uncertain, so weigh the tier descriptions "
    "carefully and pick the tier most likely to do the task well."
)


def _handle_rank_plans(engine: Jeff1Engine,
                       body: Dict[str, Any]) -> Dict[str, Any]:
    task = body.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("request must include a non-empty 'task' string")
    plans = body.get("plans")
    if not isinstance(plans, list) or not plans:
        raise ValueError("'plans' must be a non-empty list")
    for p in plans:
        if not isinstance(p, dict) or not isinstance(p.get("text"), str):
            raise ValueError("each plan must be a mapping with a 'text' string")
    ranking = []
    for i, p in enumerate(plans):
        pid = p.get("id", f"plan_{i}")
        try:
            p_success = engine.noul(
                {"task": task.strip(), "plan": (p.get("text") or "")[:2000]},
                _PLAN_NOUL_INSTRUCTIONS, _PLAN_NOUL_YES, _PLAN_NOUL_NO)
            ranking.append({"id": pid, "p_success": round(p_success, 4)})
        except Exception:
            ranking.append({"id": pid, "p_success": None})
    return {"ranking": ranking}


def _handle_second_opinion(engine: Jeff1Engine,
                           body: Dict[str, Any]) -> Dict[str, Any]:
    task = body.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("request must include a non-empty 'task' string")
    routed_tier = body.get("tier")
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("'candidates' must be a non-empty list")
    criteria: Dict[str, str] = {}
    for c in candidates:
        if isinstance(c, dict) and isinstance(c.get("tier"), str):
            criteria[c["tier"]] = str(c.get("description", ""))
    if not criteria:
        raise ValueError("no usable candidate tiers")
    state = {"task": task.strip()}
    if routed_tier is not None:
        state["routed_tier"] = str(routed_tier)
    if body.get("confidence") is not None:
        state["routing_confidence"] = str(body.get("confidence"))
    if body.get("margin") is not None:
        state["routing_margin"] = str(body.get("margin"))
    tier, probs, confidence = engine.choice(
        state, _OPINION_INSTRUCTIONS, criteria)
    agree = (tier == routed_tier)
    top2 = sorted(probs.values(), reverse=True)
    margin = top2[0] - (top2[1] if len(top2) > 1 else 0.0)
    if agree:
        rationale = (f"Jeff-1 agrees with '{routed_tier}' "
                     f"(P={confidence:.2f}, margin {margin:.2f}).")
    else:
        rationale = (f"Jeff-1 prefers '{tier}' (P={confidence:.2f}) over "
                     f"the routed '{routed_tier}' "
                     f"(P={probs.get(routed_tier, 0.0):.2f}); advisory only.")
    return {"tier": tier, "confidence": round(confidence, 4),
            "agree": agree, "rationale": rationale}


# -- POST /v1/jeff1/decide ----------------------------------------------------

_DECIDE_TYPES = ("choice", "noul", "score")


def _decide_instructions(body: Dict[str, Any]) -> str:
    instructions = body.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError(
            "request must include a non-empty 'instructions' string")
    return instructions.strip()


def _decide_choice_criteria(raw: Any) -> Dict[str, Optional[str]]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            "'criteria' must be a non-empty mapping of label -> description")
    criteria: Dict[str, Optional[str]] = {}
    for label, desc in raw.items():
        if not isinstance(label, str) or not label:
            raise ValueError("criteria labels must be non-empty strings")
        if desc is not None and not isinstance(desc, str):
            raise ValueError(
                f"description for label '{label}' must be a string or null")
        criteria[label] = desc
    return criteria


def _decide_score_levels(raw: Any) -> List[Tuple[str, Optional[str]]]:
    """Score criteria -> [(level_label, description)] in level order.

    Accepts a dict keyed by contiguous level indexes "0".."n-1" or an
    ordered list of level descriptions.
    """
    indexed: Dict[int, Any] = {}
    if isinstance(raw, list) and raw:
        indexed = dict(enumerate(raw))
    elif isinstance(raw, dict) and raw:
        for key in raw:
            try:
                i = int(key)
            except (TypeError, ValueError):
                raise ValueError(
                    "score 'criteria' keys must be level indexes '0'..'n-1'")
            if i < 0 or i in indexed:
                raise ValueError(
                    "score 'criteria' keys must be level indexes '0'..'n-1'")
            indexed[i] = raw[key]
    else:
        raise ValueError(
            "score 'criteria' must be a non-empty list of level descriptions "
            "or a mapping '0'..'n-1' -> description")
    n = len(indexed)
    if sorted(indexed) != list(range(n)):
        raise ValueError(
            "score 'criteria' keys must be contiguous level indexes "
            "'0'..'n-1'")
    levels: List[Tuple[str, Optional[str]]] = []
    for i in range(n):
        desc = indexed[i]
        if desc is not None and not isinstance(desc, str):
            raise ValueError(
                f"description for level '{i}' must be a string or null")
        levels.append((str(i), desc))
    return levels


def _decide_noul_descriptions(raw: Any) -> Tuple[Optional[str], Optional[str]]:
    """Optional noul criteria -> (yes_desc, no_desc)."""
    if raw is None:
        return None, None
    if isinstance(raw, dict):
        yes_desc, no_desc = raw.get("yes"), raw.get("no")
    elif isinstance(raw, (list, tuple)) and len(raw) == 2:
        yes_desc, no_desc = raw[0], raw[1]
    else:
        raise ValueError(
            "'criteria' for noul must be null, a {yes, no} mapping, "
            "or a 2-item [yes, no] list")
    for name, desc in (("yes", yes_desc), ("no", no_desc)):
        if desc is not None and not isinstance(desc, str):
            raise ValueError(f"'criteria.{name}' must be a string or null")
    return yes_desc, no_desc


def _handle_decide(engine: Jeff1Engine,
                   body: Dict[str, Any]) -> Dict[str, Any]:
    """Generic typed decision: {"state", "instructions", "criteria", "type"}.

    choice -> engine.choice over the criteria labels;
    noul   -> engine.noul, label yes|no;
    score  -> engine.choice over "0".."n-1" level labels.
    Confidence always comes from the TypeSafe-compatible helpers in
    systemone/api.py (confidence semantics adapted from Mapika/decider,
    Apache-2.0) — never reimplemented here.
    """
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    if "state" not in body:
        raise ValueError("request must include 'state'")
    state = body["state"]
    instructions = _decide_instructions(body)
    decide_type = body.get("type")
    if decide_type not in _DECIDE_TYPES:
        raise ValueError("'type' must be one of 'choice', 'noul', 'score'")

    if decide_type == "choice":
        criteria = _decide_choice_criteria(body.get("criteria"))
        ordered = list(criteria.keys())
        label, probs, _ = engine.choice(state, instructions, criteria)
        ordered_probs = [float(probs[lab]) for lab in ordered]
        confidence = choice_confidence(ordered_probs)
        return {
            "type": "choice",
            "label": str(label),
            "probabilities": {lab: round(float(probs[lab]), 4)
                              for lab in ordered},
            "confidence": round(confidence, 4),
        }

    if decide_type == "noul":
        yes_desc, no_desc = _decide_noul_descriptions(body.get("criteria"))
        p_yes = float(engine.noul(state, instructions, yes_desc, no_desc))
        p_yes = min(1.0, max(0.0, p_yes))
        confidence = noul_confidence(p_yes)
        return {
            "type": "noul",
            "label": "yes" if p_yes >= 0.5 else "no",
            "probabilities": {"yes": round(p_yes, 4),
                              "no": round(1.0 - p_yes, 4)},
            "confidence": round(confidence, 4),
        }

    # score: engine.choice over "0".."n-1" level labels.
    levels = _decide_score_levels(body.get("criteria"))
    criteria = {lab: desc for lab, desc in levels}
    label, probs, _ = engine.choice(state, instructions, criteria)
    ordered = [str(i) for i in range(len(levels))]
    ordered_probs = [float(probs[lab]) for lab in ordered]
    confidence = score_confidence(ordered_probs)
    return {
        "type": "score",
        "level": str(label),
        "distribution": {lab: round(float(probs[lab]), 4) for lab in ordered},
        "confidence": round(confidence, 4),
    }


class Jeff1Handler(BaseHTTPRequestHandler):
    """HTTP handler; the engine is attached as `server.engine`."""

    server_version = "Jeff1Sidecar/1.0"

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def _ensure_engine(self) -> Jeff1Engine:
        """Lazy-load on first request; 503 when the weights won't load."""
        engine: Jeff1Engine = self.server.engine  # type: ignore[attr-defined]
        if not engine.loaded:
            try:
                engine.load()
            except Exception as exc:
                engine.load_error = f"{type(exc).__name__}: {exc}"
                raise RuntimeError(f"model unavailable: {engine.load_error}")
        return engine

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/healthz"):
            engine: Jeff1Engine = self.server.engine  # type: ignore[attr-defined]
            self._send_json(200, {
                "ok": True,
                "model": engine.model_id,
                "device": engine.device,
                "loaded": engine.loaded,
                "load_error": engine.load_error,
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        t0 = time.perf_counter()
        try:
            engine = self._ensure_engine()
            body = self._read_body()
            if self.path == "/v1/jeff1/rank-plans":
                payload = _handle_rank_plans(engine, body)
            elif self.path == "/v1/jeff1/second-opinion":
                payload = _handle_second_opinion(engine, body)
            elif self.path == "/v1/jeff1/decide":
                payload = _handle_decide(engine, body)
            else:
                self._send_json(404, {"error": "not found, POST "
                    "/v1/jeff1/rank-plans, /v1/jeff1/second-opinion "
                    "or /v1/jeff1/decide"})
                return
            payload["latency_ms"] = round(
                (time.perf_counter() - t0) * 1000.0, 1)
            self._send_json(200, payload)
        except ValueError as e:
            self._send_json(400, {"error": f"bad request: {e}"})
        except RuntimeError as e:
            self._send_json(503, {"error": str(e)})
        except Exception as e:  # never leak internals beyond the class name
            self._send_json(500, {"error": f"engine failure: {type(e).__name__}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        pass  # quiet; load prints its own line


def serve(port: int = 8079, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), Jeff1Handler)
    server.engine = get_engine()  # type: ignore[attr-defined]
    return server


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Jeff-1 second decision head (sidecar for SystemOne)")
    parser.add_argument("--host", type=str,
                        default=_env("JEFF1_HOST", "127.0.0.1"),
                        help="bind address (0.0.0.0 to serve the tailnet)")
    parser.add_argument("--port", type=int,
                        default=int(_env("JEFF1_PORT", "8079")))
    args = parser.parse_args(argv)
    engine = get_engine()
    print(f"jeff-1 sidecar on http://{args.host}:{args.port}/ "
          f"(adapter {adapter_id()}, base {base_id()}, device {engine.device}; "
          "model loads lazily on first request, ~8-9 GB)",
          flush=True)
    serve(args.port, args.host).serve_forever()


if __name__ == "__main__":
    main()
