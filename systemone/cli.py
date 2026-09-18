"""systemone CLI — local equivalent of Loki's `/jev` commands.

No API keys here (everything is local), so the commands are about engine
health and one-shot judgments instead of credential management:

    python -m systemone.cli status
        # engine config check WITHOUT loading a model:
        #   configured model, device, calibrator, mcp lib compatibility

    python -m systemone.cli status --load
        # also loads the model once and reports real latency

    python -m systemone.cli ask --state "..." --questions questions.json
        # one-shot typesafe_ask; questions.json is a list of
        # {"id","type","instructions","criteria"} (or "-" for stdin)

Decision tools and routing are separate capabilities, mirroring Loki's
`/jev` design: allowing one does not imply the other.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .api import MAX_STATE_CHARS, MODEL_CANDIDATES, SystemOne, SystemOneError


def _env_model() -> str:
    return (os.environ.get("SYSTEMONE_MODEL") or "").strip() or MODEL_CANDIDATES[0]


def _env_device() -> str:
    return (os.environ.get("SYSTEMONE_DEVICE") or "auto").strip()


def cmd_status(args: argparse.Namespace) -> int:
    """Config/health check. Only loads a model with --load (one at a time)."""
    import torch

    print("systemone status")
    print(f"  configured model : {_env_model()}")
    print(f"  device setting   : {_env_device()} "
          f"(cuda available: {torch.cuda.is_available()})")
    print(f"  max state chars  : {MAX_STATE_CHARS}")

    cal_path = os.environ.get("SYSTEMONE_CALIBRATOR", "").strip()
    if cal_path:
        ok = os.path.exists(cal_path)
        print(f"  calibrator       : {cal_path} ({'found' if ok else 'MISSING'})")
    else:
        print("  calibrator       : none (raw softmax)")

    try:
        import mcp  # noqa: F401

        print(f"  mcp lib          : ok ({mcp.__version__ if hasattr(mcp, '__version__') else 'installed'})")
    except Exception as exc:
        print(f"  mcp lib          : PROBLEM ({exc})")

    if args.load:
        try:
            eng = SystemOne()
        except SystemOneError as exc:
            print(f"  model load       : FAILED — {exc}")
            return 1
        print(f"  model load       : ok ({eng.model_name} on {eng.device})")
        # tiny latency probe, one batched pass
        out = eng.systemone(
            "status probe",
            [{"name": "ok", "type": "noul", "statement": "Is this a probe?"}],
        )
        print(f"  probe latency    : {out['_meta']['latency_ms']} ms "
              f"(P(yes)={out['ok']['probability']:.3f})")
    else:
        print("  model load       : skipped (use --load to verify)")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """One-shot typed judgments over a state, Jev-style."""
    src = args.questions
    try:
        raw = sys.stdin.read() if src == "-" else open(src, encoding="utf-8").read()
    except OSError as exc:
        print(f"error: cannot read questions file: {exc.strerror or exc}", file=sys.stderr)
        return 2
    try:
        questions = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"error: questions file is not valid JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(questions, list) or not questions:
        print("error: questions must be a non-empty JSON list", file=sys.stderr)
        return 2

    # Reuse the MCP server's Jev-compatible validation/conversion so the CLI
    # and the tool accept exactly the same question shape.
    from .mcp_server import _convert_questions, _to_jev_answers, get_engine

    try:
        converted, level_orders = _convert_questions(questions)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        engine = get_engine(model_name=args.model)
        out = engine.systemone(args.state, converted)
        result = _to_jev_answers(out, questions, level_orders)
    except SystemOneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="systemone", description="Local System One CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status", help="engine health check")
    p_status.add_argument("--load", action="store_true",
                          help="also load the model and run a latency probe")
    p_status.set_defaults(func=cmd_status)

    p_ask = sub.add_parser("ask", help="one-shot typed judgments (Jev-style)")
    p_ask.add_argument("--state", required=True, help="state text to judge")
    p_ask.add_argument("--questions", required=True,
                       help='JSON file with [{"id","type","instructions","criteria"}] or "-" for stdin')
    p_ask.add_argument("--model", default=None, help="GLiClass checkpoint (default: env/smallest)")
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
