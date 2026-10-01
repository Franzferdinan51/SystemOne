"""SGLang backend demo: the same questions, local vs SGLang judges.

Shows the SGLangBackend answering identical choice/score/noul questions
through a real SGLang server's /v1/decisions endpoint, and — when the
local GLiClass engine is available — how its answers compare.

Run:
    SGLANG_BASE_URL=http://127.0.0.1:30000 python examples/demo_sglang_backend.py

The SGLang part is skipped gracefully when no server is reachable; the
local part is skipped when the GLiClass checkpoint can't load. Nothing
here needs a GPU.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone import SGLangBackend, SGLangError, make_questions  # noqa: E402

STATE = (
    "Screenshot: desktop with 47 icons. Visible: 'Q3-report-final-FINAL.pdf', "
    "'screenshot-2026-08-01.png' through 'screenshot-2026-08-19.png', "
    "'untitled-1.txt' through 'untitled-12.txt', 'installer.exe', "
    "'vacation-photos' folder, 'tax-2025' folder."
)

QUESTIONS = make_questions(
    choices={
        "dominant_mess": ["screenshots", "documents", "installers", "mixed"],
    },
    scores={
        "cleanup_value": ["low", "medium", "high"],
    },
    nouls={
        "needs_attention": "Does this desktop need organizing?",
    },
)


def show(title, answers):
    print(f"--- {title} ---")
    for name, ans in answers.items():
        if name == "_meta":
            continue
        extra = ""
        if ans.get("label_mass") is not None:
            extra = f", label_mass={ans['label_mass']:.3f}"
        if ans["type"] == "choice":
            print(f"  {name}: {ans['choice']} (conf {ans['confidence']:.2f}{extra})")
        elif ans["type"] == "score":
            print(f"  {name}: {ans['level']} (conf {ans['confidence']:.2f}{extra})")
        else:
            print(f"  {name}: {ans['answer']} (p={ans['probability']:.2f}{extra})")
    meta = answers["_meta"]
    print(f"  [{meta.get('backend', 'local')} {meta['model']}, "
          f"{meta['latency_ms']}ms]\n")


def main() -> None:
    # -- SGLang side -----------------------------------------------------
    backend = SGLangBackend()
    if backend.health():
        try:
            answers = backend.systemone(STATE, QUESTIONS)
            show("SGLang /v1/decisions", answers)
        except SGLangError as e:
            print(f"SGLang backend failed: {e}\n")
    else:
        print(f"SGLang server not reachable at {backend.base_url} — "
              "start one to see the live comparison, e.g.:\n"
              "  python -m sglang.launch_server "
              "--model-path Qwen/Qwen3.5-0.6B --port 30000\n")

    # -- local side ------------------------------------------------------
    try:
        from systemone import SystemOne
        eng = SystemOne()
        show("local GLiClass", eng.systemone(STATE, QUESTIONS))
    except Exception as e:
        print(f"local engine unavailable ({type(e).__name__}); "
              "SGLang comparison above stands alone.")


if __name__ == "__main__":
    main()
