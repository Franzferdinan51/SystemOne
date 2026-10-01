"""Desktop-cleaning planner: per-file decisions in ONE batched call.

The "desktop cleaning" agent pattern: scan a directory, then ask ONE
/v1/decisions-style question *per file* ("which folder should this go in?")
in a single batched call — 100 files = 1 HTTP round trip, not 100 chat
turns. The model proposes a plan, the human approves, the executor moves.

Safety: dry-run by default (prints the plan, moves nothing). --apply
performs the moves, and only inside the directory you pass explicitly.
The script NEVER defaults to your real Desktop.

Run:
    python examples/desktop_clean_dryrun.py            # demo on a scratch dir
    python examples/desktop_clean_dryrun.py --dir ~/Downloads --apply
    SGLANG_BASE_URL=... python examples/desktop_clean_dryrun.py --engine sglang

Engines: stub (offline heuristic), local (GLiClass), sglang (/v1/decisions).
"""

import argparse
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

FOLDER_CANDIDATES = ["images", "documents", "installers", "archives",
                     "audio", "video", "code", "keep-in-place"]

EXT_MAP = {
    ".png": "images", ".jpg": "images", ".jpeg": "images", ".gif": "images",
    ".webp": "images", ".svg": "images",
    ".pdf": "documents", ".txt": "documents", ".md": "documents",
    ".doc": "documents", ".docx": "documents",
    ".exe": "installers", ".msi": "installers", ".dmg": "installers",
    ".zip": "archives", ".tar": "archives", ".gz": "archives", ".7z": "archives",
    ".mp3": "audio", ".wav": "audio", ".flac": "audio",
    ".mp4": "video", ".mkv": "video", ".mov": "video",
    ".py": "code", ".js": "code", ".ts": "code", ".rs": "code",
}


def seed_scratch() -> str:
    """Build a demo mess in a temp dir (so the default run touches nothing)."""
    d = tempfile.mkdtemp(prefix="desktop-mess-")
    for name in ["screenshot-2026-08-01.png", "screenshot-2026-08-02.png",
                 "Q3-report-final-FINAL.pdf", "untitled-3.txt",
                 "installer.exe", "backup.zip", "voice-memo.mp3",
                 "demo-clip.mp4", "notes.py", "mystery-blob.xyz"]:
        open(os.path.join(d, name), "w").write("demo")
    return d


def stub_plan(files):
    """Offline heuristic judge: extension map, unknown -> keep-in-place."""
    return {f: EXT_MAP.get(os.path.splitext(f)[1].lower(), "keep-in-place")
            for f in files}


def engine_plan(engine_name, files):
    """One batched decisions call: a choice question per file."""
    if engine_name == "local":
        from systemone import SystemOne
        eng = SystemOne()
    elif engine_name == "sglang":
        from systemone import SGLangBackend, SGLangError
        eng = SGLangBackend()
        if not eng.health():
            raise SGLangError(f"SGLang not reachable at {eng.base_url}")
    else:
        raise ValueError(engine_name)
    state = ("You are organizing a messy desktop folder. "
             "For each file below, choose the folder it belongs in. "
             "'keep-in-place' means it is already fine where it is.")
    questions = [
        {"name": f, "type": "choice", "options": FOLDER_CANDIDATES,
         "prompt": f"Which folder should the file '{f}' go in?"}
        for f in files
    ]
    answers = eng.systemone(state, questions)
    plan = {}
    for f in files:
        ans = answers[f]
        # Uncertainty gate: low confidence -> keep-in-place, never guess.
        plan[f] = ans["choice"] if ans["confidence"] >= 0.4 else "keep-in-place"
    return plan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None,
                    help="directory to organize (default: temp scratch mess)")
    ap.add_argument("--apply", action="store_true",
                    help="actually move files (default: dry-run, print plan)")
    ap.add_argument("--engine", default="stub",
                    choices=["stub", "local", "sglang"])
    args = ap.parse_args()

    target = args.dir or seed_scratch()
    if not os.path.isdir(target):
        raise SystemExit(f"not a directory: {target}")
    files = sorted(f for f in os.listdir(target)
                   if os.path.isfile(os.path.join(target, f)))
    if not files:
        print("nothing to organize.")
        return

    plan = (stub_plan(files) if args.engine == "stub"
            else engine_plan(args.engine, files))

    moves = [(f, folder) for f, folder in plan.items()
             if folder != "keep-in-place"]
    print(f"scanned {len(files)} files in {target} "
          f"[engine={args.engine}]")
    for f, folder in plan.items():
        mark = "MOVE -> " + folder if folder != "keep-in-place" else "keep"
        print(f"  {f:40s} {mark}")

    if not args.apply:
        print(f"\ndry-run: {len(moves)} moves proposed, none executed "
              "(pass --apply to execute).")
        return
    for f, folder in moves:
        dest = os.path.join(target, folder)
        os.makedirs(dest, exist_ok=True)
        shutil.move(os.path.join(target, f), os.path.join(dest, f))
    print(f"\napplied {len(moves)} moves.")


if __name__ == "__main__":
    main()
