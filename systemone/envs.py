"""systemone.envs — Real Env implementations for the DecisionLoop.

Three concrete worlds that plug into systemone.loop.DecisionLoop
(See > Decide > Act). Each implements the Env protocol:

    observe() -> Observation   # what the world looks like now
    act(action: str) -> ActResult  # do something, report what happened

Envs:
    FileSystemEnv  — navigate and operate on the filesystem
    ShellEnv       — run shell commands with a safety gate
    WebFetchEnv    — fetch and summarize web content

Quick demo (offline, stub engine):
    python -m systemone.envs demo
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from systemone.loop import ActResult, Observation


# ---------------------------------------------------------------------------
# FileSystemEnv
# ---------------------------------------------------------------------------

class FileSystemEnv:
    """Navigate and operate on the local filesystem.

    Actions (one per act() call):
        ls [path]            list directory (default: cwd)
        cd <path>            change directory
        pwd                  print working directory
        read <file>          read a text file (first 4000 chars)
        write <file> <text>  write text to a file (creates/overwrites)
        append <file> <text> append text to a file
        mkdir <path>         create a directory
        rm <path>            delete a file (not directories)
        done                 mark the task complete

    State: cwd (str), last listing cached for observe().
    """

    PREVIEW_LIMIT = 4000

    def __init__(self, start_dir: str | None = None) -> None:
        self.cwd = os.path.abspath(start_dir or os.getcwd())
        self._last_output = ""

    # -- Env protocol -----------------------------------------------------
    def observe(self) -> Observation:
        try:
            entries = sorted(os.listdir(self.cwd))
        except OSError as e:
            entries = [f"<error listing dir: {e}>"]
        lines = [f"cwd: {self.cwd}", f"entries ({len(entries)}):"]
        for name in entries[:100]:
            full = os.path.join(self.cwd, name)
            tag = "d" if os.path.isdir(full) else "f"
            lines.append(f"  [{tag}] {name}")
        if len(entries) > 100:
            lines.append(f"  ... and {len(entries) - 100} more")
        if self._last_output:
            lines.append("")
            lines.append("last action output:")
            lines.append(self._last_output)
        return Observation(text="\n".join(lines), info={"cwd": self.cwd})

    def act(self, action: str) -> ActResult:
        action = action.strip()
        if not action:
            return ActResult(progressed=False, info={"error": "empty action"})
        parts = shlex.split(action)
        verb, args = parts[0].lower(), parts[1:]

        try:
            if verb == "done":
                self._last_output = "task marked done"
                return ActResult(progressed=True, done=True)
            if verb == "pwd":
                self._last_output = self.cwd
                return ActResult(progressed=True)
            if verb == "ls":
                target = self._resolve(args[0]) if args else self.cwd
                entries = sorted(os.listdir(target))
                self._last_output = "\n".join(entries[:200])
                return ActResult(progressed=True, info={"path": target})
            if verb == "cd":
                if not args:
                    return ActResult(progressed=False, info={"error": "cd needs a path"})
                target = self._resolve(args[0])
                if not os.path.isdir(target):
                    self._last_output = f"not a directory: {args[0]}"
                    return ActResult(progressed=False, info={"error": self._last_output})
                self.cwd = target
                self._last_output = f"cwd -> {self.cwd}"
                return ActResult(progressed=True)
            if verb == "read":
                if not args:
                    return ActResult(progressed=False, info={"error": "read needs a file"})
                target = self._resolve(args[0])
                with open(target, "r", errors="replace") as f:
                    content = f.read(self.PREVIEW_LIMIT)
                self._last_output = content
                return ActResult(progressed=True, info={"file": target})
            if verb in ("write", "append"):
                if len(args) < 2:
                    return ActResult(progressed=False, info={"error": f"{verb} needs <file> <text>"})
                target = self._resolve(args[0])
                text = " ".join(args[1:])
                mode = "w" if verb == "write" else "a"
                with open(target, mode) as f:
                    f.write(text)
                self._last_output = f"{verb}: {len(text)} chars -> {target}"
                return ActResult(progressed=True, info={"file": target})
            if verb == "mkdir":
                if not args:
                    return ActResult(progressed=False, info={"error": "mkdir needs a path"})
                os.makedirs(self._resolve(args[0]), exist_ok=True)
                self._last_output = f"created {args[0]}"
                return ActResult(progressed=True)
            if verb == "rm":
                if not args:
                    return ActResult(progressed=False, info={"error": "rm needs a file"})
                target = self._resolve(args[0])
                if os.path.isdir(target):
                    self._last_output = f"refusing to rm directory: {args[0]}"
                    return ActResult(progressed=False, info={"error": self._last_output})
                os.remove(target)
                self._last_output = f"removed {args[0]}"
                return ActResult(progressed=True)
            self._last_output = f"unknown command: {verb}"
            return ActResult(progressed=False, info={"error": self._last_output})
        except OSError as e:
            self._last_output = f"error: {e}"
            return ActResult(progressed=False, info={"error": str(e)})

    def _resolve(self, path: str) -> str:
        if not os.path.isabs(path):
            path = os.path.join(self.cwd, path)
        return os.path.normpath(path)


# ---------------------------------------------------------------------------
# ShellEnv
# ---------------------------------------------------------------------------

# Patterns that are never allowed through act(). Conservative on purpose.
_BLOCKED = [
    r"\brm\s+-[a-z]*r[a-z]*\s+/",      # rm -rf /
    r"\brm\s+--no-preserve-root\b",
    r":\(\)\s*{\s*:\s*|\s*:\s*&\s*}\s*;",  # fork bomb
    r"\bsudo\b",
    r"\bsu\b",
    r"\bmkfs\b",
    r"\bdd\b.*\bof=/dev/",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bhalt\b",
    r"\bpoweroff\b",
    r">\s*/dev/sd",
    r"\bchmod\s+-R\s+777\s+/",
    r"\bcurl\b.*\|\s*(ba)?sh\b",       # curl|sh
    r"\bwget\b.*\|\s*(ba)?sh\b",
]
_BLOCKED_RE = re.compile("|".join(_BLOCKED), re.IGNORECASE)


class ShellEnv:
    """Execute shell commands with a safety gate and timeout.

    Actions: any shell command string, e.g. "ls -la", "git status".
    Special: "done" marks the task complete.

    Safety: blocks destructive patterns (rm -rf /, sudo, fork bombs,
    curl|sh, writes to /dev, ...). Blocked commands return
    progressed=False with the reason — they never execute.

    State: cwd, history (list of {cmd, rc, output}).
    """

    TIMEOUT_S = 60
    OUTPUT_LIMIT = 8000

    def __init__(self, start_dir: str | None = None) -> None:
        self.cwd = os.path.abspath(start_dir or os.getcwd())
        self.history: List[Dict[str, Any]] = []
        self._last: Dict[str, Any] = {}

    # -- Env protocol -----------------------------------------------------
    def observe(self) -> Observation:
        lines = [f"cwd: {self.cwd}"]
        if self._last:
            lines.append(f"last cmd: {self._last.get('cmd', '')}")
            lines.append(f"exit code: {self._last.get('rc')}")
            lines.append("output:")
            lines.append(self._last.get("output", "") or "(empty)")
        else:
            lines.append("no commands run yet")
        lines.append(f"history: {len(self.history)} commands")
        return Observation(
            text="\n".join(lines),
            info={"cwd": self.cwd, "history_len": len(self.history)},
        )

    def act(self, action: str) -> ActResult:
        action = action.strip()
        if not action:
            return ActResult(progressed=False, info={"error": "empty action"})
        if action.lower() == "done":
            return ActResult(progressed=True, done=True)

        blocked = _BLOCKED_RE.search(action)
        if blocked:
            reason = f"blocked unsafe command (matched: {blocked.group(0)!r})"
            rec = {"cmd": action, "rc": -1, "output": reason}
            self.history.append(rec)
            self._last = rec
            return ActResult(progressed=False, info={"error": reason})

        try:
            proc = subprocess.run(
                action,
                shell=True,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT_S,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            out = out[: self.OUTPUT_LIMIT]
            rec = {"cmd": action, "rc": proc.returncode, "output": out}
            self.history.append(rec)
            self._last = rec
            # track cd so cwd stays accurate
            if action.strip().startswith("cd "):
                try:
                    self.cwd = os.getcwd()  # fallback; shell cd doesn't persist
                except OSError:
                    pass
            return ActResult(
                progressed=proc.returncode == 0,
                info={"rc": proc.returncode},
            )
        except subprocess.TimeoutExpired:
            reason = f"command timed out after {self.TIMEOUT_S}s"
            rec = {"cmd": action, "rc": -1, "output": reason}
            self.history.append(rec)
            self._last = rec
            return ActResult(progressed=False, info={"error": reason})
        except OSError as e:
            rec = {"cmd": action, "rc": -1, "output": str(e)}
            self.history.append(rec)
            self._last = rec
            return ActResult(progressed=False, info={"error": str(e)})


# ---------------------------------------------------------------------------
# WebFetchEnv
# ---------------------------------------------------------------------------

class WebFetchEnv:
    """Fetch web pages and keep a summary of what was retrieved.

    Actions:
        fetch <url>     download a URL, store truncated text
        done            mark the task complete

    State: fetched (dict url -> summary), last_url, last_content.
    Content is HTML-stripped and truncated for the Observation.
    """

    CONTENT_LIMIT = 6000
    FETCH_TIMEOUT_S = 30

    def __init__(self) -> None:
        self.fetched: Dict[str, str] = {}
        self.last_url: str = ""
        self.last_content: str = ""

    # -- Env protocol -----------------------------------------------------
    def observe(self) -> Observation:
        lines = [f"fetched {len(self.fetched)} url(s)"]
        if self.fetched:
            lines.append("urls:")
            for u in list(self.fetched)[-10:]:
                lines.append(f"  - {u}")
        if self.last_url:
            lines.append("")
            lines.append(f"last: {self.last_url}")
            lines.append("content preview:")
            lines.append(self.last_content[: self.CONTENT_LIMIT])
        else:
            lines.append("nothing fetched yet — use: fetch <url>")
        return Observation(
            text="\n".join(lines),
            info={"fetched": list(self.fetched)},
        )

    def act(self, action: str) -> ActResult:
        action = action.strip()
        if not action:
            return ActResult(progressed=False, info={"error": "empty action"})
        parts = action.split(None, 1)
        verb = parts[0].lower()

        if verb == "done":
            return ActResult(progressed=True, done=True)
        if verb != "fetch" or len(parts) < 2:
            return ActResult(
                progressed=False,
                info={"error": "unknown action; use: fetch <url>"},
            )

        url = parts[1].strip()
        if not re.match(r"^https?://", url, re.IGNORECASE):
            url = "http://" + url
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "SystemOne/1.0"}
            )
            with urllib.request.urlopen(req, timeout=self.FETCH_TIMEOUT_S) as resp:
                raw = resp.read(200_000).decode("utf-8", errors="replace")
            text = re.sub(r"<script.*?</script>", " ", raw, flags=re.S | re.I)
            text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            summary = text[: self.CONTENT_LIMIT]
            self.fetched[url] = summary
            self.last_url = url
            self.last_content = summary
            return ActResult(progressed=True, info={"url": url, "chars": len(summary)})
        except Exception as e:  # noqa: BLE001 - network errors are all "failed"
            return ActResult(
                progressed=False, info={"error": f"fetch failed: {e}"}
            )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _stub_decide(questions):
    """Offline stand-in: always picks the first option at 0.9 confidence."""
    out = {}
    for q in questions:
        name = q.get("name", "q")
        qtype = q.get("type", "choice")
        if qtype == "choice":
            opts = q.get("options", ["noop"])
            out[name] = {"option": opts[0], "confidence": 0.9}
        elif qtype == "noul":
            out[name] = {"option": "false", "confidence": 0.9}
        elif qtype == "score":
            out[name] = {"option": "medium", "confidence": 0.9}
    return out


def demo() -> None:
    """Run each Env through a few manual act/observe ticks (no model needed)."""
    import tempfile

    print("=== FileSystemEnv ===")
    with tempfile.TemporaryDirectory() as tmp:
        fs = FileSystemEnv(tmp)
        print(fs.act("write hello.txt hello from systemone").info)
        print(fs.act("read hello.txt").info)
        print(fs.observe().text.splitlines()[0])
        print("done tick:", fs.act("done").done)

    print("\n=== ShellEnv ===")
    sh = ShellEnv()
    print("echo:", sh.act("echo hi").info)
    print("blocked:", sh.act("rm -rf /").info)
    print("blocked:", sh.act("sudo ls").info)
    print(sh.observe().text.splitlines()[1])

    print("\n=== WebFetchEnv ===")
    wf = WebFetchEnv()
    r = wf.act("fetch https://example.com")
    print("fetch:", r.info)
    print(wf.observe().text.splitlines()[0])


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        print("usage: python -m systemone.envs demo")
