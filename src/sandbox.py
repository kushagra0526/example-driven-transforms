"""Execute untrusted candidate functions.

Three layers, none of which is trusted alone:
  1. A static AST check rejects code that reaches for dunder attributes, frame and
     code objects, or names like eval/open before anything runs.
  2. The candidate runs in a fresh Python process (isolated mode, no site-packages,
     empty environment, throwaway working directory) with a builtins whitelist and
     an import hook that only serves filtered views of pure-computation modules.
  3. The process is killed after a hard timeout, and on POSIX also gets CPU,
     memory, and file-size limits.

This is containment for code that is usually just wrong, not a hardened boundary
against a determined attacker. For that, run it inside a container or VM.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

CHILD_SCRIPT = Path(__file__).with_name("_sandbox_child.py")
DEFAULT_TIMEOUT = 2.0

FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "input", "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr", "breakpoint", "help", "exit", "quit", "memoryview",
}

# Attributes that lead from ordinary objects to frames, code, or interpreter state.
FORBIDDEN_ATTRS = {
    "gi_frame", "gi_code", "gi_yieldfrom", "cr_frame", "cr_code", "cr_await",
    "ag_frame", "ag_code", "ag_await", "f_back", "f_globals", "f_locals", "f_builtins",
    "f_code", "tb_frame", "tb_next", "mro",
}


@dataclass
class CaseResult:
    value: str | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class RunResult:
    """Outcome of running one candidate over a batch of inputs."""

    status: str  # "ok", "rejected", "load_error", "timeout", "crash"
    detail: str | None = None
    cases: list[CaseResult] = field(default_factory=list)


def static_check(code: str) -> str | None:
    """Return a reason to reject the code without running it, or None if it looks fine."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRS:
                return f"uses forbidden attribute '.{node.attr}'"
        elif isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in FORBIDDEN_NAMES:
                return f"uses forbidden name '{node.id}'"
        elif isinstance(node, ast.ClassDef):
            return "defines a class"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("_"):
                    return f"imports private module '{alias.name}'"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").startswith("_"):
                return f"imports private module '{node.module}'"
            if any(a.name == "*" for a in node.names):
                return "uses a wildcard import"
    if not any(isinstance(n, ast.FunctionDef) and n.name == "transform" for n in tree.body):
        return "does not define a top-level function named 'transform'"
    return None


def _posix_limits():  # pragma: no cover - not exercised on Windows
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 2**20, 512 * 2**20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))


def _child_env() -> dict[str, str]:
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0"}
    if os.name == "nt":
        # Windows needs SYSTEMROOT for basic runtime initialisation.
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    return env


def run_candidate(code: str, inputs: list[tuple[str | None, ...]],
                  timeout: float = DEFAULT_TIMEOUT) -> RunResult:
    """Run `transform(*cols)` for every entry of `inputs` in one sandboxed process."""
    reason = static_check(code)
    if reason:
        return RunResult("rejected", reason)

    payload = json.dumps({"code": code, "inputs": [list(i) for i in inputs]},
                         ensure_ascii=True).encode("utf-8")
    with tempfile.TemporaryDirectory(prefix="cand_") as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", str(CHILD_SCRIPT)],
                input=payload,
                capture_output=True,
                timeout=timeout,
                cwd=workdir,
                env=_child_env(),
                preexec_fn=_posix_limits if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return RunResult("timeout", f"did not finish within {timeout:g}s")

    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["no output"]
        return RunResult("crash", f"process exited with code {proc.returncode}: {tail[0][:300]}")
    try:
        reply = json.loads(proc.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return RunResult("crash", "sandbox process returned unreadable output")

    if reply["load_error"]:
        return RunResult("load_error", reply["load_error"])
    cases = [CaseResult(r["value"], r["error"]) for r in reply["results"]]
    return RunResult("ok", None, cases)
