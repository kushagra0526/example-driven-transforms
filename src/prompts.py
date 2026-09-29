"""Prompt construction and parsing of the model's candidate list."""

from __future__ import annotations

import json
import re
import warnings

from .dataset import Example

ALLOWED_MODULES_TEXT = ("re, string, math, datetime, calendar, decimal, fractions, itertools, "
                        "functools, collections, unicodedata, textwrap, time")

SYSTEM = (
    "You infer string transformations from input/output examples and implement them as "
    "small Python functions. You always reply with a single JSON object and nothing else."
)

RULES = f"""Rules for every candidate:
- Standard library only, and only these modules: {ALLOWED_MODULES_TEXT}. Put imports inside the source string.
- No file, network or OS access. No eval, exec, getattr, classes, or double-underscore attributes.
- Each candidate is a complete, self-contained Python source string that defines `transform`.
- Keep each candidate compact: no comments or docstrings, ideally under 25 lines.
- Implement a general rule. Do not hard-code the example outputs or build a lookup table from them.
- `transform` must return a str, or None where the examples show a null output."""

FORMAT = (
    'Reply with exactly this JSON shape: {"candidates": ["<python source>", ...]}\n'
    "JSON-escape each source string exactly once: a line break becomes \\n, and a single "
    "backslash in the Python code (for example in a regex) becomes \\\\."
)

MAX_CODE_CHARS_IN_FEEDBACK = 1500


def signature(n_columns: int) -> str:
    if n_columns == 1:
        return "def transform(x):  # x is a str or None"
    args = ", ".join(f"c{i}" for i in range(n_columns))
    return f"def transform({args}):  # each argument is a str or None"


def _fmt_inputs(inputs: tuple) -> str:
    if len(inputs) == 1:
        return json.dumps(inputs[0], ensure_ascii=False)
    return json.dumps(list(inputs), ensure_ascii=False)


def format_examples(examples: list[Example]) -> str:
    return "\n".join(
        f"{i}. {_fmt_inputs(ex.inputs)} -> {json.dumps(ex.output, ensure_ascii=False)}"
        for i, ex in enumerate(examples, 1)
    )


def generation_messages(examples: list[Example], n_columns: int, n_candidates: int) -> list[dict]:
    user = f"""Infer the transformation shown by these examples. Values are JSON-encoded, so quotes and whitespace are exact.

{format_examples(examples)}

Signature:
{signature(n_columns)}

Write {n_candidates} candidate implementations that take genuinely different approaches, for example: one built on regular expressions, one on str.split/str.join, one on index slicing, one that parses the value (dates, numbers) and re-formats it, one that handles cases explicitly. Where the examples leave the rule ambiguous, let different candidates bet on different readings.

{RULES}

{FORMAT}"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def revision_messages(examples: list[Example], n_columns: int, n_candidates: int,
                      failures: list[dict], note: str | None = None) -> list[dict]:
    """`failures` holds, per failed candidate: {"code", "problem"} or {"code", "cases": [...]}."""
    blocks = []
    for i, f in enumerate(failures, 1):
        code = f["code"].strip()
        if len(code) > MAX_CODE_CHARS_IN_FEEDBACK:
            code = code[:MAX_CODE_CHARS_IN_FEEDBACK] + "\n# ... (truncated)"
        lines = [f"### Candidate {i}", "```python", code, "```"]
        if f.get("problem"):
            lines.append(f"Problem: {f['problem']}")
        for c in f.get("cases", []):
            got = f"raised {c['error']}" if c["error"] else f"returned {json.dumps(c['got'], ensure_ascii=False)}"
            lines.append(f"- input {_fmt_inputs(tuple(c['inputs']))}: expected "
                         f"{json.dumps(c['expected'], ensure_ascii=False)}, {got}")
        blocks.append("\n".join(lines))
    previous = "\n\n".join(blocks) if blocks else "(no usable candidates)"
    note_text = f"\n{note}\n" if note else ""

    user = f"""Infer the transformation shown by these examples. Values are JSON-encoded, so quotes and whitespace are exact.

{format_examples(examples)}

Signature:
{signature(n_columns)}

Every candidate from the previous attempt failed on at least one known example. Here is each one with what it produced versus what was expected (some failing inputs may not be in the list above; they are still correct examples of the same rule):
{note_text}
{previous}

Study what the failures have in common, then write {n_candidates} revised candidates. Fix the misunderstanding rather than special-casing the failing inputs, and keep the approaches different from each other.

{RULES}

{FORMAT}"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_candidates(text: str) -> list[str] | None:
    """Pull the list of source strings out of the model's reply, or None if there isn't one."""
    data = None
    try:
        data = json.loads(text, strict=False)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0), strict=False)
            except json.JSONDecodeError:
                pass
    if data is None:
        data = _salvage_truncated(text)
    if isinstance(data, dict):
        data = data.get("candidates")
    if not isinstance(data, list):
        return None
    out = []
    for item in data:
        if isinstance(item, dict):  # some models wrap each candidate: {"code": "..."}
            item = item.get("code") or item.get("source")
        if isinstance(item, str) and item.strip():
            out.append(_strip_fence(item))
    return out or None


def undo_double_encoding(code: str) -> tuple[str, bool]:
    """Repair a candidate the model JSON-escaped twice.

    Then the decoded source is one long line holding literal "\\n" sequences,
    which can never compile. Only that unambiguous case is touched: code that
    already has real line breaks is returned unchanged.
    """
    if "\n" in code or "\\n" not in code:
        return code, False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # unknown escapes are kept as-is
        # backslashreplace protects non-ASCII text from the latin-1 round trip.
        fixed = code.encode("latin-1", "backslashreplace").decode("unicode_escape")
    return fixed, fixed != code


def _salvage_truncated(text: str) -> list[str] | None:
    """Recover the complete strings from a reply cut off partway through the list.

    A reply that hits the token limit mid-way through candidate 4 still holds
    three usable candidates; throwing them away would waste a whole round.
    """
    start = re.search(r'"candidates"\s*:\s*\[', text)
    if not start:
        return None
    decoder = json.JSONDecoder(strict=False)  # tolerate raw newlines inside strings
    pos, out = start.end(), []
    while True:
        while pos < len(text) and text[pos] in " \t\r\n,":
            pos += 1
        try:
            item, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        out.append(item)
    return out or None


def _strip_fence(code: str) -> str:
    code = code.strip()
    m = re.fullmatch(r"```(?:python)?\s*\n(.*?)\n?```", code, re.DOTALL)
    return m.group(1) if m else code
