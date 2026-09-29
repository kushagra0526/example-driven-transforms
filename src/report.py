"""Turn per-task results into results/summary.md.

Can be re-run on its own: python -m src.report
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _pct(k: int, n: int) -> str:
    return f"{100 * k / n:.1f}%" if n else "n/a"


def build_summary(results: list[dict], meta: dict) -> str:
    n = len(results)
    passed = sum(r["passed"] for r in results)
    lines = [
        "# Evaluation summary",
        "",
        f"- Model: `{meta['model']}` via `{meta['endpoint_host']}`",
        f"- Date: {meta['date']}",
        f"- Benchmark subset: `Transformation.Text`, {meta['n_available']} tasks available",
        f"- Sample size: **{n}** tasks, drawn with `random.Random({meta['seed']}).sample(sorted(task_ids), {meta['n_requested']})`",
        f"- Per task: {meta['n_candidates']} candidates per round, up to {meta['max_revisions']} revision rounds, "
        f"{meta['timeout']:g}s execution timeout",
        f"- Protocol: all examples but the last are given; the last is held out and only used for the final check",
        "",
        f"## Overall accuracy: **{passed}/{n} ({_pct(passed, n)})**",
        "",
        "A task counts as passed only if the first candidate that reproduced every given example "
        "also produces the held-out output exactly.",
        "",
    ]

    # Where in the loop things were decided.
    by_round = defaultdict(lambda: [0, 0])
    for r in results:
        key = str(r["round_solved"])
        by_round[key][0] += 1
        by_round[key][1] += r["passed"]
    lines += ["## By round", "",
              "| Round a survivor was found | Tasks | Passed held-out |", "|---|---:|---:|"]
    labels = {"0": "0 (first generation)", "1": "1 (first revision)", "2": "2 (second revision)",
              "unsolved": "never (no candidate fit the given examples)"}
    for key in ["0", "1", "2", "unsolved"]:
        if key in by_round:
            t, p = by_round[key]
            lines.append(f"| {labels[key]} | {t} | {p} |")
    lines.append("")

    # Feature tags; a task counts toward each of its tags.
    by_feat = defaultdict(lambda: [0, 0])
    for r in results:
        for f in r["features"]:
            by_feat[f][0] += 1
            by_feat[f][1] += r["passed"]
    lines += ["## Accuracy by feature tag", "",
              "A task carries several tags and counts toward each, so rows overlap.", "",
              "| Feature | Tasks | Passed | Accuracy |", "|---|---:|---:|---:|"]
    for f, (t, p) in sorted(by_feat.items(), key=lambda kv: (-kv[1][0], kv[0])):
        lines.append(f"| {f} | {t} | {p} | {_pct(p, t)} |")
    lines.append("")

    # Things a careful reader would want to know about the numbers.
    fit = [r for r in results if r["round_solved"] != "unsolved"]
    overfit = [r for r in fit if not r["passed"]]
    rescued = [r for r in overfit if r.get("any_survivor_passes")]
    leaked = [r for r in results if r["held_out_input_seen_in_given"]]
    clean = [r for r in results if not r["held_out_input_seen_in_given"]]
    lines += [
        "## Caveats and diagnostics", "",
        f"- Fit the given examples but failed the held-out one: **{len(overfit)}** of {len(fit)} tasks where a survivor was found.",
        f"- Of those, tasks where a *different* survivor would have passed: {len(rescued)}. "
        "This is a diagnostic about ambiguity between surviving candidates, not a result: the method commits to the first survivor before looking at the held-out example.",
        f"- Tasks whose held-out input also appears among the given examples: {len(leaked)}. "
        f"Accuracy excluding them: {sum(r['passed'] for r in clean)}/{len(clean)} ({_pct(sum(r['passed'] for r in clean), len(clean))}).",
        f"- LLM usage: {meta['usage']['calls']} requests, "
        f"{meta['usage']['prompt_tokens']:,} prompt + {meta['usage']['completion_tokens']:,} completion tokens "
        f"({meta['usage']['cached_calls']} of the requests replayed from the local response cache on the final run).",
        "",
        "## Sampled task IDs", "",
        "Folder names only; the benchmark data itself is not included in this repository.", "",
        ", ".join(f"`{r['benchmark_id']}`" for r in results),
        "",
    ]
    return "\n".join(lines)


def write_summary(results: list[dict], meta: dict, out_dir: Path = RESULTS_DIR) -> Path:
    path = out_dir / "summary.md"
    path.write_text(build_summary(results, meta), encoding="utf-8")
    return path


if __name__ == "__main__":
    results = json.loads((RESULTS_DIR / "results.json").read_text(encoding="utf-8"))
    meta = json.loads((RESULTS_DIR / "run_meta.json").read_text(encoding="utf-8"))
    print(write_summary(results, meta))
