"""Run the solver over a seeded sample of benchmark tasks and write the results.

    python -m src.evaluate                # 30 tasks, seed 42
    python -m src.evaluate --n 5          # quick smoke run
    python -m src.evaluate --ids Address.000002 Number.000087

Full transcripts (prompts, replies, every candidate and what it produced) go to
runs/<run-name>/, which is gitignored because they contain benchmark examples.
Only per-task pass/fail records and the summary go to results/.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

from .dataset import DEFAULT_ROOT, list_task_ids, load_task, sample_task_ids
from .llm import LLMClient
from .report import RESULTS_DIR, write_summary
from .sandbox import DEFAULT_TIMEOUT
from .solver import MAX_REVISIONS, N_CANDIDATES, solve

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=30, help="number of tasks to sample")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed")
    ap.add_argument("--ids", nargs="*", help="run these task ids instead of a sample")
    ap.add_argument("--model", help="override LLM_MODEL from .env")
    ap.add_argument("--run-name", default="main", help="subfolder of runs/ for transcripts")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached LLM responses")
    ap.add_argument("--no-write", action="store_true", help="don't overwrite results/ (for experiments)")
    args = ap.parse_args(argv)

    ids = args.ids or sample_task_ids(args.n, args.seed)
    llm = LLMClient.from_env(model=args.model, use_cache=not args.no_cache)
    run_dir = RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model {llm.model} at {urlparse(llm.base_url).hostname}; {len(ids)} tasks", flush=True)

    results = []
    for i, task_id in enumerate(ids, 1):
        task = load_task(task_id)
        t = solve(task, llm)
        (run_dir / f"{task_id}.json").write_text(json.dumps(t, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
        passed_flags = t.get("survivors_passing_held_out") or []
        results.append({
            "benchmark_id": task_id,
            "features": task.features,
            "passed": t["passed"],
            "round_solved": t["round_solved"],
            "synthetic": task.synthetic,
            "held_out_input_seen_in_given": t["held_out_input_seen_in_given"],
            "n_survivors": t.get("n_survivors", 0),
            "any_survivor_passes": any(passed_flags),
            "rounds_used": len(t["rounds"]),
            "unparsed_rounds": sum(not r["parsed"] for r in t["rounds"]),
            "candidates_run": sum(len(r["candidates"]) for r in t["rounds"]),
            "candidates_repaired": sum(c["repaired_encoding"] for r in t["rounds"] for c in r["candidates"]),
        })
        mark = "PASS" if t["passed"] else "fail"
        print(f"[{i:>2}/{len(ids)}] {mark}  round={t['round_solved']!s:<8} {task_id}  "
              f"({', '.join(task.features)})", flush=True)

    passed = sum(r["passed"] for r in results)
    print(f"\nPassed {passed}/{len(results)}")
    if args.no_write:
        return

    meta = {
        "model": llm.model,
        "endpoint_host": urlparse(llm.base_url).hostname,
        "date": date.today().isoformat(),
        "seed": args.seed if not args.ids else None,
        "n_requested": len(ids) if args.ids else args.n,
        "n_available": len(list_task_ids(DEFAULT_ROOT)),
        "n_candidates": N_CANDIDATES,
        "max_revisions": MAX_REVISIONS,
        "timeout": DEFAULT_TIMEOUT,
        "usage": asdict(llm.usage),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (RESULTS_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {write_summary(results, meta)}")


if __name__ == "__main__":
    sys.exit(main())
