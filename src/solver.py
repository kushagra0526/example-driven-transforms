"""Generate candidate functions, rank them by execution, and iterate on failure."""

from __future__ import annotations

from .dataset import Example, Task
from .llm import LLMClient
from .prompts import generation_messages, parse_candidates, revision_messages, undo_double_encoding
from .sandbox import DEFAULT_TIMEOUT, run_candidate

N_CANDIDATES = 5
MAX_REVISIONS = 2
# Most tasks have far fewer given examples than this, and they all go in the
# prompt. For the few with dozens or hundreds, the prompt shows the first 15
# distinct ones; every given example is still used for ranking, and any that a
# candidate gets wrong is shown to the model in the revision round.
MAX_PROMPT_EXAMPLES = 15
MAX_CASES_IN_FEEDBACK = 3


def check_candidate(code: str, examples: list[Example], timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Run a candidate on the examples. It survives only if every output matches exactly."""
    run = run_candidate(code, [ex.inputs for ex in examples], timeout)
    record = {"code": code, "status": run.status, "detail": run.detail,
              "survived": False, "n_failed": None, "failures": []}
    if run.status != "ok":
        return record
    failures = [
        {"inputs": list(ex.inputs), "expected": ex.output, "got": case.value, "error": case.error}
        for ex, case in zip(examples, run.cases)
        if case.error is not None or case.value != ex.output
    ]
    record.update(survived=not failures, n_failed=len(failures),
                  failures=failures[:MAX_CASES_IN_FEEDBACK])
    return record


def _feedback(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        if r["status"] != "ok":
            label = {"rejected": "rejected before running", "load_error": "failed to load",
                     "timeout": "timed out", "crash": "crashed"}[r["status"]]
            out.append({"code": r["code"], "problem": f"{label}: {r['detail']}"})
        else:
            out.append({"code": r["code"], "cases": r["failures"]})
    return out


def _run_held_out(code: str, held_out: Example) -> dict:
    run = run_candidate(code, [held_out.inputs])
    got, error = None, run.detail
    if run.status == "ok":
        got, error = run.cases[0].value, run.cases[0].error
    return {"status": run.status, "got": got, "error": error,
            "passed": run.status == "ok" and error is None and got == held_out.output}


def solve(task: Task, llm: LLMClient, n_candidates: int = N_CANDIDATES,
          max_revisions: int = MAX_REVISIONS) -> dict:
    """Solve one task and return a full transcript of what happened."""
    given = task.unique_given()
    shown = given[:MAX_PROMPT_EXAMPLES]
    transcript = {
        "task_id": task.id,
        "features": task.features,
        "synthetic": task.synthetic,
        "n_given": len(task.given),
        "n_unique_given": len(given),
        "n_shown_in_prompt": len(shown),
        "held_out_input_seen_in_given": task.held_out_input_seen(),
        "rounds": [],
    }

    survivors: list[dict] = []
    last_records: list[dict] = []
    parse_note = None
    for rnd in range(max_revisions + 1):
        if rnd == 0:
            messages = generation_messages(shown, task.n_columns, n_candidates)
        else:
            messages = revision_messages(shown, task.n_columns, n_candidates,
                                         _feedback(last_records), parse_note)
        raw = llm.chat(messages, salt=f"{task.id}/round{rnd}")
        codes = parse_candidates(raw)
        round_log = {"round": rnd, "prompt": messages[-1]["content"], "response": raw,
                     "parsed": codes is not None, "candidates": []}
        transcript["rounds"].append(round_log)

        if codes is None:
            parse_note = "Your previous reply could not be parsed as the required JSON object."
            continue
        parse_note = None
        # Identical candidates would only repeat the same result.
        codes = list(dict.fromkeys(codes))[:n_candidates]
        records = []
        for code in codes:
            code, repaired = undo_double_encoding(code)
            records.append({**check_candidate(code, given), "repaired_encoding": repaired})
        round_log["candidates"] = records
        survivors = [r for r in records if r["survived"]]
        if survivors:
            break
        last_records = records

    if not survivors:
        transcript.update(round_solved="unsolved", passed=False, chosen_code=None, held_out=None)
        return transcript

    # Selection is fixed before the held-out example is touched: the first survivor wins.
    chosen = survivors[0]
    held = _run_held_out(chosen["code"], task.held_out)
    transcript.update(
        round_solved=rnd,
        passed=held["passed"],
        chosen_code=chosen["code"],
        held_out={"inputs": list(task.held_out.inputs), "expected": task.held_out.output, **held},
        n_survivors=len(survivors),
        # Diagnostic only, never used for selection: did any other survivor get it right?
        survivors_passing_held_out=[held["passed"]] + [
            _run_held_out(s["code"], task.held_out)["passed"] for s in survivors[1:]
        ],
    )
    return transcript
