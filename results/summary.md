# Evaluation summary

- Model: `openai/gpt-oss-120b` via `api.groq.com`
- Date: 2026-09-29
- Benchmark subset: `Transformation.Text`, 354 tasks available
- Sample size: **30** tasks, drawn with `random.Random(42).sample(sorted(task_ids), 30)`
- Per task: 5 candidates per round, up to 2 revision rounds, 2s execution timeout
- Protocol: all examples but the last are given; the last is held out and only used for the final check

## Overall accuracy: **28/30 (93.3%)**

A task counts as passed only if the first candidate that reproduced every given example also produces the held-out output exactly.

## By round

| Round a survivor was found | Tasks | Passed held-out |
|---|---:|---:|
| 0 (first generation) | 27 | 27 |
| 1 (first revision) | 1 | 1 |
| never (no candidate fit the given examples) | 2 | 0 |

## Accuracy by feature tag

A task carries several tags and counts toward each, so rows overlap.

| Feature | Tasks | Passed | Accuracy |
|---|---:|---:|---:|
| DateTime | 15 | 13 | 86.7% |
| Substring | 11 | 11 | 100.0% |
| DateTimeRounding | 7 | 7 | 100.0% |
| DateTimeRange | 5 | 5 | 100.0% |
| Conditional | 3 | 3 | 100.0% |
| Multicolumn | 3 | 3 | 100.0% |
| Numeric | 3 | 3 | 100.0% |
| Concatenation | 2 | 2 | 100.0% |
| NumericRounding | 2 | 2 | 100.0% |

## Caveats and diagnostics

- Fit the given examples but failed the held-out one: **0** of 28 tasks where a survivor was found.
- Of those, tasks where a *different* survivor would have passed: 0. This is a diagnostic about ambiguity between surviving candidates, not a result: the method commits to the first survivor before looking at the held-out example.
- Tasks whose held-out input also appears among the given examples: 0. Accuracy excluding them: 28/30 (93.3%).
- Model replies that held no parseable candidate list: 2 of 35 generation rounds. Each still counts as a used round.
- Candidates the model JSON-escaped twice (one long line of literal `\n`), repaired by decoding once more: 78 of 159 executed.
- LLM usage: 35 requests, 27,410 prompt + 95,732 completion tokens (0 of the requests replayed from the local response cache on the final run).

## Sampled task IDs

Folder names only; the benchmark data itself is not included in this repository.

`Product.000003`, `DateTime.000021`, `Address.000013`, `DateTime.000104`, `DateTime.000089`, `DateTime.000078`, `DateTime.000035`, `DateTime.000016`, `UserAgent.000002`, `Number.000066`, `DateTime.000008`, `Number.000089`, `Name.000042`, `BillingCode.000001`, `Author.000001`, `DateTime.000011`, `DateTime.000075`, `DateTime.000083`, `Number.000045`, `Phone.000001`, `Address.000014`, `Number.000074`, `DateTime.000065`, `ShippingCode.000002`, `Name.000040`, `DateTime.000076`, `Number.000016`, `Number.000088`, `DateTime.000106`, `Address.000004`
